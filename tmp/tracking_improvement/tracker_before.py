from dataclasses import dataclass

import cv2
import numpy as np

from detection.x_detector import detect_x
from geometry.homography import estimate_homography, order_corners, transform_corners, rectify_pad


@dataclass
class TrackingOutput:
    valid: bool
    corners: np.ndarray | None
    num_points: int
    inlier_ratio: float
    mean_lk_error: float
    reason: str = ''


class LandingPadTracker:
    """Sparse LK tracker. Public corners use input-frame pixels.

    Initialize from a detection on the SAME frame, then update once per subsequent
    frame, in order. Features/H/RANSAC use a smaller grayscale working image.
    """

    def __init__(self, *, max_width=1280, min_points=8, max_points=150,
                 min_inlier_ratio=.5, max_fb_error=1.5, max_lk_error=25.):
        if (max_width < 100 or not 4 <= min_points <= max_points or
                not 0 < min_inlier_ratio <= 1 or max_fb_error <= 0 or max_lk_error <= 0):
            raise ValueError('Invalid tracker limits.')
        self.max_width = max_width
        self.min_points = min_points
        self.max_points = max_points
        self.min_inlier_ratio = min_inlier_ratio
        self.max_fb_error = max_fb_error
        self.max_lk_error = max_lk_error
        self.reset()

    def reset(self):
        self.prev_gray = None
        self.prev_points = None
        self.corners = None
        self.initialized = False
        self.frame_size = None
        self.scale_xy = None
        self.age = 0
        self.reference_points = None
        self.reference_corners = None

    def _gray(self, frame):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                frame.ndim not in (2, 3) or min(frame.shape[:2]) < 24):
            raise ValueError('Expected a nonempty uint8 image.')
        if frame.ndim == 3:
            if frame.shape[2] not in (3, 4):
                raise ValueError('Expected grayscale, BGR or BGRA.')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY if frame.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        height, width = frame.shape
        if width > self.max_width:
            frame = cv2.resize(frame, (self.max_width, max(1, round(height*self.max_width/width))),
                               interpolation=cv2.INTER_AREA)
        return frame

    @staticmethod
    def _mask(gray, corners):
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255)
        return mask

    def _features(self, gray, corners, existing=None):
        mask = self._mask(gray, corners)
        count = self.max_points - (0 if existing is None else len(existing))
        if count <= 0:
            return existing
        # Measure quality against all pad features even when replenishing.
        # Masking existing strong points first would promote weak edge points.
        new = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_points, qualityLevel=.005,
                                     minDistance=5, mask=mask, blockSize=3, useHarrisDetector=True)
        if new is None:
            return existing
        if existing is not None:
            distances = np.linalg.norm(new.reshape(-1, 1, 2)-existing.reshape(1, -1, 2), axis=2)
            # Replenish only inside the paper, not on an uncertain moving edge
            # where fabric features could gradually replace the original anchors.
            interior = cv2.erode(mask, np.ones((9, 9), np.uint8))
            xy = np.round(new.reshape(-1, 2)).astype(int)
            new = new[(distances.min(axis=1) >= 5) & (interior[xy[:, 1], xy[:, 0]] > 0)][:count]
        return new if existing is None else np.concatenate((existing, new))

    @staticmethod
    def _spread(points, corners):
        if points is None or len(points) < 4:
            return 0.
        return cv2.contourArea(cv2.convexHull(points)) / max(cv2.contourArea(corners), 1.)

    @staticmethod
    def _inside(corners, shape):
        height, width = shape
        return (np.isfinite(corners).all() and np.all(corners >= 1) and
                np.all(corners[:, 0] < width-1) and np.all(corners[:, 1] < height-1))

    def _lost(self, reason, num_points=0, ratio=0., error=float('inf')):
        self.reset()
        return TrackingOutput(False, None, num_points, ratio, error, reason)

    def initialize(self, frame, corners) -> bool:
        """Choose Harris corners inside a verified pad; return success."""
        self.reset()
        try:
            gray = self._gray(frame)
            corners = order_corners(corners)
        except (ValueError, TypeError, cv2.error):
            return False
        height, width = frame.shape[:2]
        scale_xy = np.float32([width/gray.shape[1], height/gray.shape[0]])
        work_corners = corners / scale_xy
        if not self._inside(work_corners, gray.shape):
            return False
        points = self._features(gray, work_corners)
        if points is None or len(points) < self.min_points or self._spread(points, work_corners) < .05:
            return False
        self.prev_gray = gray.copy()
        self.prev_points = points
        self.reference_points = points.copy()
        self.reference_corners = work_corners.copy()
        self.corners = corners.copy()
        self.frame_size = (width, height)
        self.scale_xy = scale_xy
        self.initialized = True
        return True

    def update(self, frame) -> TrackingOutput:
        if not self.initialized:
            return self._lost('not_initialized')
        try:
            gray = self._gray(frame)
        except (ValueError, cv2.error):
            return self._lost('invalid_frame')
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            return self._lost('frame_size_changed')

        # Forward/backward LK: a reliable match returns close to its origin.
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        current, status, errors = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None, **lk)
        if current is None or status is None or errors is None:
            return self._lost('optical_flow_failed')
        keep = ((status.ravel() == 1) & np.isfinite(current).all(axis=(1, 2)) &
                np.isfinite(errors.ravel()) & (errors.ravel() <= self.max_lk_error))
        before, after, errors = self.prev_points[keep], current[keep], errors.ravel()[keep]
        reference = self.reference_points[keep]
        if len(after) < self.min_points:
            return self._lost('too_few_lk_points', len(after))
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, after, None, **lk)
        if back is None or status_back is None:
            return self._lost('backward_flow_failed')
        keep = ((status_back.ravel() == 1) &
                (np.linalg.norm(back-before, axis=(1, 2)) <= self.max_fb_error))
        before, after, errors = before[keep], after[keep], errors[keep]
        reference = reference[keep]
        if len(after) < self.min_points:
            return self._lost('too_few_consistent_points', len(after))

        # Fit the current pose of the plane relative to its initialization.
        # Multiplying a new noisy homography every frame accumulates corner drift.
        H, inliers = estimate_homography(reference, after, reprojection_threshold=1.5)
        if H is None:
            return self._lost('homography_failed', len(after))
        ratio, count = float(inliers.mean()), int(inliers.sum())
        mean_error = float(errors[inliers].mean()) if count else float('inf')
        if count < self.min_points or ratio < self.min_inlier_ratio:
            return self._lost('too_few_inliers', count, ratio, mean_error)
        old_corners = self.corners / self.scale_xy
        if (self._spread(before[inliers], old_corners) < .20 or
                self._spread(reference[inliers], self.reference_corners) < .20):
            return self._lost('points_not_spread_over_pad', count, ratio, mean_error)
        try:
            new_corners = transform_corners(self.reference_corners, H)
        except ValueError:
            return self._lost('invalid_projection', count, ratio, mean_error)
        old_area = cv2.contourArea(old_corners, oriented=True)
        new_area = cv2.contourArea(new_corners, oriented=True)
        if (not self._inside(new_corners, gray.shape) or not cv2.isContourConvex(new_corners) or
                not .6 <= new_area/old_area <= 1.6 or
                np.linalg.norm(new_corners-old_corners, axis=1).max() > .4*np.sqrt(abs(old_area))):
            return self._lost('implausible_quad', count, ratio, mean_error)

        # Cheap verification of the tracked crop, not full-frame redetection.
        # LK consensus alone can follow background texture after gradual drift.
        if (self.age+1) % 10 == 0:
            if not detect_x(rectify_pad(frame, new_corners*self.scale_xy, (280, 396)))[0]:
                return self._lost('x_verification_failed', count, ratio, mean_error)

        points = after[inliers]
        reference = reference[inliers]
        mask = self._mask(gray, new_corners)
        # Keep already tracked paper-edge features despite subpixel boundary
        # rounding. Dropping these anchors causes the estimated corners to drift.
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        xy = np.round(points.reshape(-1, 2)).astype(int)
        inside = ((xy[:, 0] >= 0) & (xy[:, 0] < gray.shape[1]) &
                  (xy[:, 1] >= 0) & (xy[:, 1] < gray.shape[0]))
        indices = np.flatnonzero(inside)
        selected = indices[mask[xy[indices, 1], xy[indices, 0]] > 0]
        points, reference = points[selected], reference[selected]
        if len(points) < self.min_points or self._spread(points, new_corners) < .05:
            return self._lost('insufficient_pad_support', len(points), ratio, mean_error)
        self.age += 1
        if self.age % 10 == 0:
            old_count = len(points)
            points = self._features(gray, new_corners, points)
            if len(points) > old_count:
                added_reference = cv2.perspectiveTransform(points[old_count:], np.linalg.inv(H))
                reference = np.concatenate((reference, added_reference))
        self.prev_gray = gray.copy()
        self.prev_points = points
        self.reference_points = reference
        self.corners = (new_corners * self.scale_xy).astype(np.float32)
        return TrackingOutput(True, self.corners.copy(), count, ratio, mean_error, 'tracked')

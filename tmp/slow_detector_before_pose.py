from dataclasses import dataclass
import cv2
import numpy as np

from detection.x_detector import detect_x
from geometry.homography import order_corners, rectify_pad


@dataclass
class DetectionOutput:
    valid: bool
    corners: np.ndarray | None
    confidence: float


class SlowDetector:
    def __init__(self, *, max_width=1280, min_area_ratio=0.0005,
                 min_area_px=180, max_candidates=100):
        if max_width < 100 or not 0 < min_area_ratio < 1 or min_area_px <= 0 or max_candidates < 1:
            raise ValueError("Invalid detector size, area or candidate limits.")
        self.max_width = int(max_width)
        self.min_area_ratio = min_area_ratio
        self.min_area_px = min_area_px
        self.max_candidates = int(max_candidates)
        self.debug_images = {}
        self.candidate_diagnostics = []
        self.last_candidate_count = 0

    def detect(self, frame, *, debug=False) -> DetectionOutput:
        """Find one visible light sheet containing a large dark X.

        Corners are float32 (4, 2), clockwise, in the supplied frame's pixels.
        If input is undistorted, output is in undistorted coordinates too.
        Detection alone cannot prove paper dimensions or physical orientation.
        debug_images is refreshed per call and is intended for a single worker.
        """
        self.debug_images = {}
        self.candidate_diagnostics = []
        self.last_candidate_count = 0
        invalid = DetectionOutput(False, None, 0.0)
        if (not isinstance(frame, np.ndarray) or frame.size == 0 or
                frame.dtype != np.uint8 or frame.ndim not in (2, 3)):
            return invalid
        if frame.ndim == 3 and frame.shape[2] not in (3, 4):
            return invalid
        height, width = frame.shape[:2]
        if min(height, width) < 24:
            return invalid
        scale = min(1., self.max_width/width)
        work = cv2.resize(frame, (round(width*scale), max(1, round(height*scale)))) if scale < 1 else frame
        if work.ndim == 3:
            code = cv2.COLOR_BGR2GRAY if work.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
            gray = cv2.cvtColor(work, code)
        else:
            gray = work
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # A complementary region mask helps when the paper has a weak edge.
        _, paper = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if debug:
            self.debug_images.update(gray=gray, edges=edges, paper_mask=paper)

        masks = [edges, paper]
        if work.ndim == 3:
            # Near-neutral paper can merge with equally bright colored fabric
            # in grayscale. This is an additional cue, not a required color:
            # shaded/tinted paper still has the edge and intensity routes.
            hsv = cv2.cvtColor(work[:, :, :3], cv2.COLOR_BGR2HSV)
            neutral = ((hsv[:, :, 1] < 10) & (hsv[:, :, 2] > 100)).astype(np.uint8) * 255
            neutral = cv2.morphologyEx(neutral, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            masks.append(neutral)
            if debug:
                self.debug_images['neutral_paper_mask'] = neutral

        minimum = max(self.min_area_px, gray.size*self.min_area_ratio)
        contours = []
        for mask in masks:
            found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(found)
        contours.sort(key=cv2.contourArea, reverse=True)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not minimum <= area <= 0.97*gray.size:
                continue
            perimeter = cv2.arcLength(contour, True)
            for epsilon in (0.02, 0.035):
                polygon = cv2.approxPolyDP(contour, epsilon*perimeter, True)
                if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                    continue
                corners = order_corners(polygon)
                if np.linalg.norm(corners-np.roll(corners, 1, axis=0), axis=1).min() < 12:
                    continue
                # Reject clipped borders: we need four observed paper corners.
                if (corners[:, 0].min() < 2 or corners[:, 1].min() < 2 or
                        corners[:, 0].max() > gray.shape[1]-3 or corners[:, 1].max() > gray.shape[0]-3):
                    continue
                if any(min(np.linalg.norm(corners-np.roll(old, shift, axis=0), axis=1).mean()
                           for shift in range(4)) < 4 for old in candidates):
                    continue
                candidates.append(corners)
                break
            if len(candidates) >= self.max_candidates:
                break

        self.last_candidate_count = len(candidates)
        best = invalid
        scale_xy = np.array([width/gray.shape[1], height/gray.shape[0]], dtype=np.float32)
        for index, corners in enumerate(candidates):
            original_corners = corners * scale_xy
            rectified = rectify_pad(frame, original_corners, output_size=(280, 396))
            details = {} if debug else None
            valid, confidence = detect_x(rectified, diagnostics=details)
            if debug:
                self.debug_images[f'candidate_{index:03d}'] = rectified
                self.candidate_diagnostics.append(dict(candidate=index, valid=valid,
                    confidence=confidence, corners=original_corners.tolist(), x_check=details))
            if valid and confidence > best.confidence:
                best = DetectionOutput(True, original_corners.copy(), confidence)
                if debug:
                    self.debug_images['rectified'] = rectified
        return best

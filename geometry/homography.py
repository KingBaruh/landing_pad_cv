import cv2
import numpy as np


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Return four convex vertices clockwise in image coordinates.

    Start at the smallest x+y (tie: y, then x). Unlike assigning corners with
    four separate min/max operations, this keeps four distinct points even for
    a diamond. This is an image ordering, not a physical A4 orientation label.
    """
    points = np.asarray(corners, dtype=np.float32)
    if points.size != 8 or not np.isfinite(points).all():
        raise ValueError("Expected four finite 2D corners.")
    points = points.reshape(4, 2)
    if len(np.unique(points, axis=0)) != 4:
        raise ValueError("Corners must be distinct.")
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    if not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < 1e-3:
        raise ValueError("Corners must form a non-degenerate convex quadrilateral.")
    start = np.lexsort((points[:, 0], points[:, 1], points.sum(axis=1)))[0]
    return np.roll(points, -int(start), axis=0).copy()


def estimate_homography(prev_points, curr_points, reprojection_threshold=3.0):
    """Return (H, boolean inliers), or (None, None) for insufficient geometry.

    Points and the RANSAC threshold must use the same pixel coordinate system.
    """
    before = np.asarray(prev_points, dtype=np.float32).reshape(-1, 2)
    after = np.asarray(curr_points, dtype=np.float32).reshape(-1, 2)
    if len(before) != len(after) or reprojection_threshold <= 0:
        raise ValueError('Point counts must match and the threshold must be positive.')
    if len(before) < 4 or not np.isfinite(before).all() or not np.isfinite(after).all():
        return None, None
    if any(np.linalg.matrix_rank(p-p.mean(axis=0)) < 2 for p in (before, after)):
        return None, None
    H, mask = cv2.findHomography(before, after, cv2.RANSAC, reprojection_threshold)
    if H is None or mask is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        return None, None
    return H, mask.ravel().astype(bool)


def transform_corners(corners, H):
    """Transform corners without reordering their identities between frames."""
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.isfinite(points).all():
        raise ValueError('Expected finite corners and a 3x3 homography.')
    homogeneous = np.column_stack((points, np.ones(4))) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-8):
        raise ValueError('A corner projects to infinity.')
    return cv2.perspectiveTransform(points[:, None, :], matrix).reshape(4, 2)


def rectify_pad(frame, corners, output_size=(420, 594)):
    """Warp a quadrilateral to a canonical portrait view (width, height).

    This warp assumes a candidate is the pad; it does not prove its A4 ratio.
    A symmetric X alone cannot determine a unique physical corner orientation.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Cannot rectify an empty image.")
    if (len(output_size) != 2 or
            any(not isinstance(n, (int, np.integer)) or n < 2 for n in output_size)):
        raise ValueError("output_size must contain integer width and height >= 2.")
    width, height = output_size
    source = order_corners(corners)
    target = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]],
                      dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(frame, matrix, (width, height))

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class PoseOutput:
    valid: bool
    position_xyz: np.ndarray | None
    distance_m: float | None
    orientation_rpy: np.ndarray | None
    reprojection_error_px: float | None
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    reason: str = ''
    corner_indices: np.ndarray | None = None
    orientation_ambiguous: bool = False
    yaw_ambiguous_180: bool = True


A4_OBJECT_POINTS = np.array(
    [
        [-0.105, -0.1485, 0.0],
        [ 0.105, -0.1485, 0.0],
        [ 0.105,  0.1485, 0.0],
        [-0.105,  0.1485, 0.0],
    ],
    dtype=np.float64,
)


def rotation_to_rpy(rotation):
    """Radians, object-to-camera rotation R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

    At gimbal lock use yaw=0; Euler angles are not unique there.
    """
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = np.hypot(R[0, 0], R[1, 0])
    pitch = np.arctan2(-R[2, 0], sy)
    if sy > 1e-8:
        roll, yaw = np.arctan2(R[2, 1], R[2, 2]), np.arctan2(R[1, 0], R[0, 0])
    else:
        roll, yaw = np.arctan2(-R[1, 2], R[1, 1]), 0.
    return np.array([roll, pitch, yaw])


def _rotation_distance(first, second):
    return float(np.arccos(np.clip((np.trace(first.T @ second)-1)/2, -1., 1.)))


def estimate_pose(corners, camera_matrix, dist_coeffs, *, previous=None,
                  max_reprojection_error_px=5.0) -> PoseOutput:
    """Fit a 210 x 297 mm plane to four cyclic boundary vertices.

    Returns the pad centre in camera coordinates (x right, y down, z forward),
    in metres. distance_m is ||tvec||, not just optical-axis depth tvec[2].
    Intrinsics and distortion MUST describe the corner coordinate system:
    pass zero/None distortion for already-undistorted points using the same K.

    Image corner ordering does not label physical A4 corners. Test both choices
    of the short edge and both planar IPPE solutions, refined with solvePnP.
    The 180-degree equivalent orientation is intrinsically unobservable for X.
    A previous valid pose selects a nearby orientation only among fits within
    0.25 px RMS of the best; this is continuity, not independent evidence.
    Reset previous after tracking loss. Invalid results contain no stale pose.
    """
    def invalid(reason, error=None):
        return PoseOutput(False, None, None, None, error, reason=reason)

    K = np.asarray(camera_matrix, dtype=np.float64)
    dist = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or
            K[0, 0] <= 0 or K[1, 1] <= 0 or
            not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1])+abs(K[1, 0]) > 1e-8):
        raise ValueError('Expected OpenCV intrinsics with positive focal lengths and zero skew.')
    if dist is not None and (dist.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(dist).all()):
        raise ValueError('Invalid OpenCV distortion coefficients.')
    if not np.isfinite(max_reprojection_error_px) or max_reprojection_error_px <= 0:
        raise ValueError('Reprojection threshold must be finite and positive.')
    if corners is None:
        return invalid('no_corners')
    points = np.asarray(corners, dtype=np.float64)
    if points.size != 8 or not np.isfinite(points).all():
        return invalid('invalid_corners')
    points = points.reshape(4, 2)
    contour = points.astype(np.float32)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 1.:
        return invalid('degenerate_corners')
    indices = np.arange(4)
    if cv2.contourArea(contour, oriented=True) < 0:
        indices = indices[[0, 3, 2, 1]]
    candidates = []
    half_turn = np.diag([-1., -1., 1.])
    for shift in (0, 1):
        ids = np.roll(indices, -shift)
        image_points = np.ascontiguousarray(points[ids])
        seeds = []
        try:
            count, rotations, translations, _ = cv2.solvePnPGeneric(
                A4_OBJECT_POINTS, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE)
            if count:
                seeds.extend(zip(rotations, translations))
        except cv2.error:
            pass
        try:
            # ITERATIVE also handles singular fronto-parallel IPPE seeds.
            ok, rv, tv = cv2.solvePnP(A4_OBJECT_POINTS, image_points, K, dist,
                                    flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                seeds.append((rv, tv))
        except cv2.error:
            pass
        for rv, tv in seeds:
            if not np.isfinite(rv).all() or not np.isfinite(tv).all():
                continue
            try:
                ok, rv, tv = cv2.solvePnP(A4_OBJECT_POINTS, image_points, K, dist,
                                        rv.copy(), tv.copy(), True, flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok or not np.isfinite(rv).all() or not np.isfinite(tv).all():
                    continue
                R = cv2.Rodrigues(rv)[0]
                camera_points = A4_OBJECT_POINTS @ R.T + tv.reshape(3)
                if np.any(camera_points[:, 2] <= 1e-6):
                    continue
                projected = cv2.projectPoints(A4_OBJECT_POINTS, rv, tv, K, dist)[0].reshape(4, 2)
            except cv2.error:
                continue
            error = float(np.sqrt(np.mean(np.sum((projected-image_points)**2, axis=1))))
            if not np.isfinite(error):
                continue
            candidates.append((error, R, rv, tv, ids.copy()))
            flipped = R @ half_turn
            candidates.append((error, flipped, cv2.Rodrigues(flipped)[0], tv, np.roll(ids, -2)))
    if not candidates:
        return invalid('pnp_failed')
    best_error = min(c[0] for c in candidates)
    if best_error > max_reprojection_error_px:
        return invalid('high_reprojection_error', best_error)
    close = [c for c in candidates if c[0] <= min(best_error+.25, max_reprojection_error_px)]
    if previous is not None and previous.valid and previous.rvec is not None:
        old_rotation = cv2.Rodrigues(previous.rvec)[0]
        chosen = min(close, key=lambda c: _rotation_distance(old_rotation, c[1]))
    else:
        # Deterministic initial representative: best fit, yaw nearest zero.
        best = [c for c in candidates if c[0] <= min(best_error+1e-7, max_reprojection_error_px)]
        chosen = min(best, key=lambda c: abs(rotation_to_rpy(c[1])[2]))
    error, R, rv, tv, ids = chosen
    ambiguous = any(min(_rotation_distance(R, c[1]),
                        _rotation_distance(R, c[1] @ half_turn)) > np.deg2rad(5)
                    for c in close)
    return PoseOutput(True, tv.reshape(3).copy(), float(np.linalg.norm(tv)),
                      rotation_to_rpy(R), error, rv.copy(), tv.copy(), 'ok',
                      ids.copy(), ambiguous)

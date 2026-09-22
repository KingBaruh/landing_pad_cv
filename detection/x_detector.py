import cv2
import numpy as np

_MAX_LINE_FIT_POINTS = 256


def detect_x(rectified_image, *, diagnostics=None):
    """Return (valid, quality_score) for a large dark X on a light sheet.

    Normalize to a square and find two diagonal directions with Hough lines.
    Test arms relative to their measured intersection, allowing thin hand-drawn
    strokes and a moderately off-center crossing. The score is not a probability.
    If Otsu loses a lower-contrast stroke, try local Gaussian thresholding with
    the same geometry checks. Dark-sheet rejection still applies. An optional
    diagnostics dict receives measurements and both threshold attempts.
    """
    if diagnostics is not None:
        diagnostics.clear()

    def reject(reason):
        if diagnostics is not None:
            diagnostics['rejection_reason'] = reason
        return False, 0.0

    if (not isinstance(rectified_image, np.ndarray) or rectified_image.size == 0 or
            rectified_image.dtype != np.uint8 or rectified_image.ndim not in (2, 3)):
        return reject('invalid_image')
    if rectified_image.ndim == 3:
        if rectified_image.shape[2] not in (3, 4):
            return reject('unsupported_channels')
        code = cv2.COLOR_BGR2GRAY if rectified_image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        gray = cv2.cvtColor(rectified_image, code)
    else:
        gray = rectified_image
    if min(gray.shape) < 24:
        return reject('image_too_small')

    # The marker tests below use this normalized crop, not source-frame pixels.
    # These tolerances therefore keep the same meaning as the pad changes size.
    size = 200
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # A thin pen stroke may occupy less than 1% of a resized crop. Estimate
    # contrast from a smaller tail; the independent line/arm checks below
    # still have to explain the foreground as a complete X.
    low, high = np.percentile(gray, [.25, 95])
    if high - low < 35:
        return reject('insufficient_contrast')
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    dark = (gray <= min(threshold, high - 25)).astype(np.uint8) * 255

    # Otsu can keep the dark stroke but erase a lighter stroke. Try local
    # Gaussian thresholding only after the same geometry checks reject Otsu.
    metrics = {}
    valid, score = _verify_mask(dark, metrics)
    method = 'otsu'
    attempts = [dict(method=method, valid=valid, quality=score, **metrics)]
    # Preserve rejection of dark sheets/inverse markers: adaptive thresholding
    # can otherwise turn the dark margins of a white X into a false dark X.
    if not valid and metrics.get('black_fraction', 0.) <= .43:
        local = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 31, 10)
        local_metrics = {}
        local_valid, local_score = _verify_mask(local, local_metrics)
        attempts.append(dict(method='adaptive_gaussian', valid=local_valid,
                             quality=local_score, **local_metrics))
        if local_valid or local_score > score:
            valid, score, metrics = local_valid, local_score, local_metrics
            method = 'adaptive_gaussian'
    if diagnostics is not None:
        diagnostics.update(metrics, threshold_method=method, attempts=attempts)
    return valid, score


def _verify_mask(dark, diagnostics):
    """Apply identical X geometry/arm tests to either binary foreground mask."""
    size = dark.shape[0]

    def reject(reason):
        diagnostics['rejection_reason'] = reason
        return False, 0.0

    # Ignore the paper boundary; only the interior should contain the marker.
    border = 16
    region = np.zeros_like(dark, dtype=bool)
    region[border:-border, border:-border] = True
    dark[~region] = 0
    foreground = dark > 0
    black_fraction = float(foreground[region].mean())
    if diagnostics is not None:
        diagnostics['black_fraction'] = black_fraction
    if not 0.012 <= black_fraction <= 0.43:
        return reject('foreground_fraction_out_of_range')

    edges = cv2.Canny(dark, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                            minLineLength=50, maxLineGap=18)
    diagonal_lines = [[], []]
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = float(x2-x1), float(y2-y1)
            if abs(dx) < 1 or not 0.60 <= abs(dy/dx) <= 1.67:
                continue
            length = float(np.hypot(dx, dy))
            sign = 0 if dx*dy > 0 else 1
            origin = np.array([x1, y1], dtype=np.float64)
            direction = np.array([dx, dy], dtype=np.float64) / length
            diagonal_lines[sign].append((length, origin, direction))
    for group in diagonal_lines:
        group.sort(key=lambda line: line[0], reverse=True)
        del group[6:]  # Bound the pair search for cluttered candidates.
    if diagnostics is not None:
        diagnostics['diagonal_line_counts'] = [len(group) for group in diagonal_lines]
    if not all(diagonal_lines):
        return reject('missing_diagonal_direction')

    distance_to_dark = cv2.distanceTransform(255-dark, cv2.DIST_L2, 3)
    ys, xs = np.nonzero(foreground)
    dark_points = np.column_stack((xs, ys))
    # Hough often sees one straight section/edge of a hand-drawn stroke. Fit
    # its centerline from nearby ink so that a slightly curved arm is not
    # judged against the extension of just its opposite arm's edge.
    for group in diagonal_lines:
        for index, (length, origin, direction) in enumerate(group):
            relative = dark_points-origin
            near = np.abs(relative[:,0]*direction[1]-relative[:,1]*direction[0]) <= 18
            support = dark_points[near].astype(np.float32)
            if len(support) < 8:
                continue
            # Uniformly cover the full stroke while bounding robust-fit cost.
            # All foreground pixels still participate in the acceptance tests.
            if len(support) > _MAX_LINE_FIT_POINTS:
                indices = np.linspace(0, len(support)-1, _MAX_LINE_FIT_POINTS).astype(int)
                support = support[indices]
            vx, vy, x0, y0 = cv2.fitLine(support, cv2.DIST_HUBER, 0, .01, .01).ravel()
            fitted_direction = np.array([vx, vy], dtype=np.float64)
            if np.any(np.abs(fitted_direction) < 1e-6) or abs(np.dot(fitted_direction, direction)) < .9:
                continue
            if np.dot(fitted_direction, direction) < 0:
                fitted_direction *= -1
            group[index] = (length, np.array([x0,y0], dtype=np.float64), fitted_direction)
    return _score_line_pairs(diagonal_lines, dark_points, distance_to_dark, border, diagnostics)


def _score_line_pairs(groups, dark_points, distance_to_dark, border, diagnostics):
    """Evaluate the same bounded line pairs in arrays, retaining every gate.

    Pair order matches the original nested loop, including first-wins ties.
    Ink-to-line distances are computed once per line instead of once per pair.
    """
    size = distance_to_dark.shape[0]
    lengths, origins, directions, ink_distances = [], [], [], []
    for group in groups:
        lengths.append(np.array([line[0] for line in group]))
        origins.append(np.stack([line[1] for line in group]))
        directions.append(np.stack([line[2] for line in group]))
        relative = dark_points[None, :, :]-origins[-1][:, None, :]
        ink_distances.append(np.abs(relative[:, :, 0]*directions[-1][:, None, 1]
                                    - relative[:, :, 1]*directions[-1][:, None, 0]))
    a = np.repeat(np.arange(len(groups[0])), len(groups[1]))
    b = np.tile(np.arange(len(groups[1])), len(groups[0]))
    matrices = np.stack((directions[0][a], -directions[1][b]), axis=2)
    keep = np.abs(np.linalg.det(matrices)) >= .2
    a, b, matrices = a[keep], b[keep], matrices[keep]
    parameters = np.linalg.solve(matrices, (origins[1][b]-origins[0][a])[..., None])[..., 0]
    crossings = origins[0][a]+parameters[:, :1]*directions[0][a]
    keep = ((crossings >= .30*size) & (crossings <= .70*size)).all(axis=1)
    a, b, crossings = a[keep], b[keep], crossings[keep]
    if not len(a):
        diagnostics['rejection_reason'] = 'intersection_outside_center'
        return False, 0.0
    explained = (np.minimum(ink_distances[0][a], ink_distances[1][b]) <= 14).mean(axis=1)

    # Four rays per pair, each with the original 30 samples along the arm.
    rays = np.stack((directions[0][a], -directions[0][a],
                     directions[1][b], -directions[1][b]), axis=1)
    edges = np.where(rays > 0, size-1-border, border)
    reach = ((edges-crossings[:, None, :])/rays).min(axis=2)
    samples = (crossings[:, None, None, :]
               + np.linspace(.12, .82, 30)[None, None, :, None]
               * reach[:, :, None, None]*rays[:, :, None, :])
    samples = np.clip(np.round(samples).astype(int), 0, size-1)
    arms = (distance_to_dark[samples[..., 1], samples[..., 0]] <= 10).mean(axis=2)
    xy = np.round(crossings).astype(int)
    crossing_distance = distance_to_dark[xy[:, 1], xy[:, 0]]
    centrality = np.maximum(0., 1-np.linalg.norm(crossings-99.5, axis=1)/60)
    strength = np.minimum(1., np.minimum(lengths[0][a], lengths[1][b])/185)
    scores = .25*strength + .35*arms.min(axis=1) + .25*explained + .15*centrality
    valid = (arms.min(axis=1) >= .70) & (explained >= .75) & (crossing_distance <= 4)
    # Every valid candidate outranks every invalid one; scores are within [0,1].
    best = int(np.argmax(scores + 2*valid))
    reason = None
    if arms[best].min() < .70:
        reason = 'missing_or_short_arm'
    elif explained[best] < .75:
        reason = 'foreground_not_concentrated_on_lines'
    elif crossing_distance[best] > 4:
        reason = 'no_dark_intersection'
    diagnostics.update(intersection_normalized=(crossings[best]/(size-1)).tolist(),
                       arm_support=arms[best].tolist(), explained_foreground=float(explained[best]),
                       crossing_distance_px=float(crossing_distance[best]), rejection_reason=reason)
    return bool(valid[best]), min(1. if valid[best] else .49, float(scores[best]))

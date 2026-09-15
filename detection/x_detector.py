import cv2
import numpy as np


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
            vx, vy, x0, y0 = cv2.fitLine(support, cv2.DIST_HUBER, 0, .01, .01).ravel()
            fitted_direction = np.array([vx, vy], dtype=np.float64)
            if np.any(np.abs(fitted_direction) < 1e-6) or abs(np.dot(fitted_direction, direction)) < .9:
                continue
            if np.dot(fitted_direction, direction) < 0:
                fitted_direction *= -1
            group[index] = (length, np.array([x0,y0], dtype=np.float64), fitted_direction)
    best_valid, best_score = False, 0.0
    best_metrics = {'rejection_reason': 'intersection_outside_center'}
    for length_a, origin_a, direction_a in diagonal_lines[0]:
        for length_b, origin_b, direction_b in diagonal_lines[1]:
            matrix = np.column_stack((direction_a, -direction_b))
            if abs(np.linalg.det(matrix)) < .2:
                continue
            t, _ = np.linalg.solve(matrix, origin_b-origin_a)
            crossing = origin_a + t*direction_a
            if np.any(crossing < .30*size) or np.any(crossing > .70*size):
                continue

            # Explain foreground using the fitted lines, not a fixed X template.
            distances = []
            for origin, direction in ((origin_a, direction_a), (origin_b, direction_b)):
                relative = dark_points-origin
                distances.append(np.abs(relative[:,0]*direction[1]-relative[:,1]*direction[0]))
            explained = float((np.minimum(*distances) <= 14).mean())

            # Each ray must extend well towards the paper edge. A slash, V or
            # truncated arm cannot pass just because two line extensions cross.
            arm_scores = []
            for direction in (direction_a, -direction_a, direction_b, -direction_b):
                edge = np.where(direction > 0, size-1-border, border)
                reach = float(np.min((edge-crossing)/direction))
                samples = crossing + np.linspace(.12, .82, 30)[:,None]*reach*direction
                samples = np.clip(np.round(samples).astype(int), 0, size-1)
                arm_scores.append(float((distance_to_dark[samples[:,1],samples[:,0]] <= 10).mean()))
            cx, cy = np.round(crossing).astype(int)
            crossing_distance = float(distance_to_dark[cy, cx])
            centrality = max(0., 1-float(np.linalg.norm(crossing-99.5))/60)
            strength = min(1., min(length_a, length_b)/185)
            # This weighted score ranks candidates. The separate hard checks
            # below still require all arms, concentrated ink and a dark crossing.
            score = float(.25*strength + .35*min(arm_scores) + .25*explained + .15*centrality)
            reason = None
            if min(arm_scores) < .70:
                reason = 'missing_or_short_arm'
            elif explained < .75:
                reason = 'foreground_not_concentrated_on_lines'
            elif crossing_distance > 4:
                reason = 'no_dark_intersection'
            valid = reason is None
            if (valid and not best_valid) or (valid == best_valid and score > best_score):
                best_valid, best_score = valid, score
                best_metrics = dict(intersection_normalized=(crossing/(size-1)).tolist(),
                                    arm_support=arm_scores, explained_foreground=explained,
                                    crossing_distance_px=crossing_distance, rejection_reason=reason)
    if diagnostics is not None:
        diagnostics.update(best_metrics)
    return bool(best_valid), min(1., best_score) if best_valid else min(.49, best_score)

import cv2
import numpy as np

from utils import removeComputerGraphics, adaptive_s_threshold, hue2Opencv
from config import (
    PERFORMANCE_MODE,
    WATER_LOWER,
    WATER_UPPER,
    COLOR_SIMILARITY_TOLERANCE,
    MAX_JUMP_PIXELS,
    SOLIDITY_MIN,
)

# morphological kernel used by mask creation
kernel3 = np.ones((3,3),np.uint8)


def _preprocess_patch_for_ml(patch, patch_size=20):
    """Convert patch to flattened normalized features for ML prediction."""
    if patch is None:
        return None
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch
    if gray.shape != (patch_size, patch_size):
        gray = cv2.resize(gray, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.reshape(-1)


def _classify_contour_patch_ml(
    inputImage,
    cx,
    cy,
    model,
    patch_size=20,
    classifier_name="ml",
    use_confidence_gate=False,
    min_margin=0.0,
):
    """Classify contour-centered patch with trained ML model."""
    if model is None:
        if not hasattr(_classify_contour_patch_ml, "_warned_no_model"):
            print(f"[{classifier_name.upper()}] model is None during contour classification")
            _classify_contour_patch_ml._warned_no_model = True
        return False, None
    half = patch_size // 2
    padded = cv2.copyMakeBorder(inputImage, half, half, half, half, cv2.BORDER_REFLECT_101)
    cx_i, cy_i = int(round(cx)), int(round(cy))
    patch = padded[cy_i:cy_i + patch_size, cx_i:cx_i + patch_size]
    features = _preprocess_patch_for_ml(patch, patch_size=patch_size)
    if features is None:
        return False, None

    try:
        sample = [features]
        pred = model.predict(sample)
        is_ball = bool(pred[0])

        margin = None
        if hasattr(model, "decision_function"):
            decision = model.decision_function(sample)
            if isinstance(decision, (list, tuple, np.ndarray)):
                margin = float(np.abs(np.asarray(decision).reshape(-1)[0]))
            else:
                margin = float(abs(decision))

        if use_confidence_gate and margin is not None and margin < float(min_margin):
            return False, margin

        return is_ball, margin
    except Exception as exc:
        if not hasattr(_classify_contour_patch_ml, "_warned_predict_error"):
            print(f"[{classifier_name.upper()}] predict failed in contour classification: {exc}")
            _classify_contour_patch_ml._warned_predict_error = True
        return False, None


def _calcContourMetrics(cnt):
    """Compute shared contour metrics used for filtering and fused-blob checks."""
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 0.0
    if perimeter > 0:
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = max(w, h) / max(1, min(w, h))
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
    }


def _checkFusedBlob(cnt, m, expected_ball_area, binaryMask, blob_num):
    """Detect whether a contour looks like multiple objects fused together."""
    area, circularity = m['area'], m['circularity']
    aspect_ratio, solidity = m['aspect_ratio'], m['solidity']
    x, y, w, h = m['x'], m['y'], m['w'], m['h']

    fused_score = 0
    if area > expected_ball_area * 1.3:
        if 0.3 < circularity < 0.7:
            fused_score += 1
        if aspect_ratio > 1.4:
            fused_score += 1
        if 0.6 < solidity < 0.93:
            fused_score += 1

    if fused_score <= 2:
        return False, [], False

    print(
        f"Fused blob #{blob_num}: "
        f"area={area:.1f}, circularity={circularity:.2f}, "
        f"aspect_ratio={aspect_ratio:.2f}, solidity={solidity:.2f}, "
        f"score_flags={fused_score}"
    )

    statuses = [(cnt, 'fused_blob')]
    roi = binaryMask[y:y + h, x:x + w]
    if roi.size == 0:
        return True, statuses, True

    dist_transform = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    _, radius, _, loc = cv2.minMaxLoc(dist_transform)
    cx = x + loc[0]
    cy = y + loc[1]
    radius_i = int(max(1.0, float(radius)))
    angles = np.arange(0, 360, 10, dtype=np.float32)
    xs = int(cx) + (radius_i * np.cos(np.deg2rad(angles))).astype(np.int32)
    ys = int(cy) + (radius_i * np.sin(np.deg2rad(angles))).astype(np.int32)
    split_cnt = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    statuses.append((split_cnt, 'splitted_blob'))
    return True, statuses, False


def _scan_fused_blob_for_ball(
    inputImage,
    cnt,
    model,
    patch_size=20,
    use_confidence_gate=False,
    min_margin=0.0,
    scan_step=4,
    expected_radius=9.0,
):
    """Scan positions inside a fused blob and return the strongest SVM-positive location."""
    if model is None:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 0 or h <= 0:
        return None

    contour_mask = np.zeros((h, w), dtype=np.uint8)
    shifted_cnt = cnt.copy()
    shifted_cnt[:, 0, 0] -= x
    shifted_cnt[:, 0, 1] -= y
    cv2.drawContours(contour_mask, [shifted_cnt], -1, 255, -1)

    half = patch_size // 2
    padded = cv2.copyMakeBorder(inputImage, half, half, half, half, cv2.BORDER_REFLECT_101)

    positions = []
    feature_rows = []
    step = max(1, int(scan_step))

    for cy in range(y, y + h, step):
        local_y = min(h - 1, max(0, cy - y))
        for cx in range(x, x + w, step):
            local_x = min(w - 1, max(0, cx - x))
            if contour_mask[local_y, local_x] == 0:
                continue

            patch = padded[int(round(cy)):int(round(cy)) + patch_size, int(round(cx)):int(round(cx)) + patch_size]
            features = _preprocess_patch_for_ml(patch, patch_size=patch_size)
            if features is None:
                continue

            positions.append((float(cx), float(cy)))
            feature_rows.append(features)

    if not feature_rows:
        return None

    X = np.asarray(feature_rows)
    try:
        predictions = model.predict(X)
        decision_scores = None
        if hasattr(model, 'decision_function'):
            decision_scores = np.asarray(model.decision_function(X)).reshape(-1)
    except Exception:
        return None

    best_index = None
    best_score = float('-inf')
    for index, pred in enumerate(predictions):
        if not bool(pred):
            continue

        signed_margin = None
        if decision_scores is not None:
            signed_margin = float(decision_scores[index])
            if signed_margin <= 0.0:
                continue
            if use_confidence_gate and signed_margin < float(min_margin):
                continue
            rank_score = signed_margin
        else:
            rank_score = 1.0

        if rank_score > best_score:
            best_score = rank_score
            best_index = index

    if best_index is None:
        return None

    cx, cy = positions[best_index]
    margin = float(decision_scores[best_index]) if decision_scores is not None else None
    return {
        'cx': cx,
        'cy': cy,
        'r': float(max(3.0, expected_radius)),
        'margin': margin,
    }


def _build_fused_blob_candidate(
    cnt,
    m,
    scan_match,
    last_ball,
    frames_with_ball,
    mean_diam,
    std_diam,
    weight_history_dist,
    weight_diameter_likelihood,
    weight_y_boost,
    frame_height,
    ml_score_boost,
):
    """Build a candidate using the SVM-located position inside a fused blob."""
    cx = float(scan_match['cx'])
    cy = float(scan_match['cy'])
    r = float(scan_match['r'])
    margin = scan_match.get('margin')

    diameter = r * 2.0
    diameter_likelihood = np.exp(-((diameter - mean_diam) ** 2.0) / (2.0 * std_diam ** 2.0))

    if last_ball is not None and isinstance(last_ball, dict):
        dist = np.hypot(cx - last_ball['cx'], cy - last_ball['cy'])
        historyWeight = 3.0 * (1 - np.exp(-0.2 * frames_with_ball))
    else:
        dist = 0.0
        historyWeight = 0.0

    dist_norm = dist / (r + 1.0)
    y_boost = weight_y_boost if cy > frame_height / 2 else 0.0
    margin_bonus = min(2.0, float(margin)) if margin is not None else 0.5
    score = (
        -historyWeight * dist_norm * weight_history_dist
        + weight_diameter_likelihood * diameter_likelihood
        + y_boost
        + float(ml_score_boost)
        + margin_bonus
    )

    return {
        'cnt': cnt,
        'score': score,
        'circularity': float(m.get('circularity', 0.0)),
        'diameter_likelihood': diameter_likelihood,
        'y_boost': y_boost,
        'historyWeight': historyWeight,
        'dist_norm': dist_norm,
        'cx': cx,
        'cy': cy,
        'r': r,
        'ml_is_ball': True,
        'ml_margin': margin,
        'from_fused_blob_scan': True,
    }


def _build_circle_contour(cx, cy, r, points=36):
    """Build a polygon contour approximating a circle for candidate flow usage."""
    radius = max(2.0, float(r))
    theta = np.linspace(0.0, 2.0 * np.pi, num=max(12, int(points)), endpoint=False)
    xs = np.round(float(cx) + radius * np.cos(theta)).astype(np.int32)
    ys = np.round(float(cy) + radius * np.sin(theta)).astype(np.int32)
    return np.stack([xs, ys], axis=1).reshape(-1, 1, 2)


def waterMaks(inputImage, hsv):
    # use configured HSV ranges for water areas
    return cv2.inRange(hsv, WATER_LOWER, WATER_UPPER)


def waterMirroMask(inputImage, hsv):
    lower = (hue2Opencv(0), 0, 0)
    upper = (hue2Opencv(59), 255, 255)
    return cv2.inRange(hsv, lower, upper)


def calculateGolfballhsvMask(inputImage, hsv, debug, adaptive):
    # Fast RGB similarity detection using cv2.absdiff (optimized C++ implementation)
    threshold = COLOR_SIMILARITY_TOLERANCE
    diff_rg = cv2.absdiff(inputImage[:,:,2], inputImage[:,:,1])
    diff_rb = cv2.absdiff(inputImage[:,:,2], inputImage[:,:,0])
    diff_gb = cv2.absdiff(inputImage[:,:,1], inputImage[:,:,0])
    color_similarity = (diff_rg < threshold) & (diff_rb < threshold) & (diff_gb < threshold)
    rgb_mask = color_similarity.astype(np.uint8) * 255

    # Cache trackbar values in single unpacking for speed
    h_min, h_max, s_min, s_max, v_min, v_max = [cv2.getTrackbarPos(n, "Controls") for n in ["Hmin", "Hmax", "Smin", "Smax", "Vmin", "Vmax"]]

    if adaptive:
        adaptive_threshold = adaptive_s_threshold(inputImage)
        lowerhsv = (h_min, 0, v_min)
        upperhsv = (h_max, adaptive_threshold, v_max)
    else:
        lowerhsv = (h_min, s_min, v_min)
        upperhsv = (h_max, s_max, v_max)

    maskhsv = cv2.inRange(hsv, lowerhsv, upperhsv)
    return cv2.bitwise_or(rgb_mask, maskhsv)


def createBinaryMask(inputImage, hsv, debug, adaptive, player_exclusion_mask=None):

    #watermask = waterMaks(inputImage, hsv)
    #inv_water_mask = cv2.bitwise_not(watermask)
    #waterMirrorMask = waterMirroMask(inputImage, hsv)
    #inv_waterMirror_mask = cv2.bitwise_not(waterMirrorMask)

    golfBallHSVMask = calculateGolfballhsvMask(inputImage, hsv, debug, adaptive)

    closing = cv2.morphologyEx(golfBallHSVMask, cv2.MORPH_CLOSE, kernel3)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel3)
    dilate = cv2.dilate(opening, kernel3, iterations=1)
    
    # Black out region with graphics
    dilate[44:231, 1150:1850] = 0 #right side
    dilate[70:190, 75:1200] = 0  #left side
    dilate[:80, :] = 0
    dilate[-80:, :] = 0
    dilate[:, :80] = 0
    dilate[:, -80:] = 0

    from config import DETECT_GRAPHICS
    graphics_mask = None
    if DETECT_GRAPHICS:
        from utils import detect_graphics_rectangles, create_graphics_exclusion_mask
        graphics_rects = detect_graphics_rectangles(inputImage)
        if graphics_rects:
            graphics_mask = create_graphics_exclusion_mask(inputImage.shape, graphics_rects, margin=5)

    combined_graphics_dilate = cv2.bitwise_and(dilate, graphics_mask) if graphics_mask is not None else dilate

    if player_exclusion_mask is not None:
        final_mask = cv2.bitwise_and(combined_graphics_dilate, player_exclusion_mask)
    else:
        final_mask = combined_graphics_dilate

    from config import GRASS_LOWER, GRASS_UPPER, CLASSIFY_GRASS_MIN_PERCENT
    grass_mask = cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER)
    grass_percent = np.count_nonzero(grass_mask) / (inputImage.shape[0] * inputImage.shape[1]) * 100
    if grass_percent >= CLASSIFY_GRASS_MIN_PERCENT:
        final_mask = cv2.bitwise_and(final_mask, grass_mask)

    if not PERFORMANCE_MODE:
        cv2.imshow(f'final_mask', final_mask)

    return final_mask


def identifyContours(inputImage, binaryMask, output, tracker, last_ball, debug, return_diagnostics=False):
    from config import WEIGHT_Y_BOOST, ML_SCORE_BOOST, ML_MIN_CIRCULARITY_FOR_POSITIVE, FUSED_BLOB_SCAN_STEP
    # Cache trackbar values for performance
    trackbar_values = {
        "check_area": cv2.getTrackbarPos("Area", "Controls") == 1,
        "check_solidity": cv2.getTrackbarPos("Solidity", "Controls") == 1,
        "check_circularity": cv2.getTrackbarPos("Circularity", "Controls") == 1,
        "circularityLimit": cv2.getTrackbarPos("Circ", "Controls") / 100.0,
        "sizeMinLimit": cv2.getTrackbarPos("SzMin", "Controls"),
        "sizeMaxLimit": cv2.getTrackbarPos("SzMax", "Controls"),
        "check_dist": cv2.getTrackbarPos("Dist", "Controls") == 1
    }
    check_area = trackbar_values["check_area"]
    check_solidity = trackbar_values["check_solidity"]
    check_circularity = trackbar_values["check_circularity"]
    circularityLimit = trackbar_values["circularityLimit"]
    sizeMinLimit = trackbar_values["sizeMinLimit"]
    sizeMaxLimit = trackbar_values["sizeMaxLimit"]
    check_dist = trackbar_values["check_dist"]
    best_score = -1
    best_ball = None
    # Score weights (define above score calculation)
    WEIGHT_CIRCULARITY = 8.0
    WEIGHT_HISTORY_DIST = 0.1
    WEIGHT_DIAMETER_LIKELIHOOD = 6.0

    contours, _ = cv2.findContours(binaryMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Visualize contours in the mask image
    mask_vis = cv2.cvtColor(binaryMask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask_vis, contours, -1, (0, 255, 255), 2)
    # Write number of contours in lower left corner
    h, w = mask_vis.shape[:2]
    cv2.putText(mask_vis, f"Contours: {len(contours)}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

    countourWrongSize = 0
    countoursWrongPerimeter = 0
    countoursWrongCircularity = 0
    countoursCompletedLoop = 0
    countourCorrectScore = 0
    countourWrongSolidity = 0
    frames_with_ball = getattr(getattr(tracker, 'golfBall', tracker), 'framesWithBall', 0)
    # --- Diameter likelihood scoring ---
    # Use tracker.diameter_history (to be set/reset externally) for last 10 diameters
    # If not available, use general distribution mean=18, std=6
    #if hasattr(tracker, 'diameter_history') and tracker.diameter_history and len(tracker.diameter_history) >= 3:
    #    diameters = tracker.diameter_history[-10:]
    #    mean_diam = float(np.mean(diameters))
    #    std_diam = float(np.std(diameters))
    #else:
    mean_diam = 18.0
    std_diam = 6.0
    expected_ball_area = np.pi * (mean_diam / 2.0) ** 2

    # Collect all candidate contours and their scores/components
    candidate_contours = []
    # For color-coding: store tuples (cnt, reason)
    contour_statuses = []
    raw_ml_positive_contour_ids = set()
    positive_ml_contour_ids = set()
    fused_blob_counter = 0

    classifier_name = getattr(tracker, 'patch_classifier_name', 'ml') or 'ml'
    classifier_model = getattr(tracker, 'patch_classifier', None)
    if classifier_model is None:
        classifier_model = getattr(tracker, 'knn_model', None)
    classifier_patch_size = int(getattr(tracker, 'patch_classifier_patch_size', 20))
    use_confidence_gate = bool(getattr(tracker, 'patch_classifier_use_confidence_gate', False))
    min_margin = float(getattr(tracker, 'patch_classifier_min_margin', 0.0))

    for cnt in contours:
        m = _calcContourMetrics(cnt)
        if m['aspect_ratio'] > 3:
            continue

        is_fused, fused_statuses, should_skip = _checkFusedBlob(
            cnt,
            m,
            expected_ball_area,
            binaryMask,
            fused_blob_counter + 1,
        )
        if is_fused:
            fused_blob_counter += 1
            contour_statuses.extend(fused_statuses)
            if should_skip:
                continue

            fused_scan_match = _scan_fused_blob_for_ball(
                inputImage,
                cnt,
                classifier_model,
                patch_size=classifier_patch_size,
                use_confidence_gate=use_confidence_gate,
                min_margin=min_margin,
                scan_step=FUSED_BLOB_SCAN_STEP,
                expected_radius=mean_diam / 2.0,
            )
            if fused_scan_match is not None:
                fused_ball_cnt = _build_circle_contour(
                    fused_scan_match['cx'],
                    fused_scan_match['cy'],
                    fused_scan_match['r'],
                )
                raw_ml_positive_contour_ids.add(id(fused_ball_cnt))
                fused_candidate = _build_fused_blob_candidate(
                    fused_ball_cnt,
                    m,
                    fused_scan_match,
                    last_ball,
                    frames_with_ball,
                    mean_diam,
                    std_diam,
                    WEIGHT_HISTORY_DIST,
                    WEIGHT_DIAMETER_LIKELIHOOD,
                    WEIGHT_Y_BOOST,
                    inputImage.shape[0],
                    ML_SCORE_BOOST,
                )
                candidate_contours.append(fused_candidate)
                countoursCompletedLoop += 1
                if fused_candidate['score'] > best_score:
                    countourCorrectScore += 1
                    best_score = fused_candidate['score']
                    best_ball = (fused_candidate['cx'], fused_candidate['cy'], fused_candidate['r'])
                contour_statuses.append((fused_ball_cnt, 'fused_svm_ball'))
                if debug:
                    margin_text = f", margin={fused_candidate['ml_margin']:.3f}" if fused_candidate['ml_margin'] is not None else ""
                    print(
                        f"Fused blob scanned by {classifier_name.upper()} -> position=({fused_candidate['cx']:.1f}, {fused_candidate['cy']:.1f})"
                        f"{margin_text}, score={fused_candidate['score']:.2f}"
                    )
                continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        cx, cy = x, y

        area = cv2.contourArea(cnt)
        # Area check
        if check_area:
            if area < sizeMinLimit or area > sizeMaxLimit:
                countourWrongSize += 1
                contour_statuses.append((cnt, 'wrong_area'))
                continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
        else:
            solidity = 0
        # Solidity check
        if check_solidity:
            if solidity < SOLIDITY_MIN:
                countourWrongSolidity += 1
                contour_statuses.append((cnt, 'wrong_solidity'))
                continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            countoursWrongPerimeter += 1
            contour_statuses.append((cnt, 'wrong_perimeter'))
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        # Circularity check
        if check_circularity:
            if circularity < circularityLimit:
                countoursWrongCircularity += 1
                contour_statuses.append((cnt, 'wrong_circularity'))
                continue
        diameter = r * 2
        diameter_likelihood = np.exp(-((diameter - mean_diam) ** 2.0) / (2.0 * std_diam ** 2.0))
        if last_ball is not None and isinstance(last_ball, dict):
            px, py, pr = last_ball['cx'], last_ball['cy'], last_ball['r']
            dist = np.hypot(cx - px, cy - py)
            historyWeight = 3.0 * (1 - np.exp(-0.2 * frames_with_ball))
        else:
            dist = 0
            historyWeight = 0
        # Distance check
        if check_dist and dist > MAX_JUMP_PIXELS:
            contour_statuses.append((cnt, 'too_far'))
            continue

        # Only run SVM on contours that are already relevant geometric candidates.
        raw_ml_is_ball, ml_margin = _classify_contour_patch_ml(
            inputImage,
            cx,
            cy,
            classifier_model,
            patch_size=classifier_patch_size,
            classifier_name=classifier_name,
            use_confidence_gate=False,
            min_margin=0.0,
        )
        if raw_ml_is_ball:
            raw_ml_positive_contour_ids.add(id(cnt))

        ml_is_ball = raw_ml_is_ball
        if use_confidence_gate and ml_margin is not None and ml_margin < float(min_margin):
            ml_is_ball = False

        dist_norm = dist / (r + 1)
        frame_height = inputImage.shape[0]
        y_boost = WEIGHT_Y_BOOST if cy > frame_height / 2 else 0
        score = (
            WEIGHT_CIRCULARITY * circularity
            - historyWeight * dist_norm * WEIGHT_HISTORY_DIST
            + WEIGHT_DIAMETER_LIKELIHOOD * diameter_likelihood
            + y_boost
        )

        # Optional ML vote: boost score if contour-centered patch is classified as ball.
        if ml_is_ball and circularity < float(ML_MIN_CIRCULARITY_FOR_POSITIVE):
            ml_is_ball = False

        if ml_is_ball:
            positive_ml_contour_ids.add(id(cnt))
            score += float(ML_SCORE_BOOST)
            if debug:
                margin_text = f", margin={ml_margin:.3f}" if ml_margin is not None else ""
                print(
                    f"Contour at ({cx:.1f}, {cy:.1f}) classified as ball by {classifier_name.upper()}{margin_text}, "
                    f"boosting score to {score:.2f}"
                )

        candidate_contours.append({
            'cnt': cnt,
            'score': score,
            'circularity': circularity,
            'diameter_likelihood': diameter_likelihood,
            'y_boost': y_boost,
            'historyWeight': historyWeight,
            'dist_norm': dist_norm,
            'cx': cx,
            'cy': cy,
            'r': r,
            'ml_is_ball': ml_is_ball,
            'ml_margin': ml_margin,
        })
        countoursCompletedLoop += 1
        if score > best_score:
            countourCorrectScore += 1
            best_score = score
            best_ball = (cx, cy, r)
        # Passed all checks
        contour_statuses.append((cnt, 'ml_ball' if ml_is_ball else 'passed'))

    # Color map for reasons
    color_map = {
        'wrong_area': (0, 0, 255),        # Red
        'wrong_solidity': (255, 0, 255), # Magenta
        'wrong_perimeter': (0, 255, 255),# Yellow
        'wrong_circularity': (255, 0, 0),# Blue
        'too_far': (0, 165, 255),        # Orange
        'ml_ball': (0, 255, 255),       # Yellow
        'fused_svm_ball': (0, 255, 255),# Yellow
        'passed': (0, 255, 0),           # Green
        'fused_blob': (255, 255, 0),     # Cyan
        'splitted_blob': (0, 0, 139),    # Dark red
    }

    # Draw regular contours first, then fused diagnostics so they stay visible.
    non_fused_statuses = [
        (cnt, reason)
        for cnt, reason in contour_statuses
        if reason not in ('fused_blob', 'splitted_blob')
    ]
    fused_statuses = [(cnt, reason) for cnt, reason in contour_statuses if reason == 'fused_blob']
    splitted_statuses = [(cnt, reason) for cnt, reason in contour_statuses if reason == 'splitted_blob']

    for cnt, reason in non_fused_statuses:
        color = color_map.get(reason, (200, 200, 200))
        cv2.drawContours(mask_vis, [cnt], -1, color, 2)
        if id(cnt) in raw_ml_positive_contour_ids:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(mask_vis, "svm", (x + w + 6, max(20, y + h // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    for blob_idx, (cnt, _) in enumerate(fused_statuses, start=1):
        cv2.drawContours(mask_vis, [cnt], -1, color_map['fused_blob'], 2)
        x, y, w, h = cv2.boundingRect(cnt)
        label_pos = (x + w + 6, max(20, y + h // 2))
        cv2.putText(mask_vis, str(blob_idx), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        if id(cnt) in raw_ml_positive_contour_ids:
            svm_pos = (x + w + 6, max(40, y + h // 2 + 20))
            cv2.putText(mask_vis, "svm", svm_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    for cnt, _ in splitted_statuses:
        cv2.drawContours(mask_vis, [cnt], -1, color_map['splitted_blob'], 2)

    # Always highlight the top-scoring candidate in green.
    if candidate_contours:
        best_cand_for_mask = max(candidate_contours, key=lambda x: x['score'])
        cv2.drawContours(mask_vis, [best_cand_for_mask['cnt']], -1, (0, 255, 0), 3)

    # Draw score next to each candidate contour
    for cand in candidate_contours:
        text_pos = (int(cand['cx'] + cand['r'] + 10), int(cand['cy']))
        cv2.putText(mask_vis, f"{cand['score']:.1f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
        if cand.get('from_fused_blob_scan'):
            cv2.circle(mask_vis, (int(cand['cx']), int(cand['cy'])), int(max(3, cand['r'])), (0, 255, 255), 2)

    # Overlay breakdown for top 2 candidates: 1 left, 2 right
    top_candidates = sorted(candidate_contours, key=lambda x: x['score'], reverse=True)[:2]
    h, w = mask_vis.shape[:2]
    for idx, cand in enumerate(top_candidates):
        breakdown = [
            f"Score breakdown for candidate {idx+1}:",
            f"  WEIGHT_CIRCULARITY * circularity = {WEIGHT_CIRCULARITY:.1f} * {cand['circularity']:.2f} = {WEIGHT_CIRCULARITY * cand['circularity']:.2f}",
            f"  - historyWeight * dist_norm * WEIGHT_HISTORY_DIST = -{cand['historyWeight']:.2f} * {cand['dist_norm']:.2f} * {WEIGHT_HISTORY_DIST:.2f} = {-cand['historyWeight'] * cand['dist_norm'] * WEIGHT_HISTORY_DIST:.2f}",
            f"  WEIGHT_DIAMETER_LIKELIHOOD * diameter_likelihood = {WEIGHT_DIAMETER_LIKELIHOOD:.1f} * {cand['diameter_likelihood']:.2f} = {WEIGHT_DIAMETER_LIKELIHOOD * cand['diameter_likelihood']:.2f}",
            f"  y_boost = {cand['y_boost']:.2f}",
            f"  Total score = {cand['score']:.2f}"
        ]
        for line_num, line in enumerate(breakdown):
            y_pos = 30 + line_num * 20
            if idx == 0:
                x_pos = 10  # left
            else:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                x_pos = w - text_size[0] - 10  # right
            cv2.putText(mask_vis, line, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2, cv2.LINE_AA)

    if not PERFORMANCE_MODE:
        cv2.imshow("final_mask", mask_vis)
        
    # Draw only the top 3 candidates by score and print their score/components
    top_candidates = sorted(candidate_contours, key=lambda x: x['score'], reverse=True)[:3]
    for idx, cand in enumerate(top_candidates):
        cv2.drawContours(output, [cand['cnt']], -1, (0, 255, 255), 2)
        if cand.get('from_fused_blob_scan'):
            cv2.circle(output, (int(cand['cx']), int(cand['cy'])), int(max(3, cand['r'])), (0, 255, 255), 2)
        # Draw score next to contour center
        text_pos = (int(cand['cx'] + cand['r'] + 10), int(cand['cy']))
        cv2.putText(output, f"{cand['score']:.1f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
    if debug:
        too_far_count = sum(1 for _, reason in contour_statuses if reason == 'too_far')
        passed_scores = [f"{cand['score']:.2f}" for cand in sorted(candidate_contours, key=lambda x: x['score'], reverse=True)]
        print(
            f"[identifyContours] found={len(contours)} | "
            f"checks(area={check_area}, solidity={check_solidity}, circularity={check_circularity}, dist={check_dist}) | "
            f"limits(size=[{sizeMinLimit},{sizeMaxLimit}], circ>={circularityLimit:.2f}, solidity>={SOLIDITY_MIN:.2f}, max_jump={MAX_JUMP_PIXELS}) | "
            f"filtered(area={countourWrongSize}, solidity={countourWrongSolidity}, perimeter={countoursWrongPerimeter}, "
            f"circularity={countoursWrongCircularity}, too_far={too_far_count}) | "
            f"remaining={len(candidate_contours)} scores={passed_scores}"
        )
    # Returner dict for best_ball, så contour kan tegnes
    best_ball_dict = None
    if candidate_contours:
        best_cand = max(candidate_contours, key=lambda x: x['score'])
        best_ball_dict = best_cand

    diagnostics = {
        'total_contours': len(contours),
        'candidate_contours': len(candidate_contours),
        'filtered_area': countourWrongSize,
        'filtered_solidity': countourWrongSolidity,
        'filtered_perimeter': countoursWrongPerimeter,
        'filtered_circularity': countoursWrongCircularity,
        'filtered_too_far': sum(1 for _, reason in contour_statuses if reason == 'too_far'),
        'candidate_balls': [
            {
                'cx': float(cand['cx']),
                'cy': float(cand['cy']),
                'r': float(cand['r']),
                'score': float(cand['score']),
            }
            for cand in candidate_contours
        ],
    }

    if return_diagnostics:
        return best_ball_dict, diagnostics
    return best_ball_dict


def findBall(inputImage, last_ball, tracker, debug, adaptive, player_rect=None):
    output = inputImage.copy()
    hsv = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)
    player_exclusion_mask = None
    if player_rect is not None:
        from utils import create_player_exclusion_mask
        from config import PLAYER_MASK_MARGIN
        player_exclusion_mask = create_player_exclusion_mask(inputImage.shape, player_rect, margin=PLAYER_MASK_MARGIN)

    waterAndBackgroundAndMorphologyMask = createBinaryMask(inputImage, hsv, debug, adaptive, player_exclusion_mask)
    best_ball, diagnostics = identifyContours(
        inputImage,
        waterAndBackgroundAndMorphologyMask,
        output,
        tracker,
        last_ball,
        debug,
        return_diagnostics=True,
    )
    return {'output': output, 'best_ball': best_ball, 'diagnostics': diagnostics}


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse click at: x={x}, y={y}")


def enable_mouse_capture(window_name):
    cv2.setMouseCallback(window_name, mouse_callback)

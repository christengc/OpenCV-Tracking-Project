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
        return False
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


def identifyContours(inputImage, binaryMask, output, tracker, last_ball, debug):
    from config import WEIGHT_Y_BOOST, ML_SCORE_BOOST
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

    # Collect all candidate contours and their scores/components
    candidate_contours = []
    # For color-coding: store tuples (cnt, reason)
    contour_statuses = []
    for cnt in contours:
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
        (x, y), r = cv2.minEnclosingCircle(cnt)
        cx, cy = x, y
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
        classifier_name = getattr(tracker, 'patch_classifier_name', 'ml') or 'ml'
        classifier_model = getattr(tracker, 'patch_classifier', None)
        if classifier_model is None:
            classifier_model = getattr(tracker, 'knn_model', None)
        use_confidence_gate = bool(getattr(tracker, 'patch_classifier_use_confidence_gate', False))
        min_margin = float(getattr(tracker, 'patch_classifier_min_margin', 0.0))
        ml_is_ball, ml_margin = _classify_contour_patch_ml(
            inputImage,
            cx,
            cy,
            classifier_model,
            patch_size=20,
            classifier_name=classifier_name,
            use_confidence_gate=use_confidence_gate,
            min_margin=min_margin,
        )
        if ml_is_ball:
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
        'passed': (0, 255, 0),           # Green
    }
    # Draw contours with color depending on reason
    for cnt, reason in contour_statuses:
        color = color_map.get(reason, (200, 200, 200))
        cv2.drawContours(mask_vis, [cnt], -1, color, 2)
        if reason == 'ml_ball':
            x, y, w, h = cv2.boundingRect(cnt)
            ml_tag = (getattr(tracker, 'patch_classifier_name', 'ml') or 'ml')[:4]
            cv2.putText(mask_vis, ml_tag, (x + w + 6, max(20, y + h // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # Draw score next to each candidate contour
    for cand in candidate_contours:
        text_pos = (int(cand['cx'] + cand['r'] + 10), int(cand['cy']))
        cv2.putText(mask_vis, f"{cand['score']:.1f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

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
    best_ball = identifyContours(inputImage, waterAndBackgroundAndMorphologyMask, output, tracker, last_ball, debug)
    return {'output': output, 'best_ball': best_ball}


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse click at: x={x}, y={y}")


def enable_mouse_capture(window_name):
    cv2.setMouseCallback(window_name, mouse_callback)

import cv2
import numpy as np
import config

from utils import removeComputerGraphics, adaptive_s_threshold, hue2Opencv
from config import (
    PERFORMANCE_MODE,
    WATER_LOWER,
    WATER_UPPER,
    COLOR_SIMILARITY_TOLERANCE,
)

# morphological kernel used by mask creation
kernel3 = np.ones((3,3),np.uint8)


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

    #cv2.imshow('golfBallHSVMask', golfBallHSVMask)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,3))
    #mask_no_club = cv2.morphologyEx(golfBallHSVMask, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('mask_no_club', mask_no_club)

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

    cv2.imshow('totalmask', final_mask)


    from config import GRASS_LOWER, GRASS_UPPER, CLASSIFY_GRASS_MIN_PERCENT
    grass_mask = cv2.inRange(hsv, GRASS_LOWER, GRASS_UPPER)
    grass_percent = np.count_nonzero(grass_mask) / (inputImage.shape[0] * inputImage.shape[1]) * 100
    if grass_percent >= CLASSIFY_GRASS_MIN_PERCENT:
        final_mask = cv2.bitwise_and(final_mask, grass_mask)


    if not PERFORMANCE_MODE:
        cv2.imshow(f'final_mask', final_mask)

    return final_mask


def _calcContourMetrics(cnt):
    """Compute common shape metrics for a single contour.

    Returns a dict with keys:
        area, perimeter, circularity, x, y, w, h, aspect_ratio, solidity
    """
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = max(w, h) / max(1, min(w, h))
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'x': x, 'y': y, 'w': w, 'h': h,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
    }


def _checkFusedBlob(cnt, m, expected_ball_area, binaryMask, blob_num):
    """Detect whether a contour looks like multiple objects fused together.

    Returns (is_fused, statuses, should_skip):
        is_fused    – True if the contour scores as a fused blob
        statuses    – list of (cnt, reason) tuples to add to contour_statuses
        should_skip – True if the caller should `continue` to the next contour
    """
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
    roi = binaryMask[y:y+h, x:x+w]
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


def _scoreBallCandidate(cnt, m, thresholds,
                        last_ball, frames_with_ball, mean_diam, std_diam,
                        WEIGHT_CIRCULARITY, WEIGHT_HISTORY_DIST, WEIGHT_DIAMETER_LIKELIHOOD,
                        WEIGHT_Y_BOOST, frame_height):
    """Apply all ball-detection filters and compute a candidate score.

    Returns (candidate_dict, filter_reason):
        candidate_dict – dict with contour data and score, or None if filtered
        filter_reason  – string reason for filtering, or None if the contour passed
    """
    area, perimeter = m['area'], m['perimeter']
    circularity, solidity = m['circularity'], m['solidity']

    if thresholds['check_area'] and (area < thresholds['sizeMinLimit'] or area > thresholds['sizeMaxLimit']):
        return None, 'wrong_area'

    if thresholds['check_solidity'] and solidity < config.SOLIDITY_MIN:
        return None, 'wrong_solidity'

    if perimeter == 0:
        return None, 'wrong_perimeter'

    if thresholds['check_circularity'] and circularity < thresholds['circularityLimit']:
        return None, 'wrong_circularity'

    (x, y), r = cv2.minEnclosingCircle(cnt)
    cx, cy = x, y
    diameter = r * 2
    diameter_likelihood = np.exp(-((diameter - mean_diam) ** 2.0) / (2.0 * std_diam ** 2.0))

    if last_ball is not None and isinstance(last_ball, dict):
        dist = np.hypot(cx - last_ball['cx'], cy - last_ball['cy'])
        historyWeight = 3.0 * (1 - np.exp(-0.2 * frames_with_ball))
    else:
        dist = 0
        historyWeight = 0

    if thresholds['check_dist'] and dist > config.MAX_JUMP_PIXELS:
        return None, 'too_far'

    dist_norm = dist / (r + 1)
    y_boost = WEIGHT_Y_BOOST if cy > frame_height / 2 else 0
    score = (
        WEIGHT_CIRCULARITY * circularity
        - historyWeight * dist_norm * WEIGHT_HISTORY_DIST
        + WEIGHT_DIAMETER_LIKELIHOOD * diameter_likelihood
        + y_boost
    )
    return {
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
    }, None


def drawCountours(binaryMask, output, contours, contour_statuses, candidate_contours, weight_circularity, weight_history_dist, weight_diameter_likelihood):
    # Visualize contours in the mask image
    mask_vis = cv2.cvtColor(binaryMask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask_vis, contours, -1, (0, 255, 255), 2)
    # Write number of contours in lower left corner
    h, w = mask_vis.shape[:2]
    cv2.putText(mask_vis, f"Contours: {len(contours)}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

    # Color map for reasons
    color_map = {
        'wrong_area': (0, 0, 255),       # Red
        'wrong_solidity': (255, 0, 255), # Magenta
        'wrong_perimeter': (0, 255, 255),# Yellow
        'wrong_circularity': (255, 0, 0),# Blue
        'too_far': (0, 165, 255),        # Orange
        'passed': (0, 255, 0),           # Green
        'fused_blob': (255, 255, 0),     # Cyan
        'splitted_blob': (0, 0, 139),    # Dark red
    }
    # Draw regular contours first, then blob diagnostics last so they stay visible.
    non_fused_statuses = [(cnt, reason) for cnt, reason in contour_statuses if reason not in ('fused_blob', 'splitted_blob')]
    fused_statuses = [(cnt, reason) for cnt, reason in contour_statuses if reason == 'fused_blob']
    splitted_statuses = [(cnt, reason) for cnt, reason in contour_statuses if reason == 'splitted_blob']

    for cnt, reason in non_fused_statuses:
        color = color_map.get(reason, (200, 200, 200))
        cv2.drawContours(mask_vis, [cnt], -1, color, 2)

    for blob_idx, (cnt, _) in enumerate(fused_statuses, start=1):
        cv2.drawContours(mask_vis, [cnt], -1, color_map['fused_blob'], 2)
        x, y, w, h = cv2.boundingRect(cnt)
        label_pos = (x + w + 6, max(20, y + h // 2))
        cv2.putText(mask_vis, str(blob_idx), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    for cnt, _ in splitted_statuses:
        cv2.drawContours(mask_vis, [cnt], -1, color_map['splitted_blob'], 2)

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
            f"  WEIGHT_CIRCULARITY * circularity = {weight_circularity:.1f} * {cand['circularity']:.2f} = {weight_circularity * cand['circularity']:.2f}",
            f"  - historyWeight * dist_norm * WEIGHT_HISTORY_DIST = -{cand['historyWeight']:.2f} * {cand['dist_norm']:.2f} * {weight_history_dist:.2f} = {-cand['historyWeight'] * cand['dist_norm'] * weight_history_dist:.2f}",
            f"  WEIGHT_DIAMETER_LIKELIHOOD * diameter_likelihood = {weight_diameter_likelihood:.1f} * {cand['diameter_likelihood']:.2f} = {weight_diameter_likelihood * cand['diameter_likelihood']:.2f}",
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
    for cand in top_candidates:
        cv2.drawContours(output, [cand['cnt']], -1, (0, 255, 255), 2)
        # Draw score next to contour center
        text_pos = (int(cand['cx'] + cand['r'] + 10), int(cand['cy']))
        cv2.putText(output, f"{cand['score']:.1f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)


def identifyContours(inputImage, binaryMask, output, tracker, last_ball, debug, return_diagnostics=False):
    from config import WEIGHT_Y_BOOST
    thresholds = {
        'check_area': cv2.getTrackbarPos("Area", "Controls") == 1,
        'check_solidity': cv2.getTrackbarPos("Solidity", "Controls") == 1,
        'check_circularity': cv2.getTrackbarPos("Circularity", "Controls") == 1,
        'circularityLimit': cv2.getTrackbarPos("Circ", "Controls") / 100.0,
        'sizeMinLimit': cv2.getTrackbarPos("SzMin", "Controls"),
        'sizeMaxLimit': cv2.getTrackbarPos("SzMax", "Controls"),
        'check_dist': cv2.getTrackbarPos("Dist", "Controls") == 1,
    }
    best_score = -1
    best_ball = None
    # Score weights (define above score calculation)
    WEIGHT_CIRCULARITY = 8.0
    WEIGHT_HISTORY_DIST = 0.1
    WEIGHT_DIAMETER_LIKELIHOOD = 6.0

    contours, _ = cv2.findContours(binaryMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    countourWrongSize = 0
    countoursWrongPerimeter = 0
    countoursWrongCircularity = 0
    countoursCompletedLoop = 0
    countourCorrectScore = 0
    countourWrongSolidity = 0
    expected_ball_area = 0
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
    expected_ball_area = np.pi * (mean_diam / 2) ** 2
    print(f"Expected ball area for scoring: {expected_ball_area:.1f} (mean diameter={mean_diam}, std={std_diam})")
    # Collect all candidate contours and their scores/components
    candidate_contours = []
    # For color-coding: store tuples (cnt, reason)
    contour_statuses = []
    fused_blob_counter = 0
    for cnt in contours:
        m = _calcContourMetrics(cnt)

        if m['aspect_ratio'] > 3:
            continue

        # Check for fused blob
        is_fused, fused_statuses, should_skip = _checkFusedBlob(
            cnt, m, expected_ball_area, binaryMask, fused_blob_counter + 1,
        )
        if is_fused:
            fused_blob_counter += 1
            contour_statuses.extend(fused_statuses)
            if should_skip:
                continue

        # Ball candidate scoring
        candidate, filter_reason = _scoreBallCandidate(
            cnt, m, thresholds,
            last_ball, frames_with_ball, mean_diam, std_diam,
            WEIGHT_CIRCULARITY, WEIGHT_HISTORY_DIST, WEIGHT_DIAMETER_LIKELIHOOD,
            WEIGHT_Y_BOOST, inputImage.shape[0],
        )
        if filter_reason is not None:
            if filter_reason == 'wrong_area':
                countourWrongSize += 1
            elif filter_reason == 'wrong_solidity':
                countourWrongSolidity += 1
            elif filter_reason == 'wrong_perimeter':
                countoursWrongPerimeter += 1
            elif filter_reason == 'wrong_circularity':
                countoursWrongCircularity += 1
            contour_statuses.append((cnt, filter_reason))
            continue

        candidate['is_fused_blob'] = is_fused
        candidate_contours.append(candidate)
        countoursCompletedLoop += 1
        if candidate['score'] > best_score:
            countourCorrectScore += 1
            best_score = candidate['score']
            best_ball = (candidate['cx'], candidate['cy'], candidate['r'])
        # Passed all checks
        contour_statuses.append((cnt, 'passed'))

    drawCountours(
        binaryMask,
        output,
        contours,
        contour_statuses,
        candidate_contours,
        WEIGHT_CIRCULARITY,
        WEIGHT_HISTORY_DIST,
        WEIGHT_DIAMETER_LIKELIHOOD,
    )
    if debug:
        too_far_count = sum(1 for _, reason in contour_statuses if reason == 'too_far')
        passed_scores = [f"{cand['score']:.2f}" for cand in sorted(candidate_contours, key=lambda x: x['score'], reverse=True)]
        print(
            f"[identifyContours] found={len(contours)} | "
            f"checks(area={thresholds['check_area']}, solidity={thresholds['check_solidity']}, circularity={thresholds['check_circularity']}, dist={thresholds['check_dist']}) | "
            f"limits(size=[{thresholds['sizeMinLimit']},{thresholds['sizeMaxLimit']}], circ>={thresholds['circularityLimit']:.2f}, solidity>={config.SOLIDITY_MIN:.2f}, max_jump={config.MAX_JUMP_PIXELS}) | "
            f"filtered(area={countourWrongSize}, solidity={countourWrongSolidity}, perimeter={countoursWrongPerimeter}, "
            f"circularity={countoursWrongCircularity}, too_far={too_far_count}) | "
            f"remaining={len(candidate_contours)} scores={passed_scores}"
        )
    diagnostics = {
        'total_contours': len(contours),
        'candidate_contours': len(candidate_contours),
        'filtered_area': countourWrongSize,
        'filtered_solidity': countourWrongSolidity,
        'filtered_perimeter': countoursWrongPerimeter,
        'filtered_circularity': countoursWrongCircularity,
        'filtered_too_far': sum(1 for _, reason in contour_statuses if reason == 'too_far'),
    }

    # Returner dict for best_ball, så contour kan tegnes
    best_ball_dict = None
    if candidate_contours:
        best_cand = max(candidate_contours, key=lambda x: x['score'])
        best_ball_dict = best_cand
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

import cv2
import numpy as np

from config import TRACKBAR_SETTINGS

# functions used by controls will be imported lazily to avoid circular dependency

# HOG person detector shared instance (initialised on first use)
_hog_detector = None

# YOLOv8 detector shared instance (initialised on first use)
_yolo_detector = None


def nothing(x):
    pass


def initialize_controls():
    """Create trackbars and window used for interactive parameter tuning.

    The names, initial values and ranges are defined in ``config.TRACKBAR_SETTINGS``
    so that they can be inspected or modified from a single location.
    """
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 900, 700)
    # Use shorter labels for controls to avoid truncation
    label_map = {
        "H min": "Hmin",
        "H max": "Hmax",
        "S min": "Smin",
        "S max": "Smax",
        "V min": "Vmin",
        "V max": "Vmax",
        "Size min": "SzMin",
        "Size Max": "SzMax",
        "circularity": "Circ",
    }
    for name, (initial, maximum) in TRACKBAR_SETTINGS.items():
        short_name = label_map.get(name, name)
        cv2.createTrackbar(short_name, "Controls", initial, maximum, nothing)
    # Add checkboxes for toggling checks in identifyContours
    cv2.createTrackbar("Area", "Controls", 1, 1, nothing)
    cv2.createTrackbar("Solidity", "Controls", 1, 1, nothing)
    cv2.createTrackbar("Circularity", "Controls", 1, 1, nothing)
    cv2.createTrackbar("Dist", "Controls", 1, 1, nothing)  # 1=on, 0=off


def removeComputerGraphics(image, x1, y1, x2, y2):
    image[y1:y2, x1:x2] = 0
    return image


def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)


def hue2Opencv(worldValue):
    return (worldValue/255)*180


def adaptive_s_threshold(inputImage, percentile_value=15, min_v=45):
    # Nedskalér inputbilledet til 25% for hurtigere beregning
    scale = 0.25
    h0, w0 = inputImage.shape[:2]
    new_size = (int(w0 * scale), int(h0 * scale))
    small_img = cv2.resize(inputImage, new_size)

    hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mu = np.mean(s)
    sigma = np.std(s)

    threshold = mu - 1.5 * sigma
    if(threshold < min_v):
        threshold = min_v

    return threshold


def draw_ball_overlay(output, golfBall, framesWithoutBall):
    """Render ball location, prediction circles, and arrow based on tracker state."""
    lastPosition = golfBall.getLastKnownPosition()
    if lastPosition is None:
        return output

    # Try to get radius from lastPosition if available
    r = getattr(lastPosition, 'r', None)
    if r is None:
        # Fallback to default radius if not available
        r = 10

    if golfBall.framesWithoutBall == 0:
        x, y = lastPosition.x, lastPosition.y
        cv2.circle(output, (int(x), int(y)), int(r), (0,0,255), 2)
        cv2.circle(output, (int(x), int(y)), int(r+2), (0,255,0), 2)
        cv2.circle(output, (int(x), int(y)), int(r+4), (255,0,0), 2)

        state = golfBall.getCurrentData()
        if state is not None and golfBall.framesWithBall>5 and state.speed >5:
            startPoint = (int(state.x), int(state.y))
            endPoint = (
                int(state.x + state.dx * 5),
                int(state.y + state.dy * 5)
            )
            cv2.arrowedLine(output, startPoint, endPoint, (0,255,0), 5)

    elif golfBall.framesWithoutBall < 15:
        x = lastPosition.x + lastPosition.dx*(framesWithoutBall+1)
        y = lastPosition.y + lastPosition.dy*(framesWithoutBall+1)
        cv2.circle(output, (int(x), int(y)), int(r), (0,0,255), 2)
        cv2.circle(output, (int(x), int(y)), int(r+2), (0,255,0), 2)
        cv2.circle(output, (int(x), int(y)), int(r+4), (255,0,0), 2)

    return output


def handle_keyboard_controls(key, paused, last_ball, adaptive, image, currentFrameNumber, vid):
    """
    Handles keyboard input and updates state accordingly.
    Returns updated state dictionary.
    """
    should_break = False

    if key == ord('q'):        # quit
        should_break = True

    elif key == ord('p'):      # pause / play toggle
        paused = not paused
        print("Paused, current frame:", vid.get(cv2.CAP_PROP_POS_FRAMES))

    elif key == ord('r'):      # reset tracker
        last_ball = None

    elif key == ord('b'):      # debug and show binary mask and contour information
        # import here to avoid circular import
        from ball_detector import createBinaryMask, identifyContours
        mask = createBinaryMask(image, cv2.cvtColor(image, cv2.COLOR_BGR2HSV), True, True)
        identifyContours(image, mask, image, None, None, True)

    elif key == ord('w'):      # save current frame
        success = cv2.imwrite("debugFrame.jpg", image)
        print("Saved current frame as debugFrame.jpg:", success)

    elif key == ord('n'):  # rewind
        print("rewind")
        vid.set(cv2.CAP_PROP_POS_FRAMES, currentFrameNumber - 30)

    elif key == ord('m'):  # forward
        print("forward")
        vid.set(cv2.CAP_PROP_POS_FRAMES, currentFrameNumber + 30)

    elif key == ord('a'):      # adaptive mode toggle
        adaptive = not adaptive
        print("Adaptive mode:", adaptive)

    elif key == ord('d'):      # debug flag toggle
        import config
        config.DEBUG_FLAG = not config.DEBUG_FLAG
        print("debugFlag:", config.DEBUG_FLAG)

    elif key == ord('t'):      # toggle player detection method
        import config
        config.DETECTION_METHOD = "yolov8" if config.DETECTION_METHOD == "hog" else "hog"
        print(f"Player detection method: {config.DETECTION_METHOD}")

    return {
        "paused": paused,
        "last_ball": last_ball,
        "adaptive": adaptive,
        "should_break": should_break
    }


def detect_players_hog(frame):
    """Run HOG+SVM pedestrian detector on the given frame.

    Uses OpenCV's default people detector. Returns a list of
    bounding boxes (x,y,w,h). The detector is initialised once per
    session to avoid expensive recreation on every frame.
    """
    global _hog_detector
    if _hog_detector is None:
        _hog_detector = cv2.HOGDescriptor()
        _hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # run detection on the whole frame; scale/stride settings are typical
    rects, weights = _hog_detector.detectMultiScale(
        frame, winStride=(8, 8), padding=(16, 16), scale=1.05
    )
    # convert to contours for tight fit
    contours = []
    for (x, y, w, h) in rects:
        contours.append(np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]]))
    return contours


def detect_players_yolov8(frame):
    """Run YOLOv8 person detector on the given frame.

    Uses Ultralytics YOLOv8 model to detect people. Returns a list of
    bounding boxes (x,y,w,h). The model is loaded once on first use.
    Requires: pip install ultralytics
    """
    global _yolo_detector
    if _yolo_detector is None:
        try:
            from ultralytics import YOLO
            _yolo_detector = YOLO("yolov8n-seg.pt")  # nano segmentation model for tight fit
            # Forsøg at flytte modellen til GPU hvis tilgængelig
            try:
                _yolo_detector.to('cuda')
                _yolo_detector.half()
                print("YOLOv8 kører nu på GPU (CUDA) med half precision (FP16)")
            except Exception as e:
                print(f"Kunne ikke flytte YOLOv8 til GPU eller aktivere half precision: {e}")
             # Frys model weights her:
            _yolo_detector.eval()
        except ImportError:
            print("WARNING: ultralytics not installed. Run: pip install ultralytics")
            return []

    # Nedskaler inputbilledet til 50% i både bredde og højde
    scale = 0.25
    h, w = frame.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    frame_small = cv2.resize(frame, new_size)

    # run inference and keep only person detections (class 0)
    # Brug YOLOv8 med mindre inputstørrelse (imgsz=256)
    results = _yolo_detector(frame_small, verbose=False, imgsz=256, classes=[0])
    contours = []
    for result in results:
        boxes = result.boxes
        # Try to get masks from result.masks
        mask_data = None
        if hasattr(result, 'masks') and result.masks is not None:
            mask_data = result.masks.data.cpu().numpy()  # shape: (num_masks, H, W)
        for i, box in enumerate(boxes):
            if int(box.cls[0]) == 0:
                # Use mask from result.masks if available and index matches
                if mask_data is not None and i < mask_data.shape[0]:
                    mask = mask_data[i]
                    # Opskalér masken til original størrelse
                    mask = cv2.resize(mask, (w, h))
                    mask = (mask > 0.5).astype(np.uint8)
                    # Dilate mask to make it looser
                    dilate_kernel = np.ones((15, 15), np.uint8)
                    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        largest = max(cnts, key=cv2.contourArea)
                        contours.append(largest)
                else:
                    # Opskalér boks-koordinater til original størrelse
                    x1, y1, x2, y2 = [int(coord / scale) for coord in box.xyxy[0]]
                    contours.append(np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]]))
    return contours


def detect_players(frame):
    """Detect players using the configured detection method.

    Supports HOG (fast, less accurate) and YOLOv8 (slower, more accurate).
    Method is controlled by config.DETECTION_METHOD.
    """
    import time
    from config import DETECTION_METHOD
    # --- Interpolering og frame-tælling ---
    if not hasattr(detect_players, "_frame_counter"):
        detect_players._frame_counter = 0
        detect_players._last_contours = None
        detect_players._prev_contours = None
    detect_players._frame_counter += 1

    if DETECTION_METHOD == "yolov8":
        # Kør YOLOv8 på hver frame.
        result = detect_players_yolov8(frame)
        detect_players._prev_contours = detect_players._last_contours
        detect_players._last_contours = result
        contours = result
    else:
        result = detect_players_hog(frame)
        contours = result
    return {'contours': contours}


def create_player_exclusion_mask(frame_shape, player_rect, margin=20):
    """Create a binary mask that excludes the player's body but allows detection at feet.

    The mask is white (255) everywhere except the player's upper/middle body (black/0).
    The bottom area (feet) remains open so the ball can be detected there.

    Args:
        frame_shape: Shape tuple (height, width) of the frame
        player_rect: Bounding box tuple (x, y, w, h) from detector
        margin: Pixels to expand the exclusion zone on sides

    Returns:
        Binary mask where 0 = exclude (player body), 255 = include (rest + feet)
    """
    mask = np.ones((frame_shape[0], frame_shape[1]), dtype=np.uint8) * 255
    if player_rect is not None:
        # player_rect is now a contour
        shrink_kernel = np.ones((13, 13), np.uint8)
        contour_mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
        cv2.drawContours(contour_mask, [player_rect], -1, 255, -1)
        contour_mask = cv2.erode(contour_mask, shrink_kernel, iterations=1)
        mask[contour_mask > 0] = 0
    return mask


def detect_graphics_rectangles(frame):
    """Detect computer graphics (rectangular UI elements) in the frame.

    Uses Canny edge detection and contour analysis to find rectangular
    static graphics elements. Returns a list of bounding boxes (x,y,w,h)
    for detected rectangles.

    Args:
        frame: Input frame (BGR)

    Returns:
        List of tuples (x, y, w, h) representing detected rectangular regions
    """
    from config import (
        GRAPHICS_MIN_AREA,
        GRAPHICS_ASPECT_RATIO_RANGE,
        GRAPHICS_EDGE_THRESHOLD,
    )
    # convert to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection (lower threshold for more sensitivity)
    edges = cv2.Canny(blurred, int(GRAPHICS_EDGE_THRESHOLD * 0.7), int(GRAPHICS_EDGE_THRESHOLD * 1.5))

    # Adaptive threshold for strong graphics (helps with lighting)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours from both edge and threshold images
    contours_canny, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_thresh, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_canny + contours_thresh

    rectangles = []
    min_ar, max_ar = GRAPHICS_ASPECT_RATIO_RANGE
    min_area = int(GRAPHICS_MIN_AREA * 0.7)  # slightly lower area threshold

    height, width = frame.shape[:2]
    margin_h = int(height * 0.2)
    margin_w = int(width * 0.2)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w > 0 and h > 0:
            aspect_ratio = w / h
            # Only allow rectangles in the perimeter (outer 20% of image)
            in_left = x < margin_w
            in_right = x + w > width - margin_w
            in_top = y < margin_h
            in_bottom = y + h > height - margin_h
            if min_ar <= aspect_ratio <= max_ar and (in_left or in_right or in_top or in_bottom):
                rectangles.append((x, y, w, h))

    # Merge overlapping rectangles (graphics often overlap)
    def rect_overlap(r1, r2):
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    merged = []
    for rect in rectangles:
        found = False
        for i, m in enumerate(merged):
            if rect_overlap(rect, m):
                # merge rectangles
                x1 = min(rect[0], m[0])
                y1 = min(rect[1], m[1])
                x2 = max(rect[0] + rect[2], m[0] + m[2])
                y2 = max(rect[1] + rect[3], m[1] + m[3])
                merged[i] = (x1, y1, x2 - x1, y2 - y1)
                found = True
                break
        if not found:
            merged.append(rect)

    # --- Persistent rectangle tracking for static graphics ---
    if not hasattr(detect_graphics_rectangles, "rect_history"):
        detect_graphics_rectangles.rect_history = {}
    rect_history = detect_graphics_rectangles.rect_history
    FRAME_THRESHOLD = 3  # number of frames rectangle must persist
    POS_TOLERANCE = 10   # pixels tolerance for position matching

    def rect_key(x, y, w, h):
        # Round position for tolerance
        return (int(x / POS_TOLERANCE), int(y / POS_TOLERANCE), int(w / POS_TOLERANCE), int(h / POS_TOLERANCE))

    # Update history: increment count for persistent rectangles, reset for new ones
    new_keys = set()
    for x, y, w, h in merged:
        key = rect_key(x, y, w, h)
        new_keys.add(key)
        if key in rect_history:
            rect_history[key]["count"] += 1
            rect_history[key]["rect"] = (x, y, w, h)
        else:
            rect_history[key] = {"count": 1, "rect": (x, y, w, h)}

    # Remove rectangles not seen in this frame
    for key in list(rect_history.keys()):
        if key not in new_keys:
            rect_history[key]["count"] -= 1
            if rect_history[key]["count"] <= 0:
                del rect_history[key]

    # Only keep rectangles present for enough frames
    persistent_rects = [v["rect"] for v in rect_history.values() if v["count"] >= FRAME_THRESHOLD]

    # Draw only persistent rectangles
    for x, y, w, h in persistent_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.imshow("graphics_detected", frame)

    return persistent_rects


def create_graphics_exclusion_mask(frame_shape, rectangles, margin=5):
    """Create a binary mask that excludes detected graphics rectangles.

    Args:
        frame_shape: Shape tuple (height, width) of the frame
        rectangles: List of bounding boxes (x, y, w, h) from graphics detection
        margin: Pixels to expand exclusion zone around each rectangle

    Returns:
        Binary mask where 0 = exclude (graphics), 255 = include (rest)
    """
    mask = np.ones((frame_shape[0], frame_shape[1]), dtype=np.uint8) * 255
    print("graphics found: ",len(rectangles) )
    for x, y, w, h in rectangles:
        # expand rectangle with margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame_shape[1], x + w + margin)
        y2 = min(frame_shape[0], y + h + margin)
        # set excluded region to black
        mask[y1:y2, x1:x2] = 0
    
    return mask

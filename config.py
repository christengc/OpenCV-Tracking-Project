"""Global configuration and default constants used throughout the tracker.

Putting hard‑coded numbers in a central module makes it easier to tune
behaviour and eliminates magic literals scattered across the codebase.
"""

# video playback / UI
DEFAULT_PLAY_SPEED = 1
DEFAULT_START_FRAME = 1
WAIT_KEY_DELAY_MS = 30  # argument passed to cv2.waitKey
SCENE_SHIFT_THRESHOLD = 40.0

# tracker behaviour
MAX_FRAMES_WITHOUT_BALL = 10    # after this many frames the previous ball is forgotten
MAX_JUMP_PIXELS = 1000          # maximum allowed jump for candidate contour

# contours and mask
COLOR_SIMILARITY_TOLERANCE = 20  # tolerance for R/G/B channel differences

# water/mirror mask thresholds (HSV)
WATER_LOWER = (80, 20, 50)
WATER_UPPER = (140, 255, 255)

# trackbar defaults: name -> (initial value, maximum value)
TRACKBAR_SETTINGS = {
    "H min": (0, 179),
    "H max": (255, 179),
    "S min": (0, 255),
    "S max": (65, 255),
    "V min": (0, 255),
    "V max": (255, 255),
    "circularity": (67, 100),
    "Size min": (50, 50),
    "Size Max": (4000, 90000),
}

# scene classification ranges (HSV)
import numpy as np

SKY_LOWER = np.array([90, 0, 100])
SKY_UPPER = np.array([130, 100, 255])
GRASS_LOWER = np.array([35, 40, 40])
GRASS_UPPER = np.array([85, 255, 255])
AUDIENCE_LOWER = np.array([0, 0, 0])
AUDIENCE_UPPER = np.array([180, 255, 60])  # tighter V upper bound for audience

# thresholds for deciding classification based on percentage
CLASSIFY_SKY_MIN_PERCENT = 30
CLASSIFY_GRASS_MIN_PERCENT = 25

# overlay indicator
FRAME_INDICATOR_SIZE = 50
FRAME_INDICATOR_MARGIN = 10
FRAME_INDICATOR_COLORS = {
    "sky": (255, 255, 255),
    "blue_sky": (255, 0, 0),   # Pure blue for blue sky (BGR)
    "white_sky": (255, 255, 255), # White for white sky
    "grass": (0, 255, 0),
    "audience": (0, 0, 0),
}

# player detection method: "hog" or "yolov8"
DETECTION_METHOD = "yolov8"

# player masking during ball detection
PLAYER_MASK_MARGIN = 20  # pixels of margin around player box (allows edge detection)

# video file path
DEFAULT_VIDEO_PATH = "video2.mp4"

# computer graphics (rectangles) detection masking
DETECT_GRAPHICS = False  # enable/disable graphics detection
# minimum area (pixels²) for graphics; 100x100 box → 10000
GRAPHICS_MIN_AREA = 20000
GRAPHICS_ASPECT_RATIO_RANGE = (0.3, 3.0)  # acceptable rectangle shapes
GRAPHICS_EDGE_THRESHOLD = 70  # Canny edge detection threshold

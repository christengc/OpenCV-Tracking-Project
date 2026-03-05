
import time
import cv2
import numpy as np

from calculate_scene import CalculateScene
from golf_ball import GolfBall
from ball_detector import createBinaryMask, identifyContours, findBall
from utils import (
    draw_ball_overlay,
    handle_keyboard_controls,
    hue2Opencv,
    adaptive_s_threshold,
    zoom,
    removeComputerGraphics,
    initialize_controls,
    detect_players,
)
from config import (
    DEFAULT_PLAY_SPEED,
    DEFAULT_START_FRAME,
    WAIT_KEY_DELAY_MS,
    SCENE_SHIFT_THRESHOLD,
    MAX_FRAMES_WITHOUT_BALL,
)


class VideoProcessor:

    def calculate_iou(self, boxA, boxB):
        # boxA and boxB: [x, y, w, h]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0
        return iou
        """Handles video reading, frame processing and user interaction."""

    def __init__(self, video_path: str, play_speed: int = DEFAULT_PLAY_SPEED, start_frame: int = DEFAULT_START_FRAME):
        self.video_path = video_path
        self.play_speed = play_speed
        self.start_frame_number = start_frame

        self.vid = cv2.VideoCapture(video_path)
        self.framespersecond = int(self.vid.get(cv2.CAP_PROP_FPS))

        # prepare controls window and trackbars
        initialize_controls()

        self.sceneCalculator = CalculateScene()
        self.golfBall = GolfBall()

        self.paused = False
        self.adaptive = True
        self.debugFlag = False

        self.last_ball = None
        self.framesWithoutBall = 0
        self.previous_frame = None
        self.frameCount = 0

        # set initial position
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame_number)

        # Initialize lists for diameter statistics
        self.estimated_diameters = []
        self.actual_diameters = []
        self.diameter_diffs = []
        self.diameter_history = []  # For diameter likelihood scoring

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Mouse click at: x={x}, y={y}")

    def enable_mouse_capture(self, window_name):
        cv2.setMouseCallback(window_name, self.mouse_callback)

    def run(self):
        import os, json
        # Indlæs COCO-annotationer én gang
        coco_path = os.path.join(os.path.dirname(__file__), 'instances_Train.json')
        if os.path.exists(coco_path):
            with open(coco_path, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            # Lav opslag fra image_id til bbox
            coco_ann = {}
            for ann in coco_data.get('annotations', []):
                img_id = ann.get('image_id')
                bbox = ann.get('bbox')
                if img_id is not None and bbox:
                    coco_ann[img_id] = bbox
        else:
            coco_ann = {}

        success = True
        while success:
            start = time.time()
            lastFrameNumber = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
            currentFrameNumber = lastFrameNumber + self.play_speed

            if not self.paused:
                self.vid.set(cv2.CAP_PROP_POS_FRAMES, currentFrameNumber)
                success, image = self.vid.read()
            else:
                image = None

            if success and not self.paused:
                # scene shift detection
                if self.previous_frame is not None:
                    result = self.sceneCalculator.detect_scene_shift(
                        self.previous_frame, image, threshold=SCENE_SHIFT_THRESHOLD
                    )
                    if result["is_scene_shift"]:
                        print(f"Scene shift detected! Score: {result['shift_score']}")
                        # Reset diameter history on scene shift
                        self.diameter_history = []
                self.previous_frame = image

                classification_result = self.sceneCalculator.classify_frame(image)

                # detect player in grass or blue sky scenes for masking during ball detection
                player_rect = None
                scene_type = classification_result.get("classification")
                allow_tracking = scene_type in ["grass", "blue_sky"]
                actual_ball_diameter_pixels = None
                if allow_tracking:
                    rects = detect_players(image)
                    if rects:
                        # keep only the largest contour (assumed golfer)
                        player_rect = max(rects, key=cv2.contourArea)

                # find ball (excluding player area if in allowed scene)
                if allow_tracking:
                    output, best_ball = findBall(
                        image, self.last_ball, self, self.debugFlag, self.adaptive,
                        player_rect=player_rect
                    )
                    # Get actual ball diameter from best_ball if available
                    if best_ball is not None and 'r' in best_ball:
                        actual_ball_diameter_pixels = best_ball['r'] * 2  # radius to diameter
                        # Update diameter history
                        self.diameter_history.append(actual_ball_diameter_pixels)
                        if len(self.diameter_history) > 10:
                            self.diameter_history = self.diameter_history[-10:]
                else:
                    output = image.copy()
                    best_ball = None
                output = self.sceneCalculator.draw_frame_type_indicator(output, classification_result)

                # Print ball size estimation and actual size
                if player_rect is not None:
                    estimated_ball_diameter_pixels = self.sceneCalculator.estimate_ball_size_from_player(player_rect, image.shape, actual_ball_diameter_pixels)
                    if estimated_ball_diameter_pixels is not None:
                        self.estimated_diameters.append(estimated_ball_diameter_pixels)
                    if actual_ball_diameter_pixels is not None:
                        self.actual_diameters.append(actual_ball_diameter_pixels)
                    if estimated_ball_diameter_pixels is not None and actual_ball_diameter_pixels is not None:
                        self.diameter_diffs.append(estimated_ball_diameter_pixels - actual_ball_diameter_pixels)

                if best_ball is not None:
                    self.golfBall.addData({'x': best_ball['cx'], 'y': best_ball['cy']})
                else:
                    self.golfBall.addData(None)

                if best_ball is not None:
                    self.last_ball = best_ball
                else:
                    self.framesWithoutBall += 1
                    if self.framesWithoutBall > MAX_FRAMES_WITHOUT_BALL:
                        self.last_ball = None

                # if player was detected, draw their contour

                # Tegn best_ball contour hvis den findes
                if best_ball is not None and 'cx' in best_ball and 'cy' in best_ball and 'r' in best_ball:
                    center = (int(best_ball['cx']), int(best_ball['cy']))
                    radius = int(best_ball['r'])
                    cv2.circle(output, center, max(radius-2,1), (255, 0, 0), 2)   # r-1, rød
                    cv2.circle(output, center, radius, (0, 255, 0), 2)           # r, grøn
                    cv2.circle(output, center, radius+2, (0, 0, 255), 2)         # r+1, blå

                if player_rect is not None:
                    cv2.drawContours(output, [player_rect], -1, (255, 0, 0), 2)

                # Draw purple rectangle for ground truth ball location from COCO annotation
                frame_id = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_id in coco_ann:
                    bbox = coco_ann[frame_id]
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    # If best_ball exists, calculate IoU and print to terminal
                    if best_ball is not None and 'cx' in best_ball and 'cy' in best_ball and 'r' in best_ball:
                        det_x = int(best_ball['cx'] - best_ball['r'])
                        det_y = int(best_ball['cy'] - best_ball['r'])
                        det_w = int(2 * best_ball['r'])
                        det_h = int(2 * best_ball['r'])
                        det_bbox = [det_x, det_y, det_w, det_h]
                        iou = self.calculate_iou(det_bbox, [x, y, w, h])
                        print(f"Frame {frame_id}: IoU={iou:.3f}")

                cv2.imshow('frame output', output)
                self.enable_mouse_capture('frame output')
                self.frameCount += 1

            end = time.time()
            algorithm_time = int((end - start) * 1000)
            target_time = 30 - algorithm_time
            if target_time < 0:
                target_time = 1

            key = cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF
            control_result = handle_keyboard_controls(
                key,
                self.paused,
                self.last_ball,
                self.adaptive,
                self.debugFlag,
                output if 'output' in locals() else image,
                self.vid.get(cv2.CAP_PROP_POS_FRAMES),
                self.vid,
            )

            self.paused = control_result["paused"]
            self.last_ball = control_result["last_ball"]
            self.adaptive = control_result["adaptive"]
            self.debugFlag = control_result["debugFlag"]

            if control_result["should_break"]:
                break

        self.vid.release()
        cv2.destroyAllWindows()

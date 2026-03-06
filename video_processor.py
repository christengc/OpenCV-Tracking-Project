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
from evaluation_class import EvaluationClass
from config import (
    DEFAULT_PLAY_SPEED,
    DEFAULT_START_FRAME,
    WAIT_KEY_DELAY_MS,
    SCENE_SHIFT_THRESHOLD,
    MAX_FRAMES_WITHOUT_BALL,
)
import config


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

    def __init__(self, video_path: str, play_speed: int = DEFAULT_PLAY_SPEED, start_frame: int = DEFAULT_START_FRAME, max_frames: int = None):
        self.video_path = video_path
        self.play_speed = play_speed
        self.start_frame_number = start_frame
        self.max_frames = max_frames

        self.vid = cv2.VideoCapture(video_path)
        self.framespersecond = int(self.vid.get(cv2.CAP_PROP_FPS))

        # prepare controls window and trackbars
        initialize_controls()

        self.sceneCalculator = CalculateScene()
        self.golfBall = GolfBall()

        self.paused = False
        self.adaptive = True
        self.last_ball = None
        self.framesWithoutBall = 0
        self.previous_frame = None
        self.frameCount = 0
        self.stop_requested = False

        # set initial position
        from config import PERFORMANCE_MODE
        if not PERFORMANCE_MODE:
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
        self.stop_requested = False
        evaluation = EvaluationClass()
        found_ball_frames = 0
        iou_evaluated_frames = 0
        iou_match_frames = 0
        grass_or_blue_frames = 0
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
        processed_frames = 0
        import time
        while success:
            t0 = time.time()
            lastFrameNumber = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
            currentFrameNumber = lastFrameNumber + self.play_speed

            t1 = time.time()
            if not self.paused:
                from config import PERFORMANCE_MODE
                if not PERFORMANCE_MODE:
                    self.vid.set(cv2.CAP_PROP_POS_FRAMES, currentFrameNumber)
                success, image = self.vid.read()
            else:
                image = None
            t2 = time.time()

            if success and not self.paused:
                t3 = time.time()
                # Only run scene shift and classify every other frame
                run_analysis = (self.frameCount % 2 == 0)
                if run_analysis:
                    # scene shift detection
                    if self.previous_frame is not None:
                        t31 = time.time()
                        t31a = time.time()
                        result = self.sceneCalculator.detect_scene_shift(
                            self.previous_frame, image, threshold=SCENE_SHIFT_THRESHOLD
                        )
                        t31b = time.time()
                        if result["is_scene_shift"]:
                            print(f"Scene shift detected! Score: {result['shift_score']}")
                            self.diameter_history = []
                    self.previous_frame = image

                    t4 = time.time()
                    classification_result = self.sceneCalculator.classify_frame(image)
                    t5 = time.time()
                else:
                    # Use previous results if not recalculating
                    classification_result = getattr(self, '_last_classification_result', None)
                self._last_classification_result = classification_result

                # detect player in grass or blue sky scenes for masking during ball detection
                player_rect = None
                scene_type = classification_result.get("classification")
                allow_tracking = scene_type in ["grass", "blue_sky"]
                if allow_tracking:
                    grass_or_blue_frames += 1
                actual_ball_diameter_pixels = None
                if allow_tracking:
                    t51 = time.time()
                    # --- Profile internals of detect_players ---
                    t51a = time.time()
                    rects = detect_players(image)
                    t51b = time.time()
                    contours = rects['contours'] if isinstance(rects, dict) and 'contours' in rects else rects
                    if contours:
                        # keep only the largest contour (assumed golfer)
                        player_rect = max(contours, key=cv2.contourArea)

                # find ball (excluding player area if in allowed scene)
                if allow_tracking:
                    t61 = time.time()
                    # --- Profile internals of findBall ---
                    t61a = time.time()
                    findball_result = findBall(
                        image, self.last_ball, self, config.DEBUG_FLAG, self.adaptive,
                        player_rect=player_rect
                    )
                    output = findball_result['output']
                    best_ball = findball_result['best_ball']
                    t61b = time.time()
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
                t7 = time.time()
                output = self.sceneCalculator.draw_frame_type_indicator(output, classification_result)
                t8 = time.time()

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
                    found_ball_frames += 1
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
                        iou_evaluated_frames += 1
                        if iou > 0.5:
                            iou_match_frames += 1
                        evaluation.add_iou(iou)

                t9 = time.time()
                cv2.imshow('frame output', output)
                self.enable_mouse_capture('frame output')
                self.frameCount += 1
                t10 = time.time()


            t11 = time.time()
            algorithm_time = int((t11 - t0) * 1000)
            target_time = 30 - algorithm_time
            if target_time < 0:
                target_time = 1

            processed_frames += 1
            if self.max_frames is not None and processed_frames >= self.max_frames:
                current_pos = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
                print(f"[STOP] reason=processed_frames_limit processed={processed_frames} video_frame={current_pos} max_frames={self.max_frames}")
                break

            # Also stop by actual video frame index to guarantee deterministic
            # cut-off for parameter sweeps (e.g. first 4000 frames).
            if self.max_frames is not None:
                current_pos = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
                processed_video_span = current_pos - int(self.start_frame_number)
                if processed_video_span >= self.max_frames:
                    print(f"[STOP] reason=video_frame_limit processed={processed_frames} video_frame={current_pos} span={processed_video_span} max_frames={self.max_frames}")
                    break

            key = cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF
            control_result = handle_keyboard_controls(
                key,
                self.paused,
                self.last_ball,
                self.adaptive,
                output if 'output' in locals() else image,
                self.vid.get(cv2.CAP_PROP_POS_FRAMES),
                self.vid,
            )

            self.paused = control_result["paused"]
            self.last_ball = control_result["last_ball"]
            self.adaptive = control_result["adaptive"]

            if control_result["should_break"]:
                self.stop_requested = True
                break

        self.vid.release()
        cv2.destroyAllWindows()
        # Print summary score after video ends
        final_score = evaluation.summary_score()
        print(f"\nSummary IoU score: {final_score:.3f}")
        final_video_frame = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
        total_video_frames = max(0, final_video_frame - int(self.start_frame_number))
        run_metrics = {
            "summary_iou": final_score,
            "total_video_frames": total_video_frames,
            "processed_loop_frames": processed_frames,
            "found_ball_frames": found_ball_frames,
            "iou_evaluated_frames": iou_evaluated_frames,
            "iou_match_frames": iou_match_frames,
            "grass_or_blue_frames": grass_or_blue_frames,
            "stop_requested": self.stop_requested,
        }
        print(
            "[RUN METRICS] "
            f"total_video_frames={run_metrics['total_video_frames']} | "
            f"found_ball={run_metrics['found_ball_frames']} | "
            f"iou_match_gt_0.5={run_metrics['iou_match_frames']} | "
            f"grass_or_blue={run_metrics['grass_or_blue_frames']}"
        )
        return run_metrics

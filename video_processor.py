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
    collect_annotation_and_random_patch_features,
)
from evaluation_class import EvaluationClass
from config import (
    DEFAULT_PLAY_SPEED,
    DEFAULT_START_FRAME,
    WAIT_KEY_DELAY_MS,
    SCENE_SHIFT_THRESHOLD,
    MAX_FRAMES_WITHOUT_BALL,
    KNN_DISTANCE_QUANTILE,
    KNN_DISTANCE_MARGIN,
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
        self.knn_model = None
        self.knn_train_features = None
        self.knn_train_labels = None
        self.knn_distance_threshold = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Mouse click at: x={x}, y={y}")

    def enable_mouse_capture(self, window_name):
        cv2.setMouseCallback(window_name, self.mouse_callback)

    def initialize_knn_model(self, coco_ann, max_frames=2000, patch_size=20):
        """Build training data from annotation/random patches and train KNN."""
        try:
            import importlib
            sklearn_neighbors = importlib.import_module("sklearn.neighbors")
            KNeighborsClassifier = getattr(sklearn_neighbors, "KNeighborsClassifier")
        except ImportError:
            print("KNN init skipped: scikit-learn is not installed")
            self.knn_model = None
            return

        feature_matrix, label_vector = collect_annotation_and_random_patch_features(
            self.vid,
            coco_ann,
            start_frame=self.start_frame_number,
            max_frames=max_frames,
            patch_size=patch_size,
            random_seed=0,
        )

        if feature_matrix.shape[1] == 0 or label_vector.size == 0:
            print("KNN init skipped: no training samples collected")
            self.knn_model = None
            self.knn_train_features = feature_matrix
            self.knn_train_labels = label_vector
            return

        X_train = feature_matrix.T
        y_train = label_vector

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        # Estimate a practical distance threshold from positive samples.
        pos_idx = np.where(y_train == 1)[0]
        if pos_idx.size >= 2:
            pos_X = X_train[pos_idx]
            pos_dists, _ = knn.kneighbors(pos_X, n_neighbors=2)
            nearest_nonself = pos_dists[:, 1]
            distance_threshold = float(np.quantile(nearest_nonself, KNN_DISTANCE_QUANTILE) * KNN_DISTANCE_MARGIN)
        elif pos_idx.size == 1:
            pos_X = X_train[pos_idx]
            pos_dists, _ = knn.kneighbors(pos_X, n_neighbors=1)
            distance_threshold = float(pos_dists[0, 0] * KNN_DISTANCE_MARGIN)
        else:
            distance_threshold = None

        train_pred = knn.predict(X_train)
        train_acc = float(np.mean(train_pred == y_train)) if y_train.size > 0 else 0.0
        train_pos_pred = int(np.sum(train_pred == 1))

        self.knn_model = knn
        self.knn_train_features = X_train
        self.knn_train_labels = y_train
        self.knn_distance_threshold = distance_threshold
        setattr(self.knn_model, "distance_threshold", distance_threshold)

        print(
            f"KNN data shape: X={X_train.shape}, y={y_train.shape}, "
            f"class_balance=(ones={int(np.sum(y_train == 1))}, zeros={int(np.sum(y_train == 0))})"
        )
        print(
            f"KNN training sanity: accuracy={train_acc:.3f}, "
            f"predicted_ones={train_pos_pred}"
        )
        if distance_threshold is not None:
            print(f"KNN distance threshold: {distance_threshold:.3f}")
        else:
            print("KNN distance threshold: disabled (insufficient positive samples)")
        print("Training complete")

    def run(self):
        self.stop_requested = False
        evaluation = EvaluationClass()
        found_ball_frames = 0
        iou_evaluated_frames = 0
        iou_match_frames = 0
        grass_or_blue_frames = 0
        label_frames = 0
        label_tracked_frames = 0
        label_not_tracked_frames = 0
        iou_low_or_no_score_frames = 0
        no_label_frames = 0
        no_label_tracked_ball_frames = 0
        no_label_not_tracked_frames = 0
        no_label_not_tracked_background_frames = 0
        no_label_not_tracked_detector_frames = 0
        no_label_detector_no_contours_frames = 0
        no_label_detector_filtered_area_frames = 0
        no_label_detector_filtered_solidity_frames = 0
        no_label_detector_filtered_circularity_frames = 0
        no_label_detector_filtered_perimeter_frames = 0
        no_label_detector_filtered_distance_frames = 0
        no_label_detector_other_frames = 0
        label_detector_no_contours_frames = 0
        label_detector_filtered_area_frames = 0
        label_detector_filtered_solidity_frames = 0
        label_detector_filtered_circularity_frames = 0
        label_detector_filtered_perimeter_frames = 0
        label_detector_filtered_distance_frames = 0
        label_detector_other_frames = 0
        label_not_tracked_background_frames = 0
        label_not_tracked_detector_frames = 0
        no_label_skipped_background_frames = 0
        import os, json
        # Indlæs COCO-annotationer én gang fra split/outline labels
        coco_path = os.path.join(os.path.dirname(__file__), 'coco splitandoutline.json')
        if os.path.exists(coco_path):
            with open(coco_path, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            # Lav opslag fra image_id til labelinfo (bbox + occluded)
            coco_ann = {}
            for ann in coco_data.get('annotations', []):
                img_id = ann.get('image_id')
                bbox = ann.get('bbox')
                attributes = ann.get('attributes') or {}
                occluded = bool(attributes.get('occluded', False))
                if img_id is not None and bbox:
                    existing = coco_ann.get(img_id)
                    if existing is None or (existing.get('occluded', True) and not occluded):
                        coco_ann[img_id] = {
                            'bbox': bbox,
                            'occluded': occluded,
                        }
            print(f"[LABELS] using={os.path.basename(coco_path)} annotations={len(coco_ann)}")
        else:
            coco_ann = {}
            print(f"[LABELS] missing required file: {os.path.basename(coco_path)}")

        self.initialize_knn_model(
            coco_ann,
            max_frames=2000,
            patch_size=20,
        )

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
                    target_frame = max(0, int(lastFrameNumber + self.play_speed - 1))
                    self.vid.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
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
                    detector_diagnostics = findball_result.get('diagnostics')
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
                    detector_diagnostics = None
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
                label_info = coco_ann.get(frame_id)
                is_occluded_label = label_info is not None and label_info.get('occluded', False)
                has_valid_label = label_info is not None and not is_occluded_label

                def detector_primary_reason(diag):
                    if not diag:
                        return 'other'
                    if diag.get('total_contours', 0) == 0:
                        return 'no_contours'
                    candidates = {
                        'filtered_area': diag.get('filtered_area', 0),
                        'filtered_solidity': diag.get('filtered_solidity', 0),
                        'filtered_circularity': diag.get('filtered_circularity', 0),
                        'filtered_perimeter': diag.get('filtered_perimeter', 0),
                        'filtered_distance': diag.get('filtered_too_far', 0),
                    }
                    reason, count = max(candidates.items(), key=lambda item: item[1])
                    return reason if count > 0 else 'other'

                if has_valid_label:
                    label_frames += 1
                    bbox = label_info['bbox']
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    # If best_ball exists, calculate IoU and print to terminal
                    if best_ball is not None and 'cx' in best_ball and 'cy' in best_ball and 'r' in best_ball:
                        label_tracked_frames += 1
                        det_x = int(best_ball['cx'] - best_ball['r'])
                        det_y = int(best_ball['cy'] - best_ball['r'])
                        det_w = int(2 * best_ball['r'])
                        det_h = int(2 * best_ball['r'])
                        det_bbox = [det_x, det_y, det_w, det_h]
                        iou = self.calculate_iou(det_bbox, [x, y, w, h])
                        iou_evaluated_frames += 1
                        if iou > 0.5:
                            iou_match_frames += 1
                        else:
                            iou_low_or_no_score_frames += 1
                        evaluation.add_iou(iou)
                    else:
                        label_not_tracked_frames += 1
                        if not allow_tracking:
                            label_not_tracked_background_frames += 1
                        else:
                            label_not_tracked_detector_frames += 1
                            detector_reason = detector_primary_reason(detector_diagnostics)
                            if detector_reason == 'no_contours':
                                label_detector_no_contours_frames += 1
                            elif detector_reason == 'filtered_area':
                                label_detector_filtered_area_frames += 1
                            elif detector_reason == 'filtered_solidity':
                                label_detector_filtered_solidity_frames += 1
                            elif detector_reason == 'filtered_circularity':
                                label_detector_filtered_circularity_frames += 1
                            elif detector_reason == 'filtered_perimeter':
                                label_detector_filtered_perimeter_frames += 1
                            elif detector_reason == 'filtered_distance':
                                label_detector_filtered_distance_frames += 1
                            else:
                                label_detector_other_frames += 1
                else:
                    no_label_frames += 1
                    if best_ball is not None:
                        no_label_tracked_ball_frames += 1
                    else:
                        no_label_not_tracked_frames += 1
                        if not allow_tracking:
                            no_label_skipped_background_frames += 1
                            no_label_not_tracked_background_frames += 1
                        else:
                            no_label_not_tracked_detector_frames += 1
                            detector_reason = detector_primary_reason(detector_diagnostics)
                            if detector_reason == 'no_contours':
                                no_label_detector_no_contours_frames += 1
                            elif detector_reason == 'filtered_area':
                                no_label_detector_filtered_area_frames += 1
                            elif detector_reason == 'filtered_solidity':
                                no_label_detector_filtered_solidity_frames += 1
                            elif detector_reason == 'filtered_circularity':
                                no_label_detector_filtered_circularity_frames += 1
                            elif detector_reason == 'filtered_perimeter':
                                no_label_detector_filtered_perimeter_frames += 1
                            elif detector_reason == 'filtered_distance':
                                no_label_detector_filtered_distance_frames += 1
                            else:
                                no_label_detector_other_frames += 1

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
            "label_frames": label_frames,
            "label_tracked_frames": label_tracked_frames,
            "label_not_tracked_frames": label_not_tracked_frames,
            "label_not_tracked_background_frames": label_not_tracked_background_frames,
            "label_not_tracked_detector_frames": label_not_tracked_detector_frames,
            "label_detector_no_contours_frames": label_detector_no_contours_frames,
            "label_detector_filtered_area_frames": label_detector_filtered_area_frames,
            "label_detector_filtered_solidity_frames": label_detector_filtered_solidity_frames,
            "label_detector_filtered_circularity_frames": label_detector_filtered_circularity_frames,
            "label_detector_filtered_perimeter_frames": label_detector_filtered_perimeter_frames,
            "label_detector_filtered_distance_frames": label_detector_filtered_distance_frames,
            "label_detector_other_frames": label_detector_other_frames,
            "iou_evaluated_frames": iou_evaluated_frames,
            "iou_match_frames": iou_match_frames,
            "iou_low_or_no_score_frames": iou_low_or_no_score_frames,
            "grass_or_blue_frames": grass_or_blue_frames,
            "no_label_frames": no_label_frames,
            "no_label_tracked_ball_frames": no_label_tracked_ball_frames,
            "no_label_not_tracked_frames": no_label_not_tracked_frames,
            "no_label_not_tracked_background_frames": no_label_not_tracked_background_frames,
            "no_label_not_tracked_detector_frames": no_label_not_tracked_detector_frames,
            "no_label_detector_no_contours_frames": no_label_detector_no_contours_frames,
            "no_label_detector_filtered_area_frames": no_label_detector_filtered_area_frames,
            "no_label_detector_filtered_solidity_frames": no_label_detector_filtered_solidity_frames,
            "no_label_detector_filtered_circularity_frames": no_label_detector_filtered_circularity_frames,
            "no_label_detector_filtered_perimeter_frames": no_label_detector_filtered_perimeter_frames,
            "no_label_detector_filtered_distance_frames": no_label_detector_filtered_distance_frames,
            "no_label_detector_other_frames": no_label_detector_other_frames,
            "no_label_skipped_background_frames": no_label_skipped_background_frames,
            "stop_requested": self.stop_requested,
        }

        return run_metrics

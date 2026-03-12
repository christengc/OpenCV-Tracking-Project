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
    SVM_MAX_TRAIN_FRAMES,
    SVM_PATCH_SIZE,
    SVM_TEST_RATIO,
    SVM_NEGATIVE_PATCH_RATIO,
    SVM_KERNEL,
    SVM_C,
    SVM_GAMMA,
    SVM_CLASS_WEIGHT,
    SVM_ENABLE_CONFIDENCE_GATE,
    SVM_MIN_DECISION_MARGIN,
    KALMAN_ENABLED,
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_MAX_DEVIATION_PIXELS,
    KALMAN_MAX_PREDICTION_WITHOUT_CANDIDATES,
)
import config


def collect_annotation_and_random_patch_features_local(
    video_capture,
    coco_ann,
    start_frame=1,
    max_frames=2000,
    patch_size=20,
    negative_ratio=1,
    random_seed=0,
):
    """Collect paired annotation/random patch features and labels."""
    feature_len = patch_size * patch_size
    feature_columns = []
    labels = []
    half = patch_size // 2
    rng = np.random.default_rng(random_seed)
    negative_ratio = max(1, int(negative_ratio))

    def _to_feature_column(patch_bgr):
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.reshape(feature_len, 1).astype(np.uint8)

    def _extract_center_patch(frame, bbox):
        x, y, w, h = [float(v) for v in bbox[:4]]
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        padded = cv2.copyMakeBorder(
            frame, half, half, half, half, cv2.BORDER_REFLECT_101
        )
        px = cx + half
        py = cy + half
        patch = padded[py - half:py + half, px - half:px + half]
        if patch.shape[:2] != (patch_size, patch_size):
            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        return patch

    def _overlaps_bbox(px, py, bbox):
        x, y, w, h = [float(v) for v in bbox[:4]]
        return not (px + patch_size <= x or px >= x + w or py + patch_size <= y or py >= y + h)

    def _extract_random_non_overlap_patch(frame, bbox, max_attempts=200):
        h, w = frame.shape[:2]
        if w < patch_size or h < patch_size:
            return None
        max_x = w - patch_size
        max_y = h - patch_size
        for _ in range(max_attempts):
            px = int(rng.integers(0, max_x + 1))
            py = int(rng.integers(0, max_y + 1))
            if _overlaps_bbox(px, py, bbox):
                continue
            return frame[py:py + patch_size, px:px + patch_size]
        return None

    original_pos = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
    first_idx = max(0, int(start_frame) - 1)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, first_idx)

    for _ in range(max_frames):
        ok, frame = video_capture.read()
        if not ok:
            break

        frame_id = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        ann_entry = coco_ann.get(frame_id)
        if ann_entry is None:
            continue

        if isinstance(ann_entry, dict):
            candidates = [ann_entry]
        else:
            candidates = [a for a in ann_entry if isinstance(a, dict)]

        bbox = None
        for ann in candidates:
            b = ann.get("bbox")
            if b is not None and len(b) >= 4:
                bbox = b
                break
        if bbox is None:
            continue

        ann_patch = _extract_center_patch(frame, bbox)
        feature_columns.append(_to_feature_column(ann_patch))
        labels.append(1)

        # Collect multiple negatives per positive to better reflect class imbalance.
        for _ in range(negative_ratio):
            rand_patch = _extract_random_non_overlap_patch(frame, bbox)
            if rand_patch is None:
                continue
            feature_columns.append(_to_feature_column(rand_patch))
            labels.append(0)

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

    if not feature_columns:
        return np.empty((feature_len, 0), dtype=np.uint8), np.empty((0,), dtype=np.uint8)

    feature_matrix = np.hstack(feature_columns)
    label_vector = np.array(labels, dtype=np.uint8)
    return feature_matrix, label_vector


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
        self.patch_classifier = None
        self.patch_classifier_name = None
        self.patch_classifier_patch_size = int(SVM_PATCH_SIZE)
        self.patch_classifier_use_confidence_gate = bool(SVM_ENABLE_CONFIDENCE_GATE)
        self.patch_classifier_min_margin = float(SVM_MIN_DECISION_MARGIN)
        # Backward compatibility for existing detector code paths
        self.knn_model = None

        self.use_kalman = bool(KALMAN_ENABLED)
        self.kalman_max_deviation_pixels = float(KALMAN_MAX_DEVIATION_PIXELS)
        self.kalman_filter = cv2.KalmanFilter(4, 2)
        self.kalman_filter.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        self.kalman_filter.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            np.float32,
        )
        self.kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * float(KALMAN_PROCESS_NOISE)
        self.kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(KALMAN_MEASUREMENT_NOISE)
        self.kalman_filter.errorCovPost = np.eye(4, dtype=np.float32)
        self.kalman_initialized = False
        self.kalman_last_prediction = None
        self.kalman_frames_without_candidates = 0
        self.kalman_max_prediction_without_candidates = int(KALMAN_MAX_PREDICTION_WITHOUT_CANDIDATES)
        self._scene_shift_display_until = 0.0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Mouse click at: x={x}, y={y}")

    def enable_mouse_capture(self, window_name):
        cv2.setMouseCallback(window_name, self.mouse_callback)

    def _kalman_predict(self):
        if not self.use_kalman or not self.kalman_initialized:
            return None
        prediction = self.kalman_filter.predict()
        pred_x = float(prediction[0, 0])
        pred_y = float(prediction[1, 0])
        self.kalman_last_prediction = (pred_x, pred_y)
        return self.kalman_last_prediction

    def _kalman_correct(self, x, y):
        if not self.use_kalman:
            return
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman_filter.correct(measurement)

    def _kalman_initialize(self, x, y):
        self.kalman_filter.statePost = np.array(
            [[np.float32(x)], [np.float32(y)], [np.float32(0.0)], [np.float32(0.0)]],
            dtype=np.float32,
        )
        self.kalman_initialized = True
        self.kalman_last_prediction = (float(x), float(y))

    def _pick_candidate_with_kalman(self, candidate_balls, predicted_xy):
        if not candidate_balls:
            return None
        if predicted_xy is None:
            return max(candidate_balls, key=lambda c: c.get('score', 0.0))

        pred_x, pred_y = predicted_xy
        nearest = min(
            candidate_balls,
            key=lambda c: np.hypot(float(c['cx']) - pred_x, float(c['cy']) - pred_y),
        )
        deviation = float(np.hypot(float(nearest['cx']) - pred_x, float(nearest['cy']) - pred_y))
        if deviation <= self.kalman_max_deviation_pixels:
            return nearest
        return None

    def _choose_ball_with_motion_model(self, detector_best_ball, detector_diagnostics):
        candidate_balls = []
        if isinstance(detector_diagnostics, dict):
            candidate_balls = detector_diagnostics.get('candidate_balls') or []
        has_candidates = bool(candidate_balls) or detector_best_ball is not None

        if not self.use_kalman:
            return detector_best_ball

        if has_candidates:
            self.kalman_frames_without_candidates = 0
        else:
            self.kalman_frames_without_candidates += 1

        predicted_xy = self._kalman_predict()

        if not self.kalman_initialized:
            if detector_best_ball is not None:
                self._kalman_initialize(detector_best_ball['cx'], detector_best_ball['cy'])
                return detector_best_ball
            return None

        chosen_candidate = self._pick_candidate_with_kalman(candidate_balls, predicted_xy)
        if chosen_candidate is not None:
            self._kalman_correct(chosen_candidate['cx'], chosen_candidate['cy'])
            chosen = dict(chosen_candidate)
            chosen['from_kalman_prediction'] = False
            return chosen

        if self.kalman_frames_without_candidates > self.kalman_max_prediction_without_candidates:
            self.kalman_initialized = False
            self.kalman_last_prediction = None
            return None

        if predicted_xy is not None:
            radius = 8.0
            if detector_best_ball is not None and 'r' in detector_best_ball:
                radius = float(detector_best_ball['r'])
            elif self.last_ball is not None and 'r' in self.last_ball:
                radius = float(self.last_ball['r'])
            return {
                'cx': float(predicted_xy[0]),
                'cy': float(predicted_xy[1]),
                'r': float(max(3.0, radius)),
                'score': float('-inf'),
                'from_kalman_prediction': True,
            }

        return detector_best_ball

    def _reset_tracking_state(self, reason="manual"):
        self.last_ball = None
        self.framesWithoutBall = 0
        self.previous_frame = None
        self._last_classification_result = None
        self.frameCount = 0
        self.diameter_history = []
        self.kalman_initialized = False
        self.kalman_last_prediction = None
        self.kalman_frames_without_candidates = 0
        self.golfBall = GolfBall()
        print(f"[TRACK RESET] reason={reason}")

    def initialize_svm_model(self, coco_ann, max_frames=2000, patch_size=20):
        """Train an SVM classifier on annotation/random patches."""
        try:
            import importlib

            sklearn_svm = importlib.import_module("sklearn.svm")
            sklearn_pipeline = importlib.import_module("sklearn.pipeline")
            sklearn_preprocessing = importlib.import_module("sklearn.preprocessing")
            SVC = getattr(sklearn_svm, "SVC")
            make_pipeline = getattr(sklearn_pipeline, "make_pipeline")
            StandardScaler = getattr(sklearn_preprocessing, "StandardScaler")
        except ImportError:
            print("SVM init skipped: scikit-learn is not installed")
            self.patch_classifier = None
            self.patch_classifier_name = None
            self.knn_model = None
            return

        feature_matrix, label_vector = collect_annotation_and_random_patch_features_local(
            self.vid,
            coco_ann,
            start_frame=self.start_frame_number,
            max_frames=max_frames,
            patch_size=patch_size,
            negative_ratio=SVM_NEGATIVE_PATCH_RATIO,
            random_seed=0,
        )

        if feature_matrix.shape[1] == 0 or label_vector.size == 0:
            print("SVM init skipped: no training samples collected")
            self.patch_classifier = None
            self.patch_classifier_name = None
            self.knn_model = None
            return

        X_all = feature_matrix.T
        y_all = label_vector

        def _binary_metrics(y_true, y_pred):
            y_true = np.asarray(y_true).astype(np.uint8)
            y_pred = np.asarray(y_pred).astype(np.uint8)
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            total = y_true.size
            accuracy = float((tp + tn) / total) if total > 0 else 0.0
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }

        pos_idx = np.where(y_all == 1)[0]
        neg_idx = np.where(y_all == 0)[0]
        if pos_idx.size == 0 or neg_idx.size == 0:
            print("SVM init skipped: training data does not contain both classes")
            self.patch_classifier = None
            self.patch_classifier_name = None
            self.knn_model = None
            return

        rng = np.random.default_rng(42)
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        test_ratio = float(SVM_TEST_RATIO)
        pos_test_count = max(1, int(round(pos_idx.size * test_ratio)))
        neg_test_count = max(1, int(round(neg_idx.size * test_ratio)))
        if pos_idx.size - pos_test_count < 1:
            pos_test_count = max(0, pos_idx.size - 1)
        if neg_idx.size - neg_test_count < 1:
            neg_test_count = max(0, neg_idx.size - 1)

        test_idx = np.concatenate([
            pos_idx[:pos_test_count],
            neg_idx[:neg_test_count],
        ])
        rng.shuffle(test_idx)
        train_mask = np.ones(y_all.size, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]

        X_test = X_all[test_idx] if test_idx.size > 0 else np.empty((0, X_all.shape[1]), dtype=X_all.dtype)
        y_test = y_all[test_idx] if test_idx.size > 0 else np.empty((0,), dtype=y_all.dtype)

        unique_classes = np.unique(y_train)
        if unique_classes.size < 2:
            print("SVM init skipped: training data only contains one class")
            self.patch_classifier = None
            self.patch_classifier_name = None
            self.knn_model = None
            return

        svm = make_pipeline(
            StandardScaler(),
            SVC(
                kernel=SVM_KERNEL,
                C=float(SVM_C),
                gamma=SVM_GAMMA,
                class_weight=SVM_CLASS_WEIGHT,
            ),
        )
        svm.fit(X_train, y_train)

        train_pred = svm.predict(X_train)
        train_metrics = _binary_metrics(y_train, train_pred)

        if y_test.size > 0:
            test_pred = svm.predict(X_test)
            test_metrics = _binary_metrics(y_test, test_pred)
        else:
            test_metrics = None

        self.patch_classifier = svm
        self.patch_classifier_name = "svm"
        self.patch_classifier_patch_size = int(patch_size)
        # Backward compatibility with existing detector code paths.
        self.knn_model = svm

        print(
            f"SVM data shape: X_all={X_all.shape}, y_all={y_all.shape}, "
            f"train={X_train.shape[0]}, test={X_test.shape[0]}, "
            f"class_balance_all=(ones={int(np.sum(y_all == 1))}, zeros={int(np.sum(y_all == 0))}), "
            f"neg_ratio_target={int(SVM_NEGATIVE_PATCH_RATIO)}"
        )
        print(
            "SVM train metrics: "
            f"accuracy={train_metrics['accuracy']:.3f}, "
            f"precision={train_metrics['precision']:.3f}, "
            f"recall={train_metrics['recall']:.3f}, "
            f"f1={train_metrics['f1']:.3f}, "
            f"cm=[[TN={train_metrics['tn']}, FP={train_metrics['fp']}], "
            f"[FN={train_metrics['fn']}, TP={train_metrics['tp']}]]"
        )
        if test_metrics is not None:
            print(
                "SVM test metrics: "
                f"accuracy={test_metrics['accuracy']:.3f}, "
                f"precision={test_metrics['precision']:.3f}, "
                f"recall={test_metrics['recall']:.3f}, "
                f"f1={test_metrics['f1']:.3f}, "
                f"cm=[[TN={test_metrics['tn']}, FP={test_metrics['fp']}], "
                f"[FN={test_metrics['fn']}, TP={test_metrics['tp']}]]"
            )
        else:
            print("SVM test metrics: skipped (insufficient test samples)")
        print("SVM training complete")

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
        # IndlÃ¦s COCO-annotationer Ã©n gang fra split/outline labels
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

        self.initialize_svm_model(
            coco_ann,
            max_frames=SVM_MAX_TRAIN_FRAMES,
            patch_size=SVM_PATCH_SIZE,
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
                        if result["shift_score"] > 5.0:
                            frame_no = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))
                            print(f"[SHIFT SCORE] frame={frame_no} score={result['shift_score']} hist={result['histogram_diff']} diff={result['frame_diff']} edge={result.get('edge_diff','?')} (threshold={SCENE_SHIFT_THRESHOLD})")
                        if result["is_scene_shift"]:
                            print(f"Scene shift detected! Score: {result['shift_score']}")
                            self._reset_tracking_state(reason="scene_shift")
                            self._scene_shift_display_until = time.time() + 1.0
                    self.previous_frame = image

                    t4 = time.time()
                    classification_result = self.sceneCalculator.classify_frame(image)
                    t5 = time.time()

                    # Secondary check: if scene classification type changed, also treat as scene shift
                    prev_classification = getattr(self, '_last_classification_result', None)
                    if prev_classification is not None and classification_result is not None:
                        prev_type = prev_classification.get("classification")
                        new_type = classification_result.get("classification")
                        if prev_type != new_type and time.time() > self._scene_shift_display_until:
                            print(f"Scene type change detected: {prev_type} -> {new_type}")
                            self._reset_tracking_state(reason="scene_type_change")
                            self._scene_shift_display_until = time.time() + 1.0
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
                    detector_best_ball = findball_result['best_ball']
                    detector_diagnostics = findball_result.get('diagnostics')
                    best_ball = self._choose_ball_with_motion_model(detector_best_ball, detector_diagnostics)
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
                    cv2.circle(output, center, max(radius-2,1), (255, 0, 0), 2)   # r-1, rÃ¸d
                    cv2.circle(output, center, radius, (0, 255, 0), 2)           # r, grÃ¸n
                    cv2.circle(output, center, radius+2, (0, 0, 255), 2)         # r+1, blÃ¥

                    source_is_kalman = bool(best_ball.get('from_kalman_prediction', False))
                    source_text = "KALMAN" if source_is_kalman else "CANDIDATE"
                    source_color = (0, 165, 255) if source_is_kalman else (0, 255, 0)
                    text_origin = (max(10, center[0] + radius + 12), max(24, center[1] - radius - 12))
                    cv2.putText(
                        output,
                        source_text,
                        text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        source_color,
                        2,
                        cv2.LINE_AA,
                    )

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

                # Live confusion-matrix counters to monitor which scenes increase errors.
                strict_tp = iou_match_frames
                strict_fn = label_not_tracked_frames + iou_low_or_no_score_frames
                strict_fp = no_label_tracked_ball_frames
                strict_tn = no_label_not_tracked_frames
                any_tp = label_tracked_frames
                any_fn = label_not_tracked_frames
                any_fp = no_label_tracked_ball_frames
                any_tn = no_label_not_tracked_frames

                cm_lines = [
                    f"CM IoU>0.5 TP:{strict_tp} FN:{strict_fn} FP:{strict_fp} TN:{strict_tn}",
                    f"CM AnyDet TP:{any_tp} FN:{any_fn} FP:{any_fp} TN:{any_tn}",
                    f"Scene:{scene_type}",
                ]
                cm_font = cv2.FONT_HERSHEY_SIMPLEX
                cm_scale = 0.52
                cm_thick = 1
                cm_x = 12
                cm_y = 22
                cm_step = 18
                for idx, line in enumerate(cm_lines):
                    y = cm_y + idx * cm_step
                    cv2.putText(output, line, (cm_x + 1, y + 1), cm_font, cm_scale, (0, 0, 0), cm_thick + 2, cv2.LINE_AA)
                    cv2.putText(output, line, (cm_x, y), cm_font, cm_scale, (255, 255, 255), cm_thick, cv2.LINE_AA)

                # Draw centered scene-shift notification for 1 second
                if time.time() < self._scene_shift_display_until:
                    h_fr, w_fr = output.shape[:2]
                    sc_text = "SCENE SHIFT"
                    sc_font = cv2.FONT_HERSHEY_SIMPLEX
                    sc_scale = 1.4
                    sc_thick = 3
                    (tw, th), _ = cv2.getTextSize(sc_text, sc_font, sc_scale, sc_thick)
                    tx = (w_fr - tw) // 2
                    ty = (h_fr + th) // 2
                    cv2.putText(output, sc_text, (tx + 2, ty + 2), sc_font, sc_scale, (0, 0, 0), sc_thick + 2, cv2.LINE_AA)
                    cv2.putText(output, sc_text, (tx, ty), sc_font, sc_scale, (0, 200, 255), sc_thick, cv2.LINE_AA)

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

            if control_result.get("reset_tracking", False):
                reset_reason = "seek" if control_result.get("did_seek", False) else "manual_reset"
                self._reset_tracking_state(reason=reset_reason)

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
        print(
            "[RUN METRICS] "
            f"total_video_frames={run_metrics['total_video_frames']} | "
            f"found_ball={run_metrics['found_ball_frames']} | "
            f"grass_or_blue={run_metrics['grass_or_blue_frames']}"
        )
        print("[RUN METRICS HIERARCHY]")
        print(f"  1a no_label={run_metrics['no_label_frames']}")
        print(f"    1a1 tracked={run_metrics['no_label_tracked_ball_frames']}")
        print(f"    1a2 not_tracked={run_metrics['no_label_not_tracked_frames']}")
        print(
            f"      reasons: background_classification={run_metrics['no_label_not_tracked_background_frames']}, "
            f"detector_no_ball_found={run_metrics['no_label_not_tracked_detector_frames']}"
        )
        print(
            f"        detector subreasons: no_contours={run_metrics['no_label_detector_no_contours_frames']}, "
            f"filtered_area={run_metrics['no_label_detector_filtered_area_frames']}, "
            f"filtered_solidity={run_metrics['no_label_detector_filtered_solidity_frames']}, "
            f"filtered_circularity={run_metrics['no_label_detector_filtered_circularity_frames']}, "
            f"filtered_perimeter={run_metrics['no_label_detector_filtered_perimeter_frames']}, "
            f"filtered_distance={run_metrics['no_label_detector_filtered_distance_frames']}, "
            f"other={run_metrics['no_label_detector_other_frames']}"
        )
        print(f"  1b label={run_metrics['label_frames']}")
        print(f"    1b1 tracked={run_metrics['label_tracked_frames']}")
        print(f"      1b1a IoU>0.5={run_metrics['iou_match_frames']}")
        print(f"      1b1b low_or_no_score={run_metrics['iou_low_or_no_score_frames']}")
        print(f"    1b2 not_tracked={run_metrics['label_not_tracked_frames']}")
        print(
            f"      reasons: background_classification={run_metrics['label_not_tracked_background_frames']}, "
            f"detector_no_ball_found={run_metrics['label_not_tracked_detector_frames']}"
        )
        print(
            f"        detector subreasons: no_contours={run_metrics['label_detector_no_contours_frames']}, "
            f"filtered_area={run_metrics['label_detector_filtered_area_frames']}, "
            f"filtered_solidity={run_metrics['label_detector_filtered_solidity_frames']}, "
            f"filtered_circularity={run_metrics['label_detector_filtered_circularity_frames']}, "
            f"filtered_perimeter={run_metrics['label_detector_filtered_perimeter_frames']}, "
            f"filtered_distance={run_metrics['label_detector_filtered_distance_frames']}, "
            f"other={run_metrics['label_detector_other_frames']}"
        )
        return run_metrics


import csv
import os


class EvaluationClass:
    TEST_BASELINE_PARAMETERS = {
        "SOLIDITY_MIN": 0.80,
        "MAX_JUMP_PIXELS": 150,
        "WEIGHT_Y_BOOST": 1.0,
        "circularity": 67,
        "Size min": 50,
        "Size Max": 20000,
    }
    TEST_PROGRESS_CSV = "test_suite_progress.csv"

    @staticmethod
    def _csv_fieldnames():
        return [
            "test_name",
            "swept_param",
            "swept_value",
            "max_frames",
            "wait_key_delay_ms",
            "input_solidity_min",
            "input_circularity",
            "input_size_min",
            "input_size_max",
            "input_max_jump_pixels",
            "input_weight_y_boost",
            "summary_iou",
            "total_video_frames",
            "processed_loop_frames",
            "found_ball_frames",
            "label_frames",
            "label_tracked_frames",
            "label_not_tracked_frames",
            "iou_evaluated_frames",
            "iou_match_frames",
            "iou_low_or_no_score_frames",
            "grass_or_blue_frames",
            "no_label_frames",
            "no_label_tracked_ball_frames",
            "no_label_not_tracked_frames",
            "stop_requested",
        ]

    @staticmethod
    def _append_progress_csv_row(
        config_module,
        run_result,
        test_name,
        swept_param,
        swept_value,
        run_max_frames,
    ):
        csv_path = os.path.join(os.getcwd(), EvaluationClass.TEST_PROGRESS_CSV)
        file_exists = os.path.exists(csv_path)
        row = {
            "test_name": test_name,
            "swept_param": swept_param,
            "swept_value": swept_value,
            "max_frames": int(run_max_frames),
            "wait_key_delay_ms": int(getattr(config_module, "WAIT_KEY_DELAY_MS", 0)),
            "input_solidity_min": float(getattr(config_module, "SOLIDITY_MIN", 0.0)),
            "input_circularity": int(config_module.TRACKBAR_SETTINGS.get("circularity", (0, 100))[0]),
            "input_size_min": int(config_module.TRACKBAR_SETTINGS.get("Size min", (0, 0))[0]),
            "input_size_max": int(config_module.TRACKBAR_SETTINGS.get("Size Max", (0, 0))[0]),
            "input_max_jump_pixels": int(getattr(config_module, "MAX_JUMP_PIXELS", 0)),
            "input_weight_y_boost": float(getattr(config_module, "WEIGHT_Y_BOOST", 0.0)),
            "summary_iou": float(run_result.get("summary_iou", 0.0)),
            "total_video_frames": int(run_result.get("total_video_frames", 0)),
            "processed_loop_frames": int(run_result.get("processed_loop_frames", 0)),
            "found_ball_frames": int(run_result.get("found_ball_frames", 0)),
            "label_frames": int(run_result.get("label_frames", 0)),
            "label_tracked_frames": int(run_result.get("label_tracked_frames", 0)),
            "label_not_tracked_frames": int(run_result.get("label_not_tracked_frames", 0)),
            "iou_evaluated_frames": int(run_result.get("iou_evaluated_frames", 0)),
            "iou_match_frames": int(run_result.get("iou_match_frames", 0)),
            "iou_low_or_no_score_frames": int(run_result.get("iou_low_or_no_score_frames", 0)),
            "grass_or_blue_frames": int(run_result.get("grass_or_blue_frames", 0)),
            "no_label_frames": int(run_result.get("no_label_frames", 0)),
            "no_label_tracked_ball_frames": int(run_result.get("no_label_tracked_ball_frames", 0)),
            "no_label_not_tracked_frames": int(run_result.get("no_label_not_tracked_frames", 0)),
            "stop_requested": bool(run_result.get("stop_requested", False)),
        }

        with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=EvaluationClass._csv_fieldnames())
            if not file_exists or os.path.getsize(csv_path) == 0:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _capture_test_parameter_state(config_module):
        return {
            "SOLIDITY_MIN": config_module.SOLIDITY_MIN,
            "MAX_JUMP_PIXELS": config_module.MAX_JUMP_PIXELS,
            "WEIGHT_Y_BOOST": config_module.WEIGHT_Y_BOOST,
            "WAIT_KEY_DELAY_MS": config_module.WAIT_KEY_DELAY_MS,
            "TRACKBAR_SETTINGS": {
                "circularity": tuple(config_module.TRACKBAR_SETTINGS["circularity"]),
                "Size min": tuple(config_module.TRACKBAR_SETTINGS["Size min"]),
                "Size Max": tuple(config_module.TRACKBAR_SETTINGS["Size Max"]),
            },
        }

    @staticmethod
    def _restore_test_parameter_state(config_module, state):
        config_module.SOLIDITY_MIN = state["SOLIDITY_MIN"]
        config_module.MAX_JUMP_PIXELS = state["MAX_JUMP_PIXELS"]
        config_module.WEIGHT_Y_BOOST = state["WEIGHT_Y_BOOST"]
        config_module.WAIT_KEY_DELAY_MS = state["WAIT_KEY_DELAY_MS"]
        for key, value in state["TRACKBAR_SETTINGS"].items():
            config_module.TRACKBAR_SETTINGS[key] = value

    @staticmethod
    def _apply_loose_test_baseline(config_module, excluded_keys=None):
        excluded = set(excluded_keys or [])
        baseline = EvaluationClass.TEST_BASELINE_PARAMETERS

        if "SOLIDITY_MIN" not in excluded:
            config_module.SOLIDITY_MIN = float(baseline["SOLIDITY_MIN"])
        if "MAX_JUMP_PIXELS" not in excluded:
            config_module.MAX_JUMP_PIXELS = int(baseline["MAX_JUMP_PIXELS"])
        if "WEIGHT_Y_BOOST" not in excluded:
            config_module.WEIGHT_Y_BOOST = float(baseline["WEIGHT_Y_BOOST"])
        if "circularity" not in excluded:
            EvaluationClass._set_trackbar_setting(config_module, "circularity", baseline["circularity"])
        if "Size min" not in excluded:
            EvaluationClass._set_trackbar_setting(config_module, "Size min", baseline["Size min"])
        if "Size Max" not in excluded:
            EvaluationClass._set_trackbar_setting(config_module, "Size Max", baseline["Size Max"])

    @staticmethod
    def _print_loose_test_baseline(excluded_keys=None):
        excluded = set(excluded_keys or [])
        baseline = EvaluationClass.TEST_BASELINE_PARAMETERS
        details = []
        for key in ["SOLIDITY_MIN", "circularity", "Size min", "Size Max", "WEIGHT_Y_BOOST", "MAX_JUMP_PIXELS"]:
            marker = " (swept)" if key in excluded else ""
            details.append(f"{key}={baseline[key]}{marker}")
        print("[TEST BASELINE] " + ", ".join(details))

    @staticmethod
    def _set_trackbar_setting(config_module, key, value):
        current_max = config_module.TRACKBAR_SETTINGS[key][1]
        config_module.TRACKBAR_SETTINGS[key] = (int(value), current_max)

    @staticmethod
    def _apply_common_parameters(config_module, common_params, excluded_keys=None):
        if not common_params:
            return

        excluded = set(excluded_keys or [])
        for key, value in common_params.items():
            if key in excluded:
                continue

            if key in config_module.TRACKBAR_SETTINGS:
                EvaluationClass._set_trackbar_setting(config_module, key, value)
                continue

            if hasattr(config_module, key):
                setattr(config_module, key, value)
                continue

            print(f"[WARN] Unknown common parameter skipped: {key}")

    @staticmethod
    def _resolve_max_frames(override_max_frames, default_max_frames):
        return int(override_max_frames) if override_max_frames is not None else int(default_max_frames)

    @staticmethod
    def _normalize_run_result(run_result):
        if isinstance(run_result, dict):
            return run_result
        return {
            "summary_iou": float(run_result),
            "total_video_frames": 0,
            "processed_loop_frames": 0,
            "found_ball_frames": 0,
            "label_frames": 0,
            "label_tracked_frames": 0,
            "label_not_tracked_frames": 0,
            "label_not_tracked_background_frames": 0,
            "label_not_tracked_detector_frames": 0,
            "label_detector_no_contours_frames": 0,
            "label_detector_filtered_area_frames": 0,
            "label_detector_filtered_solidity_frames": 0,
            "label_detector_filtered_circularity_frames": 0,
            "label_detector_filtered_perimeter_frames": 0,
            "label_detector_filtered_distance_frames": 0,
            "label_detector_other_frames": 0,
            "iou_evaluated_frames": 0,
            "iou_match_frames": 0,
            "iou_low_or_no_score_frames": 0,
            "grass_or_blue_frames": 0,
            "no_label_frames": 0,
            "no_label_tracked_ball_frames": 0,
            "no_label_not_tracked_frames": 0,
            "no_label_not_tracked_background_frames": 0,
            "no_label_not_tracked_detector_frames": 0,
            "no_label_detector_no_contours_frames": 0,
            "no_label_detector_filtered_area_frames": 0,
            "no_label_detector_filtered_solidity_frames": 0,
            "no_label_detector_filtered_circularity_frames": 0,
            "no_label_detector_filtered_perimeter_frames": 0,
            "no_label_detector_filtered_distance_frames": 0,
            "no_label_detector_other_frames": 0,
            "no_label_skipped_background_frames": 0,
            "stop_requested": False,
        }

    @staticmethod
    def _print_test_results(results, results_title, row_label, optimal_label, value_formatter=None):
        fmt = value_formatter or (lambda value: str(value))
        best_value = max(results, key=lambda key: results[key]["summary_iou"])

        print(f"\n{results_title}")
        for value, metrics in results.items():
            value_text = fmt(value)
            no_label_frames = metrics.get("no_label_frames", 0)
            no_label_tracked_frames = metrics.get("no_label_tracked_ball_frames", 0)
            no_label_not_tracked_frames = metrics.get(
                "no_label_not_tracked_frames",
                max(0, no_label_frames - no_label_tracked_frames),
            )
            no_label_not_tracked_background_frames = metrics.get(
                "no_label_not_tracked_background_frames",
                metrics.get("no_label_skipped_background_frames", 0),
            )
            no_label_not_tracked_detector_frames = metrics.get(
                "no_label_not_tracked_detector_frames",
                max(0, no_label_not_tracked_frames - no_label_not_tracked_background_frames),
            )
            no_label_detector_no_contours_frames = metrics.get("no_label_detector_no_contours_frames", 0)
            no_label_detector_filtered_area_frames = metrics.get("no_label_detector_filtered_area_frames", 0)
            no_label_detector_filtered_solidity_frames = metrics.get("no_label_detector_filtered_solidity_frames", 0)
            no_label_detector_filtered_circularity_frames = metrics.get("no_label_detector_filtered_circularity_frames", 0)
            no_label_detector_filtered_perimeter_frames = metrics.get("no_label_detector_filtered_perimeter_frames", 0)
            no_label_detector_filtered_distance_frames = metrics.get("no_label_detector_filtered_distance_frames", 0)
            no_label_detector_other_frames = metrics.get(
                "no_label_detector_other_frames",
                max(
                    0,
                    no_label_not_tracked_detector_frames
                    - (
                        no_label_detector_no_contours_frames
                        + no_label_detector_filtered_area_frames
                        + no_label_detector_filtered_solidity_frames
                        + no_label_detector_filtered_circularity_frames
                        + no_label_detector_filtered_perimeter_frames
                        + no_label_detector_filtered_distance_frames
                    ),
                ),
            )

            label_frames = metrics.get(
                "label_frames",
                max(0, metrics.get("processed_loop_frames", 0) - no_label_frames),
            )
            label_tracked_frames = metrics.get("label_tracked_frames", metrics.get("iou_evaluated_frames", 0))
            label_not_tracked_frames = metrics.get(
                "label_not_tracked_frames",
                max(0, label_frames - label_tracked_frames),
            )
            label_not_tracked_background_frames = metrics.get("label_not_tracked_background_frames", 0)
            label_not_tracked_detector_frames = metrics.get(
                "label_not_tracked_detector_frames",
                max(0, label_not_tracked_frames - label_not_tracked_background_frames),
            )
            label_detector_no_contours_frames = metrics.get("label_detector_no_contours_frames", 0)
            label_detector_filtered_area_frames = metrics.get("label_detector_filtered_area_frames", 0)
            label_detector_filtered_solidity_frames = metrics.get("label_detector_filtered_solidity_frames", 0)
            label_detector_filtered_circularity_frames = metrics.get("label_detector_filtered_circularity_frames", 0)
            label_detector_filtered_perimeter_frames = metrics.get("label_detector_filtered_perimeter_frames", 0)
            label_detector_filtered_distance_frames = metrics.get("label_detector_filtered_distance_frames", 0)
            label_detector_other_frames = metrics.get(
                "label_detector_other_frames",
                max(
                    0,
                    label_not_tracked_detector_frames
                    - (
                        label_detector_no_contours_frames
                        + label_detector_filtered_area_frames
                        + label_detector_filtered_solidity_frames
                        + label_detector_filtered_circularity_frames
                        + label_detector_filtered_perimeter_frames
                        + label_detector_filtered_distance_frames
                    ),
                ),
            )

            iou_high_frames = metrics.get("iou_match_frames", 0)
            iou_low_or_no_score_frames = metrics.get(
                "iou_low_or_no_score_frames",
                max(0, label_tracked_frames - iou_high_frames),
            )

            print(
                f"{row_label}={value_text}: normalized IoU={metrics['summary_iou']:.3f}, "
                f"total_frames={metrics['total_video_frames']}, "
                f"measured_found_ball={metrics['found_ball_frames']}, "
                f"grass_or_blue={metrics['grass_or_blue_frames']}"
            )
            print(f"  1a no_label={no_label_frames}")
            print(f"    1a1 tracked={no_label_tracked_frames}")
            print(f"    1a2 not_tracked={no_label_not_tracked_frames}")
            print(
                f"      reasons: background_classification={no_label_not_tracked_background_frames}, "
                f"detector_no_ball_found={no_label_not_tracked_detector_frames}"
            )
            print(
                f"        detector subreasons: no_contours={no_label_detector_no_contours_frames}, "
                f"filtered_area={no_label_detector_filtered_area_frames}, "
                f"filtered_solidity={no_label_detector_filtered_solidity_frames}, "
                f"filtered_circularity={no_label_detector_filtered_circularity_frames}, "
                f"filtered_perimeter={no_label_detector_filtered_perimeter_frames}, "
                f"filtered_distance={no_label_detector_filtered_distance_frames}, "
                f"other={no_label_detector_other_frames}"
            )
            print(f"  1b label={label_frames}")
            print(f"    1b1 tracked={label_tracked_frames}")
            print(f"      1b1a IoU>0.5={iou_high_frames}")
            print(f"      1b1b low_or_no_score={iou_low_or_no_score_frames}")
            print(f"    1b2 not_tracked={label_not_tracked_frames}")
            print(
                f"      reasons: background_classification={label_not_tracked_background_frames}, "
                f"detector_no_ball_found={label_not_tracked_detector_frames}"
            )
            print(
                f"        detector subreasons: no_contours={label_detector_no_contours_frames}, "
                f"filtered_area={label_detector_filtered_area_frames}, "
                f"filtered_solidity={label_detector_filtered_solidity_frames}, "
                f"filtered_circularity={label_detector_filtered_circularity_frames}, "
                f"filtered_perimeter={label_detector_filtered_perimeter_frames}, "
                f"filtered_distance={label_detector_filtered_distance_frames}, "
                f"other={label_detector_other_frames}"
            )

        best = results[best_value]
        best_text = fmt(best_value)
        print(
            f"\nOptimal {optimal_label}: {best_text} "
            f"(normalized IoU={best['summary_iou']:.3f}, total_frames={best['total_video_frames']}, "
            f"measured_found_ball={best['found_ball_frames']}, grass_or_blue={best['grass_or_blue_frames']})"
        )

    def test_size_min(self, video_path, video_processor_class, config_module, max_frames=None):
        import numpy as np
        original_state = self._capture_test_parameter_state(config_module)
        self._apply_loose_test_baseline(config_module, excluded_keys={"Size min"})
        self._print_loose_test_baseline(excluded_keys={"Size min"})
        try:
            results = {}
            run_max_frames = self._resolve_max_frames(max_frames, 4000)
            # 6 values between 30 and 200 (inclusive)
            for size_min in np.linspace(30, 200, 6, dtype=int):
                config_module.TRACKBAR_SETTINGS["Size min"] = (size_min, config_module.TRACKBAR_SETTINGS["Size min"][1])
                config_module.WAIT_KEY_DELAY_MS = 0
                print(f"Testing Size min={size_min}")
                processor = video_processor_class(video_path, max_frames=run_max_frames)
                run_result = self._normalize_run_result(processor.run())
                self._append_progress_csv_row(
                    config_module,
                    run_result,
                    test_name="size_min",
                    swept_param="Size min",
                    swept_value=int(size_min),
                    run_max_frames=run_max_frames,
                )
                if run_result.get("stop_requested", False) or getattr(processor, "stop_requested", False):
                    print("\nTest stopped by user (q). Aborting remaining Size min runs.")
                    break
                results[size_min] = run_result
            if not results:
                print("No completed Size min runs.")
                return
            self._print_test_results(
                results,
                results_title="Size min test results:",
                row_label="Size min",
                optimal_label="Size min",
            )
        finally:
            self._restore_test_parameter_state(config_module, original_state)

    def test_size_max(self, video_path, video_processor_class, config_module, max_frames=None):
        original_state = self._capture_test_parameter_state(config_module)
        self._apply_loose_test_baseline(config_module, excluded_keys={"Size Max"})
        self._print_loose_test_baseline(excluded_keys={"Size Max"})
        try:
            results = {}
            run_max_frames = self._resolve_max_frames(max_frames, 4000)
            size_max_values = list(range(80000, 35999, -7000))
            if size_max_values[-1] != 36000:
                size_max_values.append(36000)

            for size_max in size_max_values:
                current_max_limit = config_module.TRACKBAR_SETTINGS["Size Max"][1]
                config_module.TRACKBAR_SETTINGS["Size Max"] = (size_max, max(current_max_limit, size_max))
                config_module.WAIT_KEY_DELAY_MS = 0
                print(f"Testing Size max={size_max}")
                processor = video_processor_class(video_path, max_frames=run_max_frames)
                run_result = self._normalize_run_result(processor.run())
                self._append_progress_csv_row(
                    config_module,
                    run_result,
                    test_name="size_max",
                    swept_param="Size Max",
                    swept_value=int(size_max),
                    run_max_frames=run_max_frames,
                )
                if run_result.get("stop_requested", False) or getattr(processor, "stop_requested", False):
                    print("\nTest stopped by user (q). Aborting remaining Size max runs.")
                    break
                results[size_max] = run_result

            if not results:
                print("No completed Size max runs.")
                return
            self._print_test_results(
                results,
                results_title="Size max test results:",
                row_label="Size max",
                optimal_label="Size max",
            )
        finally:
            self._restore_test_parameter_state(config_module, original_state)

    def test_solidity_min(self, video_path, video_processor_class, config_module, max_frames=None):
        original_state = self._capture_test_parameter_state(config_module)
        self._apply_loose_test_baseline(config_module, excluded_keys={"SOLIDITY_MIN"})
        self._print_loose_test_baseline(excluded_keys={"SOLIDITY_MIN"})
        try:
            results = {}
            run_max_frames = self._resolve_max_frames(max_frames, 4000)
            solidity_values = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
            for solidity in solidity_values:
                config_module.SOLIDITY_MIN = solidity
                config_module.WAIT_KEY_DELAY_MS = 0
                print(f"Testing solidity_min={solidity:.2f}")
                processor = video_processor_class(video_path, max_frames=run_max_frames)
                run_result = self._normalize_run_result(processor.run())
                self._append_progress_csv_row(
                    config_module,
                    run_result,
                    test_name="solidity_min",
                    swept_param="SOLIDITY_MIN",
                    swept_value=float(solidity),
                    run_max_frames=run_max_frames,
                )
                if run_result.get("stop_requested", False) or getattr(processor, "stop_requested", False):
                    print("\nTest stopped by user (q). Aborting remaining solidity runs.")
                    break
                results[solidity] = run_result

            if not results:
                print("No completed solidity runs.")
                return
            self._print_test_results(
                results,
                results_title="Solidity test results:",
                row_label="solidity_min",
                optimal_label="solidity_min",
                value_formatter=lambda value: f"{value:.2f}",
            )
        finally:
            self._restore_test_parameter_state(config_module, original_state)

    def test_circularity(self, video_path, video_processor_class, config_module, max_frames=None):
        import numpy as np
        original_state = self._capture_test_parameter_state(config_module)
        self._apply_loose_test_baseline(config_module, excluded_keys={"circularity"})
        self._print_loose_test_baseline(excluded_keys={"circularity"})
        try:
            results = {}
            run_max_frames = self._resolve_max_frames(max_frames, 4000)
            # Test values from 0.65 to 0.95 (step 0.05)
            for circ in np.arange(0.65, 0.96, 0.05):
                # Set circularity in TRACKBAR_SETTINGS (assume percent)
                config_module.TRACKBAR_SETTINGS["circularity"] = (int(circ * 100), 100)
                config_module.WAIT_KEY_DELAY_MS = 0
                print(f"Testing circularity={circ:.2f}")
                processor = video_processor_class(video_path, max_frames=run_max_frames)
                run_result = self._normalize_run_result(processor.run())
                self._append_progress_csv_row(
                    config_module,
                    run_result,
                    test_name="circularity",
                    swept_param="circularity",
                    swept_value=float(circ),
                    run_max_frames=run_max_frames,
                )
                if run_result.get("stop_requested", False) or getattr(processor, "stop_requested", False):
                    print("\nTest stopped by user (q). Aborting remaining circularity runs.")
                    break
                results[circ] = run_result
            if not results:
                print("No completed circularity runs.")
                return
            self._print_test_results(
                results,
                results_title="Circularity test results:",
                row_label="circularity",
                optimal_label="circularity",
                value_formatter=lambda value: f"{value:.2f}",
            )
        finally:
            self._restore_test_parameter_state(config_module, original_state)

    def test_max_jump_pixels(self, video_path, video_processor_class, config_module, max_frames=None):
        original_state = self._capture_test_parameter_state(config_module)
        self._apply_loose_test_baseline(config_module, excluded_keys={"MAX_JUMP_PIXELS"})
        self._print_loose_test_baseline(excluded_keys={"MAX_JUMP_PIXELS"})
        try:
            results = {}
            run_max_frames = self._resolve_max_frames(max_frames, 1000)
            jump_values = [75, 150, 300, 600, 1000]

            for jump_pixels in jump_values:
                config_module.MAX_JUMP_PIXELS = int(jump_pixels)
                config_module.WAIT_KEY_DELAY_MS = 0
                print(f"Testing max_jump_pixels={jump_pixels}")
                processor = video_processor_class(video_path, max_frames=run_max_frames)
                run_result = self._normalize_run_result(processor.run())
                self._append_progress_csv_row(
                    config_module,
                    run_result,
                    test_name="max_jump_pixels",
                    swept_param="MAX_JUMP_PIXELS",
                    swept_value=int(jump_pixels),
                    run_max_frames=run_max_frames,
                )
                if run_result.get("stop_requested", False) or getattr(processor, "stop_requested", False):
                    print("\nTest stopped by user (q). Aborting remaining MAX_JUMP_PIXELS runs.")
                    break
                results[jump_pixels] = run_result

            if not results:
                print("No completed MAX_JUMP_PIXELS runs.")
                return
            self._print_test_results(
                results,
                results_title="MAX_JUMP_PIXELS test results:",
                row_label="max_jump_pixels",
                optimal_label="max_jump_pixels",
            )
        finally:
            self._restore_test_parameter_state(config_module, original_state)

    def test_weight_y_boost(self, video_path, video_processor_class, config_module, max_frames=None):
        original_state = self._capture_test_parameter_state(config_module)
        self._apply_loose_test_baseline(config_module, excluded_keys={"WEIGHT_Y_BOOST"})
        self._print_loose_test_baseline(excluded_keys={"WEIGHT_Y_BOOST"})
        try:
            results = {}
            run_max_frames = self._resolve_max_frames(max_frames, 1000)
            y_boost_values = list(range(0, 41, 10))

            for y_boost in y_boost_values:
                config_module.WEIGHT_Y_BOOST = float(y_boost)
                config_module.WAIT_KEY_DELAY_MS = 0
                print(f"Testing weight_y_boost={y_boost}")
                processor = video_processor_class(video_path, max_frames=run_max_frames)
                run_result = self._normalize_run_result(processor.run())
                self._append_progress_csv_row(
                    config_module,
                    run_result,
                    test_name="weight_y_boost",
                    swept_param="WEIGHT_Y_BOOST",
                    swept_value=float(y_boost),
                    run_max_frames=run_max_frames,
                )
                if run_result.get("stop_requested", False) or getattr(processor, "stop_requested", False):
                    print("\nTest stopped by user (q). Aborting remaining WEIGHT_Y_BOOST runs.")
                    break
                results[y_boost] = run_result

            if not results:
                print("No completed WEIGHT_Y_BOOST runs.")
                return
            self._print_test_results(
                results,
                results_title="WEIGHT_Y_BOOST test results:",
                row_label="weight_y_boost",
                optimal_label="weight_y_boost",
            )
        finally:
            self._restore_test_parameter_state(config_module, original_state)

    def test_suite(self, video_path, video_processor_class, config_module, tests=None, common_params=None, max_frames=None):
        available_tests = {
            "size_min": self.test_size_min,
            "size_max": self.test_size_max,
            "solidity_min": self.test_solidity_min,
            "circularity": self.test_circularity,
            "max_jump_pixels": self.test_max_jump_pixels,
            "weight_y_boost": self.test_weight_y_boost,
        }
        excluded_common_keys = {
            "size_min": {"Size min"},
            "size_max": {"Size Max"},
            "solidity_min": {"SOLIDITY_MIN"},
            "circularity": {"circularity"},
            "max_jump_pixels": {"MAX_JUMP_PIXELS"},
            "weight_y_boost": {"WEIGHT_Y_BOOST"},
        }

        run_order = tests or ["size_min", "size_max", "solidity_min", "circularity", "max_jump_pixels", "weight_y_boost"]
        invalid_tests = [name for name in run_order if name not in available_tests]
        if invalid_tests:
            valid_names = ", ".join(available_tests.keys())
            invalid_names = ", ".join(invalid_tests)
            raise ValueError(f"Unknown test name(s): {invalid_names}. Valid names: {valid_names}")

        print(f"Running test suite: {', '.join(run_order)}")
        for test_name in run_order:
            print(f"\n--- Running {test_name} ---")
            self._apply_common_parameters(
                config_module,
                common_params,
                excluded_keys=excluded_common_keys.get(test_name, set()),
            )
            available_tests[test_name](video_path, video_processor_class, config_module, max_frames=max_frames)

        print("\nTest suite completed.")

    def __init__(self):
        self.ious = []

    def add_iou(self, iou):
        self.ious.append(iou)

    def summary_score(self):
        if not self.ious:
            return 0.0
        total_iou = sum(self.ious)
        max_score = len(self.ious)  # max IoU is 1 per frame
        return total_iou / max_score

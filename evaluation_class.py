class EvaluationClass:

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
            "iou_evaluated_frames": 0,
            "iou_match_frames": 0,
            "grass_or_blue_frames": 0,
            "stop_requested": False,
        }

    def test_size_min(self, video_path, video_processor_class, config_module, max_frames=None):
        import numpy as np
        results = {}
        run_max_frames = self._resolve_max_frames(max_frames, 4000)
        # 6 values between 30 and 200 (inclusive)
        for size_min in np.linspace(30, 200, 6, dtype=int):
            config_module.TRACKBAR_SETTINGS["Size min"] = (size_min, config_module.TRACKBAR_SETTINGS["Size min"][1])
            config_module.WAIT_KEY_DELAY_MS = 0
            print(f"Testing Size min={size_min}")
            processor = video_processor_class(video_path, max_frames=run_max_frames)
            run_result = self._normalize_run_result(processor.run())
            if run_result.get("stop_requested", False) or getattr(processor, "stop_requested", False):
                print("\nTest stopped by user (q). Aborting remaining Size min runs.")
                break
            results[size_min] = run_result
        if not results:
            print("No completed Size min runs.")
            return
        best_size_min = max(results, key=lambda k: results[k]["summary_iou"])
        print("\nSize min test results:")
        for size_min, metrics in results.items():
            print(
                f"Size min={size_min}: normalized IoU={metrics['summary_iou']:.3f}, "
                f"total_frames={metrics['total_video_frames']}, "
                f"measured_found_ball={metrics['found_ball_frames']}, "
                f"iou_match_over_0.5={metrics['iou_match_frames']}, "
                f"grass_or_blue={metrics['grass_or_blue_frames']}"
            )
        best = results[best_size_min]
        print(
            f"\nOptimal Size min: {best_size_min} "
            f"(normalized IoU={best['summary_iou']:.3f}, total_frames={best['total_video_frames']}, "
            f"measured_found_ball={best['found_ball_frames']}, iou_match_over_0.5={best['iou_match_frames']}, "
            f"grass_or_blue={best['grass_or_blue_frames']})"
        )

    def test_size_max(self, video_path, video_processor_class, config_module, max_frames=None):
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
            if run_result.get("stop_requested", False) or getattr(processor, "stop_requested", False):
                print("\nTest stopped by user (q). Aborting remaining Size max runs.")
                break
            results[size_max] = run_result

        if not results:
            print("No completed Size max runs.")
            return

        best_size_max = max(results, key=lambda k: results[k]["summary_iou"])
        print("\nSize max test results:")
        for size_max, metrics in results.items():
            print(
                f"Size max={size_max}: normalized IoU={metrics['summary_iou']:.3f}, "
                f"total_frames={metrics['total_video_frames']}, "
                f"measured_found_ball={metrics['found_ball_frames']}, "
                f"iou_match_over_0.5={metrics['iou_match_frames']}, "
                f"grass_or_blue={metrics['grass_or_blue_frames']}"
            )
        best = results[best_size_max]
        print(
            f"\nOptimal Size max: {best_size_max} "
            f"(normalized IoU={best['summary_iou']:.3f}, total_frames={best['total_video_frames']}, "
            f"measured_found_ball={best['found_ball_frames']}, iou_match_over_0.5={best['iou_match_frames']}, "
            f"grass_or_blue={best['grass_or_blue_frames']})"
        )

    def test_solidity_min(self, video_path, video_processor_class, config_module, max_frames=None):
        results = {}
        run_max_frames = self._resolve_max_frames(max_frames, 1000)
        solidity_values = [0.50, 1.00]
        for solidity in solidity_values:
            config_module.SOLIDITY_MIN = solidity
            config_module.WAIT_KEY_DELAY_MS = 0
            print(f"Testing solidity_min={solidity:.2f}")
            processor = video_processor_class(video_path, max_frames=run_max_frames)
            run_result = self._normalize_run_result(processor.run())
            if run_result.get("stop_requested", False) or getattr(processor, "stop_requested", False):
                print("\nTest stopped by user (q). Aborting remaining solidity runs.")
                break
            results[solidity] = run_result

        if not results:
            print("No completed solidity runs.")
            return

        best_solidity = max(results, key=lambda k: results[k]["summary_iou"])
        print("\nSolidity test results:")
        for solidity, metrics in results.items():
            print(
                f"solidity_min={solidity:.2f}: normalized IoU={metrics['summary_iou']:.3f}, "
                f"total_frames={metrics['total_video_frames']}, "
                f"measured_found_ball={metrics['found_ball_frames']}, "
                f"iou_match_over_0.5={metrics['iou_match_frames']}, "
                f"grass_or_blue={metrics['grass_or_blue_frames']}"
            )
        best = results[best_solidity]
        print(
            f"\nOptimal solidity_min: {best_solidity:.2f} "
            f"(normalized IoU={best['summary_iou']:.3f}, total_frames={best['total_video_frames']}, "
            f"measured_found_ball={best['found_ball_frames']}, iou_match_over_0.5={best['iou_match_frames']}, "
            f"grass_or_blue={best['grass_or_blue_frames']})"
        )

    def test_circularity(self, video_path, video_processor_class, config_module, max_frames=None):
        import numpy as np
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
            if run_result.get("stop_requested", False) or getattr(processor, "stop_requested", False):
                print("\nTest stopped by user (q). Aborting remaining circularity runs.")
                break
            results[circ] = run_result
        if not results:
            print("No completed circularity runs.")
            return
        # Find best circularity
        best_circ = max(results, key=lambda k: results[k]["summary_iou"])
        print("\nCircularity test results:")
        for circ, metrics in results.items():
            print(
                f"circularity={circ:.2f}: normalized IoU={metrics['summary_iou']:.3f}, "
                f"total_frames={metrics['total_video_frames']}, "
                f"measured_found_ball={metrics['found_ball_frames']}, "
                f"iou_match_over_0.5={metrics['iou_match_frames']}, "
                f"grass_or_blue={metrics['grass_or_blue_frames']}"
            )
        best = results[best_circ]
        print(
            f"\nOptimal circularity: {best_circ:.2f} "
            f"(normalized IoU={best['summary_iou']:.3f}, total_frames={best['total_video_frames']}, "
            f"measured_found_ball={best['found_ball_frames']}, iou_match_over_0.5={best['iou_match_frames']}, "
            f"grass_or_blue={best['grass_or_blue_frames']})"
        )

    def test_max_jump_pixels(self, video_path, video_processor_class, config_module, max_frames=None):
        results = {}
        run_max_frames = self._resolve_max_frames(max_frames, 1000)
        jump_values = [75, 150, 300, 600, 1000]

        for jump_pixels in jump_values:
            config_module.MAX_JUMP_PIXELS = int(jump_pixels)
            config_module.WAIT_KEY_DELAY_MS = 0
            print(f"Testing max_jump_pixels={jump_pixels}")
            processor = video_processor_class(video_path, max_frames=run_max_frames)
            run_result = self._normalize_run_result(processor.run())
            if run_result.get("stop_requested", False) or getattr(processor, "stop_requested", False):
                print("\nTest stopped by user (q). Aborting remaining MAX_JUMP_PIXELS runs.")
                break
            results[jump_pixels] = run_result

        if not results:
            print("No completed MAX_JUMP_PIXELS runs.")
            return

        best_jump_pixels = max(results, key=lambda k: results[k]["summary_iou"])
        print("\nMAX_JUMP_PIXELS test results:")
        for jump_pixels, metrics in results.items():
            print(
                f"max_jump_pixels={jump_pixels}: normalized IoU={metrics['summary_iou']:.3f}, "
                f"total_frames={metrics['total_video_frames']}, "
                f"measured_found_ball={metrics['found_ball_frames']}, "
                f"iou_match_over_0.5={metrics['iou_match_frames']}, "
                f"grass_or_blue={metrics['grass_or_blue_frames']}"
            )
        best = results[best_jump_pixels]
        print(
            f"\nOptimal max_jump_pixels: {best_jump_pixels} "
            f"(normalized IoU={best['summary_iou']:.3f}, total_frames={best['total_video_frames']}, "
            f"measured_found_ball={best['found_ball_frames']}, iou_match_over_0.5={best['iou_match_frames']}, "
            f"grass_or_blue={best['grass_or_blue_frames']})"
        )

    def test_weight_y_boost(self, video_path, video_processor_class, config_module, max_frames=None):
        results = {}
        run_max_frames = self._resolve_max_frames(max_frames, 1000)
        y_boost_values = list(range(0, 41, 10))

        for y_boost in y_boost_values:
            config_module.WEIGHT_Y_BOOST = float(y_boost)
            config_module.WAIT_KEY_DELAY_MS = 0
            print(f"Testing weight_y_boost={y_boost}")
            processor = video_processor_class(video_path, max_frames=run_max_frames)
            run_result = self._normalize_run_result(processor.run())
            if run_result.get("stop_requested", False) or getattr(processor, "stop_requested", False):
                print("\nTest stopped by user (q). Aborting remaining WEIGHT_Y_BOOST runs.")
                break
            results[y_boost] = run_result

        if not results:
            print("No completed WEIGHT_Y_BOOST runs.")
            return

        best_y_boost = max(results, key=lambda k: results[k]["summary_iou"])
        print("\nWEIGHT_Y_BOOST test results:")
        for y_boost, metrics in results.items():
            print(
                f"weight_y_boost={y_boost}: normalized IoU={metrics['summary_iou']:.3f}, "
                f"total_frames={metrics['total_video_frames']}, "
                f"measured_found_ball={metrics['found_ball_frames']}, "
                f"iou_match_over_0.5={metrics['iou_match_frames']}, "
                f"grass_or_blue={metrics['grass_or_blue_frames']}"
            )
        best = results[best_y_boost]
        print(
            f"\nOptimal weight_y_boost: {best_y_boost} "
            f"(normalized IoU={best['summary_iou']:.3f}, total_frames={best['total_video_frames']}, "
            f"measured_found_ball={best['found_ball_frames']}, iou_match_over_0.5={best['iou_match_frames']}, "
            f"grass_or_blue={best['grass_or_blue_frames']})"
        )

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

        run_order = tests or ["size_min", "size_max", "solidity_min", "circularity"]
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

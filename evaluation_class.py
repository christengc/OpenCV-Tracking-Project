class EvaluationClass:

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

    def test_size_min(self, video_path, video_processor_class, config_module):
        import numpy as np
        results = {}
        # 6 values between 30 and 200 (inclusive)
        for size_min in np.linspace(30, 200, 6, dtype=int):
            config_module.TRACKBAR_SETTINGS["Size min"] = (size_min, config_module.TRACKBAR_SETTINGS["Size min"][1])
            config_module.WAIT_KEY_DELAY_MS = 0
            print(f"Testing Size min={size_min}")
            processor = video_processor_class(video_path, max_frames=4000)
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

    def test_size_max(self, video_path, video_processor_class, config_module):
        results = {}
        size_max_values = list(range(80000, 35999, -7000))
        if size_max_values[-1] != 36000:
            size_max_values.append(36000)

        for size_max in size_max_values:
            current_max_limit = config_module.TRACKBAR_SETTINGS["Size Max"][1]
            config_module.TRACKBAR_SETTINGS["Size Max"] = (size_max, max(current_max_limit, size_max))
            config_module.WAIT_KEY_DELAY_MS = 0
            print(f"Testing Size max={size_max}")
            processor = video_processor_class(video_path, max_frames=4000)
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

    def test_solidity_min(self, video_path, video_processor_class, config_module):
        import numpy as np
        results = {}
        # Test values from 0.65 to 0.95 (step 0.3)
        for solidity in np.arange(0.65, 0.951, 0.3):
            solidity = round(float(solidity), 2)
            config_module.SOLIDITY_MIN = solidity
            config_module.WAIT_KEY_DELAY_MS = 0
            print(f"Testing solidity_min={solidity:.2f}")
            processor = video_processor_class(video_path, max_frames=1000)
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

    def test_circularity(self, video_path, video_processor_class, config_module):
        import numpy as np
        results = {}
        # Test values from 0.65 to 0.95 (step 0.05)
        for circ in np.arange(0.65, 0.96, 0.05):
            # Set circularity in TRACKBAR_SETTINGS (assume percent)
            config_module.TRACKBAR_SETTINGS["circularity"] = (int(circ * 100), 100)
            config_module.WAIT_KEY_DELAY_MS = 0
            print(f"Testing circularity={circ:.2f}")
            processor = video_processor_class(video_path, max_frames=4000)
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

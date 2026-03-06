import numpy as np
import config
from video_processor import VideoProcessor


def run_fine_sweep(video_path="video2.mp4"):
    results = {}
    for circ in np.arange(0.65, 0.751, 0.01):
        circ = round(float(circ), 2)
        config.TRACKBAR_SETTINGS["circularity"] = (int(round(circ * 100)), 100)
        config.WAIT_KEY_DELAY_MS = 0
        print(f"Testing circularity={circ:.2f}")

        processor = VideoProcessor(video_path, max_frames=1000)
        run_result = processor.run()

        if isinstance(run_result, dict):
            metrics = run_result
        else:
            metrics = {
                "summary_iou": float(run_result),
                "total_video_frames": 0,
                "found_ball_frames": 0,
                "iou_match_frames": 0,
                "grass_or_blue_frames": 0,
                "stop_requested": False,
            }

        if metrics.get("stop_requested", False) or getattr(processor, "stop_requested", False):
            print("Test stopped by user (q).")
            break

        results[circ] = metrics

    if not results:
        print("No completed runs.")
        return

    best_circ = max(results, key=lambda k: results[k]["summary_iou"])
    print("\nCircularity fine-sweep results:")
    for circ in sorted(results.keys()):
        m = results[circ]
        print(
            f"circularity={circ:.2f}: IoU={m['summary_iou']:.3f}, "
            f"total_frames={m.get('total_video_frames', 0)}, "
            f"found_ball={m.get('found_ball_frames', 0)}, "
            f"iou_match>0.5={m.get('iou_match_frames', 0)}, "
            f"grass_or_blue={m.get('grass_or_blue_frames', 0)}"
        )

    b = results[best_circ]
    print(f"\nBest circularity: {best_circ:.2f} (IoU={b['summary_iou']:.3f})")


if __name__ == "__main__":
    run_fine_sweep()

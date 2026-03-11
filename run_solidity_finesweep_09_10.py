import config
from video_processor import VideoProcessor


def run_sweep(video_path=None, max_frames=4000):
    video_path = video_path or config.DEFAULT_VIDEO_PATH
    config.WAIT_KEY_DELAY_MS = 0

    values = [round(0.90 + i * 0.02, 2) for i in range(6)]
    results = {}

    for solidity in values:
        config.SOLIDITY_MIN = solidity
        print(f"Testing solidity_min={solidity:.2f}")
        vp = VideoProcessor(video_path, max_frames=max_frames)
        metrics = vp.run()
        if not isinstance(metrics, dict):
            metrics = {"summary_iou": float(metrics)}
        results[solidity] = metrics

        if metrics.get("stop_requested", False) or getattr(vp, "stop_requested", False):
            print("Stopped by user.")
            break

    print("\nSolidity targeted sweep results:")
    for solidity in sorted(results):
        m = results[solidity]
        print(
            f"solidity_min={solidity:.2f}: IoU={m.get('summary_iou', 0.0):.3f}, "
            f"found_ball={m.get('found_ball_frames', 0)}, "
            f"iou_match>0.5={m.get('iou_match_frames', 0)}, "
            f"label={m.get('label_frames', 0)}"
        )

    if results:
        best = max(results, key=lambda k: results[k].get("summary_iou", -1))
        print(f"Best solidity_min: {best:.2f} (IoU={results[best].get('summary_iou', 0.0):.3f})")


if __name__ == "__main__":
    run_sweep()

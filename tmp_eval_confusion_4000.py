import builtins
import cv2
import config
from video_processor import VideoProcessor


# Headless run
cv2.imshow = lambda *args, **kwargs: None
cv2.waitKey = lambda delay=0: 255
cv2.destroyAllWindows = lambda: None
cv2.setMouseCallback = lambda *args, **kwargs: None

# Speed up loop delay
config.WAIT_KEY_DELAY_MS = 1

# Optional: silence noisy per-frame shift logging
_orig_print = builtins.print


def _filtered_print(*args, **kwargs):
    if args and isinstance(args[0], str) and args[0].startswith("[SHIFT SCORE]"):
        return
    _orig_print(*args, **kwargs)


builtins.print = _filtered_print

vp = VideoProcessor(config.DEFAULT_VIDEO_PATH, max_frames=4000)
metrics = vp.run()

builtins.print = _orig_print

# Confusion matrix, frame-level:
# Positive frame: valid non-occluded annotation exists.
# Predicted positive: tracker detection with IoU > 0.5.
tp = int(metrics.get("iou_match_frames", 0))
fn = int(metrics.get("label_not_tracked_frames", 0)) + int(metrics.get("iou_low_or_no_score_frames", 0))
fp = int(metrics.get("no_label_tracked_ball_frames", 0))
tn = int(metrics.get("no_label_not_tracked_frames", 0))

n = tp + fn + fp + tn
acc = (tp + tn) / n if n else 0.0
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

print("\n=== CONFUSION MATRIX (first 4000 frames) ===")
print(f"TN={tn}  FP={fp}")
print(f"FN={fn}  TP={tp}")
print(f"Total evaluated frames={n}")
print(f"Accuracy={acc:.4f}")
print(f"Precision={precision:.4f}")
print(f"Recall={recall:.4f}")
print(f"F1={f1:.4f}")

print("\n=== RAW RUN METRICS USED ===")
for key in [
    "processed_loop_frames",
    "total_video_frames",
    "label_frames",
    "label_tracked_frames",
    "label_not_tracked_frames",
    "iou_evaluated_frames",
    "iou_match_frames",
    "iou_low_or_no_score_frames",
    "no_label_frames",
    "no_label_tracked_ball_frames",
    "no_label_not_tracked_frames",
]:
    print(f"{key}={metrics.get(key)}")

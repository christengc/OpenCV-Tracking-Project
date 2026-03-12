import json
import os

import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    DEFAULT_START_FRAME,
    DEFAULT_VIDEO_PATH,
    SVM_C,
    SVM_CLASS_WEIGHT,
    SVM_ENABLE_CONFIDENCE_GATE,
    SVM_GAMMA,
    SVM_KERNEL,
    SVM_MAX_TRAIN_FRAMES,
    SVM_MIN_DECISION_MARGIN,
    SVM_NEGATIVE_PATCH_RATIO,
    SVM_PATCH_SIZE,
)
from video_processor import collect_annotation_and_random_patch_features_local

# Practical dense scan setup.
SCAN_STEP = 4  # set to 1 for fully exhaustive pixel-by-pixel scan
MAX_SCAN_FRAMES = 12


def load_coco_ann(path):
    with open(path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    coco_ann = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann.get("image_id")
        bbox = ann.get("bbox")
        attrs = ann.get("attributes") or {}
        occluded = bool(attrs.get("occluded", False))
        if img_id is not None and bbox:
            existing = coco_ann.get(img_id)
            if existing is None or (existing.get("occluded", True) and not occluded):
                coco_ann[img_id] = {"bbox": bbox, "occluded": occluded}
    return coco_ann


def preprocess_patch(patch_bgr, patch_size):
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape != (patch_size, patch_size):
        gray = cv2.resize(gray, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.reshape(-1)


def train_svm(coco_ann):
    cap = cv2.VideoCapture(DEFAULT_VIDEO_PATH)
    X_cols, y = collect_annotation_and_random_patch_features_local(
        cap,
        coco_ann,
        start_frame=DEFAULT_START_FRAME,
        max_frames=SVM_MAX_TRAIN_FRAMES,
        patch_size=SVM_PATCH_SIZE,
        negative_ratio=SVM_NEGATIVE_PATCH_RATIO,
        random_seed=0,
    )
    cap.release()

    if X_cols.shape[1] == 0 or y.size == 0:
        raise RuntimeError("No training samples collected for SVM")

    X = X_cols.T
    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=SVM_KERNEL,
            C=float(SVM_C),
            gamma=SVM_GAMMA,
            class_weight=SVM_CLASS_WEIGHT,
        ),
    )
    model.fit(X, y)
    return model


def count_hits_in_frame(model, frame, patch_size, scan_step, use_gate, min_margin):
    h, w = frame.shape[:2]
    half = patch_size // 2
    padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_REFLECT_101)

    xs = list(range(0, w, scan_step))
    ys = list(range(0, h, scan_step))

    hits = 0
    total = 0

    for y in ys:
        row_features = []
        for x in xs:
            patch = padded[y:y + patch_size, x:x + patch_size]
            row_features.append(preprocess_patch(patch, patch_size))

        if not row_features:
            continue

        X_row = np.asarray(row_features)
        preds = model.predict(X_row)

        if use_gate and hasattr(model, "decision_function"):
            margins = np.abs(np.asarray(model.decision_function(X_row)).reshape(-1))
            row_hits = np.sum((preds == 1) & (margins >= float(min_margin)))
        else:
            row_hits = np.sum(preds == 1)

        hits += int(row_hits)
        total += len(xs)

    return hits, total


def main():
    cwd = os.getcwd()
    coco_path = os.path.join(cwd, "coco splitandoutline.json")
    if not os.path.exists(coco_path):
        raise FileNotFoundError(f"Missing file: {coco_path}")

    coco_ann = load_coco_ann(coco_path)
    model = train_svm(coco_ann)

    cap = cv2.VideoCapture(DEFAULT_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {DEFAULT_VIDEO_PATH}")

    start_zero_based = max(0, int(DEFAULT_START_FRAME) - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_zero_based)

    print(
        f"Dense SVM hit scan: frames={MAX_SCAN_FRAMES}, step={SCAN_STEP}, "
        f"gate={bool(SVM_ENABLE_CONFIDENCE_GATE)}, min_margin={float(SVM_MIN_DECISION_MARGIN):.2f}"
    )

    frame_results = []
    for idx in range(MAX_SCAN_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break

        frame_no = start_zero_based + idx + 1
        hits, total = count_hits_in_frame(
            model,
            frame,
            patch_size=SVM_PATCH_SIZE,
            scan_step=SCAN_STEP,
            use_gate=bool(SVM_ENABLE_CONFIDENCE_GATE),
            min_margin=float(SVM_MIN_DECISION_MARGIN),
        )
        hit_rate = (hits / total) if total else 0.0
        frame_results.append((frame_no, hits, total, hit_rate))
        print(f"frame={frame_no} hits={hits} total={total} hit_rate={hit_rate:.6f}")

    cap.release()

    if frame_results:
        avg_hits = float(np.mean([r[1] for r in frame_results]))
        avg_rate = float(np.mean([r[3] for r in frame_results]))
        print(f"SUMMARY frames_scanned={len(frame_results)} avg_hits={avg_hits:.2f} avg_hit_rate={avg_rate:.6f}")


if __name__ == "__main__":
    main()

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
)
from video_processor import collect_annotation_and_random_patch_features_local

SCAN_STEP = 4
MAX_SCAN_FRAMES = 8
PATCH_SIZES = [20, 30]


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


def train_svm(coco_ann, patch_size):
    cap = cv2.VideoCapture(DEFAULT_VIDEO_PATH)
    X_cols, y = collect_annotation_and_random_patch_features_local(
        cap,
        coco_ann,
        start_frame=DEFAULT_START_FRAME,
        max_frames=SVM_MAX_TRAIN_FRAMES,
        patch_size=patch_size,
        negative_ratio=SVM_NEGATIVE_PATCH_RATIO,
        random_seed=0,
    )
    cap.release()

    if X_cols.shape[1] == 0 or y.size == 0:
        raise RuntimeError(f"No training samples for patch_size={patch_size}")

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


def count_hits_in_frame(model, frame, patch_size):
    h, w = frame.shape[:2]
    half = patch_size // 2
    padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_REFLECT_101)

    xs = list(range(0, w, SCAN_STEP))
    ys = list(range(0, h, SCAN_STEP))

    hits = 0
    total = 0

    for y in ys:
        row_features = []
        for x in xs:
            patch = padded[y:y + patch_size, x:x + patch_size]
            row_features.append(preprocess_patch(patch, patch_size))

        X_row = np.asarray(row_features)
        preds = model.predict(X_row)

        if bool(SVM_ENABLE_CONFIDENCE_GATE) and hasattr(model, "decision_function"):
            margins = np.abs(np.asarray(model.decision_function(X_row)).reshape(-1))
            row_hits = np.sum((preds == 1) & (margins >= float(SVM_MIN_DECISION_MARGIN)))
        else:
            row_hits = np.sum(preds == 1)

        hits += int(row_hits)
        total += len(xs)

    return hits, total


def main():
    coco_path = os.path.join(os.getcwd(), "coco splitandoutline.json")
    coco_ann = load_coco_ann(coco_path)

    for patch_size in PATCH_SIZES:
        model = train_svm(coco_ann, patch_size)

        cap = cv2.VideoCapture(DEFAULT_VIDEO_PATH)
        start_zero_based = max(0, int(DEFAULT_START_FRAME) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_zero_based)

        print(f"\n=== patch_size={patch_size} ===")
        frame_stats = []
        for idx in range(MAX_SCAN_FRAMES):
            ok, frame = cap.read()
            if not ok:
                break
            frame_no = start_zero_based + idx + 1
            hits, total = count_hits_in_frame(model, frame, patch_size)
            rate = (hits / total) if total else 0.0
            frame_stats.append((frame_no, hits, total, rate))
            print(f"frame={frame_no} hits={hits} total={total} hit_rate={rate:.6f}")

        cap.release()

        if frame_stats:
            avg_hits = float(np.mean([s[1] for s in frame_stats]))
            avg_rate = float(np.mean([s[3] for s in frame_stats]))
            print(f"SUMMARY patch_size={patch_size} frames={len(frame_stats)} avg_hits={avg_hits:.2f} avg_hit_rate={avg_rate:.6f}")


if __name__ == "__main__":
    main()

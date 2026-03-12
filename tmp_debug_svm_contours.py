import json
import os

import cv2
import numpy as np

import config
from ball_detector import createBinaryMask, _classify_contour_patch_ml, _calcContourMetrics
from utils import detect_players
from video_processor import VideoProcessor


def load_coco_ann():
    coco_path = os.path.join(os.getcwd(), "coco splitandoutline.json")
    with open(coco_path, "r", encoding="utf-8") as f:
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


processor = VideoProcessor(config.DEFAULT_VIDEO_PATH)
processor.vid.release()
processor.vid = cv2.VideoCapture(config.DEFAULT_VIDEO_PATH)
processor.initialize_svm_model(load_coco_ann(), max_frames=config.SVM_MAX_TRAIN_FRAMES, patch_size=config.SVM_PATCH_SIZE)
print("model_loaded", processor.patch_classifier is not None, "patch_size", processor.patch_classifier_patch_size)

cap = cv2.VideoCapture(config.DEFAULT_VIDEO_PATH)
for frame_idx in range(1, 6):
    ok, frame = cap.read()
    if not ok:
        break
    rects = detect_players(frame)
    contours = rects['contours'] if isinstance(rects, dict) and 'contours' in rects else rects
    player_rect = max(contours, key=cv2.contourArea) if contours else None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    player_exclusion_mask = None
    if player_rect is not None:
        from utils import create_player_exclusion_mask
        player_exclusion_mask = create_player_exclusion_mask(frame.shape, player_rect, margin=config.PLAYER_MASK_MARGIN)
    mask = createBinaryMask(frame, hsv, False, True, player_exclusion_mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_pos = 0
    gated_pos = 0
    for cnt in cnts:
        m = _calcContourMetrics(cnt)
        if m['aspect_ratio'] > 3:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        is_ball, margin = _classify_contour_patch_ml(
            frame,
            x,
            y,
            processor.patch_classifier,
            patch_size=processor.patch_classifier_patch_size,
            classifier_name='svm',
            use_confidence_gate=bool(processor.patch_classifier_use_confidence_gate),
            min_margin=float(processor.patch_classifier_min_margin),
        )
        if is_ball:
            raw_pos += 1
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
            if circularity >= float(config.ML_MIN_CIRCULARITY_FOR_POSITIVE):
                gated_pos += 1
    print(f"frame={frame_idx} contours={len(cnts)} raw_svm={raw_pos} gated_svm={gated_pos}")
cap.release()

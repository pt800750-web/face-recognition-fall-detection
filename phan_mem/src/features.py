from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .label_mapping import ID_TO_LABEL, LABEL_TO_ID
from .utils import PROJECT_ROOT, resolve_path, select_device


FEATURE_NAMES: List[str] = [
    "center_x",
    "center_y",
    "width",
    "height",
    "aspect_width_height",
    "area",
    "bottom_y",
    "confidence",
    "keypoint_visible_ratio",
    "torso_horizontal_score",
    "hip_y",
    "shoulder_y",
    "ankle_y",
    "knee_y",
    "body_horizontal_score",
    "keypoint_height",
    "keypoint_width",
    "delta_center_y",
    "delta_center_x",
    "delta_height",
    "delta_width",
    "delta_area",
    "delta_aspect",
    "speed",
    "downward_speed",
    "motion_mean",
    "missing_flag",
    "shoulder_hip_horizontal_score",
    "leg_fold_score",
    "head_y",
    "top_y",
    "left_x",
    "right_x",
    "keypoint_confidence_mean",
    "box_diagonal",
    "area_growth_rate",
    "height_drop_rate",
]
FEATURE_DIM = len(FEATURE_NAMES)


COCO_KEYPOINTS: Dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


@dataclass
class PersonObservation:
    bbox_xyxy: np.ndarray
    confidence: float
    keypoints_xy: Optional[np.ndarray]
    keypoints_conf: Optional[np.ndarray]
    frame_shape: Tuple[int, int]
    timestamp: float
    class_id: int = 0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy.astype(float)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection_over_min(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    min_area = min(area_a, area_b)
    if min_area <= 0.0:
        return 0.0
    return float(intersection / min_area)


def _bbox_area_ratio(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    max_area = max(area_a, area_b)
    if max_area <= 0.0:
        return 0.0
    return float(min(area_a, area_b) / max_area)


def suppress_duplicate_person_observations(observations: Sequence[PersonObservation]) -> List[PersonObservation]:
    kept: List[PersonObservation] = []
    for observation in observations:
        duplicate = False
        for existing in kept:
            overlap_ratio = _bbox_intersection_over_min(observation.bbox_xyxy, existing.bbox_xyxy)
            area_ratio = _bbox_area_ratio(observation.bbox_xyxy, existing.bbox_xyxy)
            if overlap_ratio >= 0.60 and area_ratio <= 0.70:
                duplicate = True
                break
        if not duplicate:
            kept.append(observation)
    return kept


def normalize_bbox(bbox_xyxy: Sequence[float], frame_shape: Tuple[int, int]) -> Tuple[float, float, float, float, float, float, float, float]:
    frame_h, frame_w = frame_shape
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"Invalid frame shape: {frame_shape}")
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = float(np.clip(x1 / frame_w, 0.0, 1.0))
    x2 = float(np.clip(x2 / frame_w, 0.0, 1.0))
    y1 = float(np.clip(y1 / frame_h, 0.0, 1.0))
    y2 = float(np.clip(y2 / frame_h, 0.0, 1.0))
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    center_x = x1 + width * 0.5
    center_y = y1 + height * 0.5
    aspect = width / height
    area = width * height
    return center_x, center_y, width, height, aspect, area, y2, y1


def _visible_points(
    keypoints_xy: Optional[np.ndarray],
    keypoints_conf: Optional[np.ndarray],
    frame_shape: Tuple[int, int],
    min_confidence: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    if keypoints_xy is None or keypoints_conf is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    points = np.asarray(keypoints_xy, dtype=np.float32).copy()
    conf = np.asarray(keypoints_conf, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] != conf.shape[0]:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    frame_h, frame_w = frame_shape
    if frame_w > 0:
        points[:, 0] = points[:, 0] / frame_w
    if frame_h > 0:
        points[:, 1] = points[:, 1] / frame_h
    mask = conf >= min_confidence
    return points[mask], conf[mask]


def _mean_point(
    keypoints_xy: Optional[np.ndarray],
    keypoints_conf: Optional[np.ndarray],
    frame_shape: Tuple[int, int],
    names: Iterable[str],
    min_confidence: float = 0.2,
) -> Optional[np.ndarray]:
    if keypoints_xy is None or keypoints_conf is None:
        return None
    frame_h, frame_w = frame_shape
    selected: List[np.ndarray] = []
    for name in names:
        index = COCO_KEYPOINTS[name]
        if index >= len(keypoints_conf) or float(keypoints_conf[index]) < min_confidence:
            continue
        point = np.asarray(keypoints_xy[index], dtype=np.float32).copy()
        point[0] = point[0] / max(1.0, float(frame_w))
        point[1] = point[1] / max(1.0, float(frame_h))
        selected.append(point)
    if not selected:
        return None
    return np.mean(np.stack(selected, axis=0), axis=0)


def _horizontal_score(point_a: Optional[np.ndarray], point_b: Optional[np.ndarray]) -> float:
    if point_a is None or point_b is None:
        return 0.0
    vector = point_b - point_a
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return 0.0
    horizontal_component = abs(float(vector[0])) / norm
    return float(np.clip(horizontal_component, 0.0, 1.0))


def _leg_fold_score(hip: Optional[np.ndarray], knee: Optional[np.ndarray], ankle: Optional[np.ndarray]) -> float:
    if hip is None or knee is None:
        return 0.0
    hip_knee_dist = float(np.linalg.norm(knee - hip))
    if ankle is None:
        return float(np.clip(1.0 - hip_knee_dist / 0.25, 0.0, 1.0))
    knee_ankle_dist = float(np.linalg.norm(ankle - knee))
    return float(np.clip(1.0 - (hip_knee_dist + knee_ankle_dist) / 0.55, 0.0, 1.0))


def compute_motion_inside_box(
    prev_gray: Optional[np.ndarray],
    gray: np.ndarray,
    bbox_xyxy: Sequence[float],
) -> float:
    if prev_gray is None or prev_gray.shape != gray.shape:
        return 0.0
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, x1 + 1, w))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, y1 + 1, h))
    prev_crop = prev_gray[y1:y2, x1:x2]
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0 or prev_crop.size == 0:
        return 0.0
    diff = cv2.absdiff(prev_crop, crop)
    return float(np.mean(diff) / 255.0)


def compute_feature_vector(
    observation: Optional[PersonObservation],
    previous_feature: Optional[np.ndarray] = None,
    motion_mean: float = 0.0,
) -> np.ndarray:
    features = np.zeros((FEATURE_DIM,), dtype=np.float32)
    if observation is None:
        if previous_feature is not None:
            features[:8] = previous_feature[:8]
            features[26] = 1.0
        return features

    center_x, center_y, width, height, aspect, area, bottom_y, top_y = normalize_bbox(
        observation.bbox_xyxy, observation.frame_shape
    )
    features[0] = center_x
    features[1] = center_y
    features[2] = width
    features[3] = height
    features[4] = min(aspect, 5.0)
    features[5] = area
    features[6] = bottom_y
    features[7] = float(np.clip(observation.confidence, 0.0, 1.0))

    visible_points, visible_conf = _visible_points(
        observation.keypoints_xy, observation.keypoints_conf, observation.frame_shape
    )
    features[8] = float(len(visible_points) / 17.0) if len(visible_points) else 0.0
    features[33] = float(np.mean(visible_conf)) if len(visible_conf) else 0.0

    shoulder = _mean_point(
        observation.keypoints_xy,
        observation.keypoints_conf,
        observation.frame_shape,
        ["left_shoulder", "right_shoulder"],
    )
    hip = _mean_point(
        observation.keypoints_xy,
        observation.keypoints_conf,
        observation.frame_shape,
        ["left_hip", "right_hip"],
    )
    knee = _mean_point(
        observation.keypoints_xy,
        observation.keypoints_conf,
        observation.frame_shape,
        ["left_knee", "right_knee"],
    )
    ankle = _mean_point(
        observation.keypoints_xy,
        observation.keypoints_conf,
        observation.frame_shape,
        ["left_ankle", "right_ankle"],
    )
    head = _mean_point(
        observation.keypoints_xy,
        observation.keypoints_conf,
        observation.frame_shape,
        ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    )

    features[9] = _horizontal_score(shoulder, hip)
    features[10] = float(hip[1]) if hip is not None else center_y
    features[11] = float(shoulder[1]) if shoulder is not None else max(top_y, center_y - height * 0.25)
    features[12] = float(ankle[1]) if ankle is not None else bottom_y
    features[13] = float(knee[1]) if knee is not None else center_y
    features[14] = _horizontal_score(head, ankle)
    features[27] = _horizontal_score(shoulder, hip)
    features[28] = _leg_fold_score(hip, knee, ankle)
    features[29] = float(head[1]) if head is not None else top_y

    if len(visible_points):
        features[15] = float(np.max(visible_points[:, 1]) - np.min(visible_points[:, 1]))
        features[16] = float(np.max(visible_points[:, 0]) - np.min(visible_points[:, 0]))
    else:
        features[15] = height
        features[16] = width

    if previous_feature is not None:
        features[17] = features[1] - previous_feature[1]
        features[18] = features[0] - previous_feature[0]
        features[19] = features[3] - previous_feature[3]
        features[20] = features[2] - previous_feature[2]
        features[21] = features[5] - previous_feature[5]
        features[22] = features[4] - previous_feature[4]
        features[23] = float(math.sqrt(float(features[17] ** 2 + features[18] ** 2)))
        features[24] = max(0.0, float(features[17]))
        features[35] = float(features[21] / max(1e-6, abs(float(previous_feature[5]))))
        features[36] = float(-features[19])

    features[25] = float(np.clip(motion_mean, 0.0, 1.0))
    features[26] = 0.0
    features[30] = top_y
    features[31] = center_x - width * 0.5
    features[32] = center_x + width * 0.5
    features[34] = float(math.sqrt(width * width + height * height))
    return features


class YoloPersonPoseDetector:
    def __init__(
        self,
        model_name: str,
        confidence: float,
        iou: float,
        image_size: int,
        max_detections: int,
        device: str = "auto",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_name = model_name
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.max_detections = int(max_detections)
        self.device = select_device(device)
        self.logger = logger or logging.getLogger(__name__)
        try:
            yolo_config_dir = PROJECT_ROOT / ".ultralytics"
            yolo_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for detection. Install dependencies with "
                "`pip install -r requirements.txt` using the same Python interpreter that runs this script.\n"
                f"Current interpreter: {sys.executable}\n"
                f"Recommended command: \"{sys.executable}\" -m pip install ultralytics"
            ) from exc
        self.model = YOLO(model_name)
        self.logger.info("Loaded detector %s on %s", model_name, self.device)

    def detect(self, frame: np.ndarray, timestamp: float) -> List[PersonObservation]:
        if frame is None or frame.size == 0:
            return []
        frame_h, frame_w = frame.shape[:2]
        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.image_size,
                max_det=self.max_detections,
                classes=[0],
                verbose=False,
                device=self.device,
            )
        except TypeError:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.image_size,
                max_det=self.max_detections,
                verbose=False,
                device=self.device,
            )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        keypoints_container = getattr(result, "keypoints", None)
        keypoints_xy = None
        keypoints_conf = None
        if keypoints_container is not None and getattr(keypoints_container, "xy", None) is not None:
            keypoints_xy = keypoints_container.xy.detach().cpu().numpy()
            if getattr(keypoints_container, "conf", None) is not None:
                keypoints_conf = keypoints_container.conf.detach().cpu().numpy()

        observations: List[PersonObservation] = []
        xyxy_array = boxes.xyxy.detach().cpu().numpy()
        conf_array = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy_array),))
        cls_array = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros((len(xyxy_array),))

        for index, bbox in enumerate(xyxy_array):
            class_id = int(cls_array[index]) if index < len(cls_array) else 0
            if class_id != 0:
                continue
            confidence = float(conf_array[index]) if index < len(conf_array) else 0.0
            k_xy = None
            k_conf = None
            if keypoints_xy is not None and index < len(keypoints_xy):
                k_xy = keypoints_xy[index].astype(np.float32)
                if keypoints_conf is not None and index < len(keypoints_conf):
                    k_conf = keypoints_conf[index].astype(np.float32)
                else:
                    k_conf = np.ones((k_xy.shape[0],), dtype=np.float32)
            observations.append(
                PersonObservation(
                    bbox_xyxy=np.asarray(bbox, dtype=np.float32),
                    confidence=confidence,
                    keypoints_xy=k_xy,
                    keypoints_conf=k_conf,
                    frame_shape=(frame_h, frame_w),
                    timestamp=timestamp,
                    class_id=class_id,
                )
            )
        observations.sort(key=lambda obs: obs.area, reverse=True)
        return suppress_duplicate_person_observations(observations)


def choose_primary_person(observations: Sequence[PersonObservation]) -> Optional[PersonObservation]:
    if not observations:
        return None
    return max(observations, key=lambda obs: obs.area * max(obs.confidence, 1e-3))


def labels_for_timestamps(timestamps: np.ndarray, annotations: Sequence[Dict[str, object]]) -> np.ndarray:
    labels = np.zeros((len(timestamps),), dtype=np.int64)
    for index, timestamp in enumerate(timestamps):
        selected = "NORMAL"
        for annotation in annotations:
            start = float(annotation["start_sec"])
            end = float(annotation["end_sec"])
            if start <= float(timestamp) <= end:
                selected = str(annotation["label"])
                break
        labels[index] = LABEL_TO_ID.get(selected, LABEL_TO_ID["NORMAL"])
    return labels


def extract_features_from_video(
    video_path: str | Path,
    annotations: Sequence[Dict[str, object]],
    detector: YoloPersonPoseDetector,
    sample_fps: float,
    cache_path: Optional[str | Path],
    force_rebuild: bool,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_path is not None:
        resolved_cache = resolve_path(cache_path)
        if resolved_cache.exists() and not force_rebuild:
            data = np.load(resolved_cache, allow_pickle=False)
            return data["features"].astype(np.float32), data["labels"].astype(np.int64), data["timestamps"].astype(np.float32)
    else:
        resolved_cache = None

    resolved_video = resolve_path(video_path)
    cap = cv2.VideoCapture(str(resolved_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {resolved_video}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if source_fps <= 0:
        source_fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(source_fps / max(sample_fps, 0.1))))

    features: List[np.ndarray] = []
    timestamps: List[float] = []
    previous_feature: Optional[np.ndarray] = None
    previous_gray: Optional[np.ndarray] = None

    progress_total = frame_count if frame_count > 0 else None
    progress = tqdm(total=progress_total, desc=f"features {resolved_video.name}", unit="frame", leave=False)
    frame_index = 0
    sampled_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride == 0:
            timestamp = float(frame_index / source_fps)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            observations = detector.detect(frame, timestamp)
            primary = choose_primary_person(observations)
            motion_value = 0.0
            if primary is not None:
                motion_value = compute_motion_inside_box(previous_gray, gray, primary.bbox_xyxy)
            feature = compute_feature_vector(primary, previous_feature, motion_value)
            features.append(feature)
            timestamps.append(timestamp)
            previous_feature = feature
            previous_gray = gray
            sampled_index += 1
        frame_index += 1
        progress.update(1)
    progress.close()
    cap.release()

    if not features:
        raise RuntimeError(f"No frames extracted from video: {resolved_video}")

    feature_array = np.stack(features, axis=0).astype(np.float32)
    timestamp_array = np.asarray(timestamps, dtype=np.float32)
    label_array = labels_for_timestamps(timestamp_array, annotations)

    if resolved_cache is not None:
        resolved_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            resolved_cache,
            features=feature_array,
            labels=label_array,
            timestamps=timestamp_array,
        )
        logger.debug("Cached features: %s", resolved_cache)
    return feature_array, label_array, timestamp_array

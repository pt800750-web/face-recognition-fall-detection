from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .features import (
    FEATURE_DIM,
    ID_TO_LABEL,
    YoloPersonPoseDetector,
    extract_features_from_video,
)
from .utils import read_jsonl, resolve_path


class FeatureAugmentor:
    def __init__(self, noise_std: float = 0.015, temporal_dropout: float = 0.05) -> None:
        self.noise_std = float(noise_std)
        self.temporal_dropout = float(temporal_dropout)

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        augmented = sequence.copy()
        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, size=augmented.shape).astype(np.float32)
            noise[:, 26] = 0.0
            augmented += noise
        if self.temporal_dropout > 0 and len(augmented) > 2:
            mask = np.random.rand(len(augmented)) < self.temporal_dropout
            for index in np.where(mask)[0]:
                if index > 0:
                    augmented[index] = augmented[index - 1]
        return np.clip(augmented, -5.0, 5.0).astype(np.float32)


class FallFeatureSequenceDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        config: Dict[str, object],
        split: str,
        logger: logging.Logger,
        detector: Optional[YoloPersonPoseDetector] = None,
        augment: bool = False,
    ) -> None:
        self.manifest_path = resolve_path(manifest_path)
        self.config = config
        self.split = split
        self.logger = logger
        self.rows = read_jsonl(self.manifest_path)
        self.sequence_length = int(config["features"]["sequence_length"])  # type: ignore[index]
        self.window_stride = int(config["features"]["window_stride"])  # type: ignore[index]
        self.sample_fps = float(config["features"]["sample_fps"])  # type: ignore[index]
        self.force_rebuild_cache = bool(config["features"]["force_rebuild_cache"])  # type: ignore[index]
        self.cache_features = bool(config["features"]["cache_features"])  # type: ignore[index]
        self.feature_dir = resolve_path(config["paths"]["feature_dir"])  # type: ignore[index]
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.detector = detector
        self.augmentor = FeatureAugmentor(
            noise_std=float(config["train"]["augment_noise_std"]),  # type: ignore[index]
            temporal_dropout=float(config["train"]["augment_temporal_dropout"]),  # type: ignore[index]
        ) if augment else None

        self.video_features: List[np.ndarray] = []
        self.video_labels: List[np.ndarray] = []
        self.video_timestamps: List[np.ndarray] = []
        self.video_ids: List[str] = []
        self.samples: List[Tuple[int, int, int]] = []
        self.skipped: List[Dict[str, str]] = []
        self._build()

    def _cache_path_for_row(self, row: Dict[str, object]) -> Optional[Path]:
        if not self.cache_features:
            return None
        video_id = str(row.get("sample_id", row.get("video_id", "unknown_sample")))
        return self.feature_dir / f"{video_id}.npz"

    def _build(self) -> None:
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        if self.detector is None:
            detector_cfg = self.config["detector"]  # type: ignore[index]
            self.detector = YoloPersonPoseDetector(
                model_name=str(detector_cfg["model"]),
                confidence=float(detector_cfg["confidence"]),
                iou=float(detector_cfg["iou"]),
                image_size=int(detector_cfg["image_size"]),
                max_detections=int(detector_cfg["max_detections"]),
                device=str(detector_cfg["device"]),
                logger=self.logger,
            )

        for row in tqdm(self.rows, desc=f"build {self.split}", unit="video"):
            video_id = str(row.get("sample_id", row.get("video_id", "unknown_sample")))
            try:
                features, labels, timestamps = extract_features_from_video(
                    video_path=str(row["video_path"]),
                    annotations=row.get("segments", row.get("annotations", [])),  # type: ignore[arg-type]
                    detector=self.detector,
                    sample_fps=self.sample_fps,
                    cache_path=self._cache_path_for_row(row),
                    force_rebuild=self.force_rebuild_cache,
                    logger=self.logger,
                )
                if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
                    raise ValueError(f"Expected features [N,{FEATURE_DIM}], got {features.shape}")
                if len(features) != len(labels):
                    raise ValueError("Feature and label lengths do not match.")
            except Exception as exc:
                self.logger.warning("Skipping video %s during dataset build: %s", video_id, exc)
                self.skipped.append({"video_id": video_id, "error": str(exc)})
                continue

            video_index = len(self.video_features)
            self.video_features.append(features.astype(np.float32))
            self.video_labels.append(labels.astype(np.int64))
            self.video_timestamps.append(timestamps.astype(np.float32))
            self.video_ids.append(video_id)

            n_frames = len(features)
            if n_frames < self.sequence_length:
                self.samples.append((video_index, 0, n_frames))
            else:
                for end in range(self.sequence_length, n_frames + 1, self.window_stride):
                    self.samples.append((video_index, end - self.sequence_length, end))
                if self.samples and self.samples[-1][2] != n_frames:
                    self.samples.append((video_index, n_frames - self.sequence_length, n_frames))

        if not self.samples:
            raise RuntimeError(
                f"No usable training samples were built from {self.manifest_path}. "
                "Check dataset paths, detector dependency, and annotation files."
            )
        self.logger.info(
            "%s dataset: %d videos, %d samples, skipped=%d",
            self.split,
            len(self.video_features),
            len(self.samples),
            len(self.skipped),
        )

    def label_counts(self, num_classes: int) -> np.ndarray:
        counts = np.zeros((num_classes,), dtype=np.int64)
        for video_index, start, end in self.samples:
            label = self._select_target_label(self.video_labels[video_index], start, end)
            counts[label] += 1
        return counts

    def _select_target_label(self, labels: np.ndarray, start: int, end: int) -> int:
        window_labels = labels[start:end]
        if len(window_labels) == 0:
            return 0

        final_label = int(labels[max(0, end - 1)])
        falling_id = 5
        fallen_id = 6

        # FALLING is short in real videos. If the transition appears near the
        # decision time, train the model to fire FALLING instead of learning
        # only the easier post-impact FALLEN state.
        tail = window_labels[len(window_labels) // 2 :]
        if np.any(tail == falling_id):
            return falling_id

        # If the current frame is already post-impact, keep FALLEN. This keeps
        # a clear separation between a fall event and someone intentionally
        # lying down at the end of a slow movement.
        if final_label == fallen_id:
            return fallen_id

        return final_label

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_index, start, end = self.samples[index]
        features = self.video_features[video_index][start:end]
        labels = self.video_labels[video_index]
        target = self._select_target_label(labels, start, end)

        if len(features) < self.sequence_length:
            pad_count = self.sequence_length - len(features)
            pad_value = features[0:1] if len(features) else np.zeros((1, FEATURE_DIM), dtype=np.float32)
            features = np.concatenate([np.repeat(pad_value, pad_count, axis=0), features], axis=0)

        if self.augmentor is not None:
            features = self.augmentor(features)

        return torch.from_numpy(features.astype(np.float32)), torch.tensor(target, dtype=torch.long)

    def describe_labels(self, num_classes: int) -> str:
        counts = self.label_counts(num_classes)
        return ", ".join(f"{ID_TO_LABEL.get(index, str(index))}={int(count)}" for index, count in enumerate(counts))

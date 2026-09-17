from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import create_video_writer, make_unique_path, resolve_path, safe_timestamp


class EventClipRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        prefix: str,
        fps: float,
        frame_size: Tuple[int, int],
        pre_event_frames: int,
        post_event_frames: int,
        source_name: str = "source",
    ) -> None:
        self.output_dir = resolve_path(output_dir)
        self.prefix = prefix
        self.fps = float(fps)
        self.frame_size = frame_size
        self.pre_event_frames = max(0, int(pre_event_frames))
        self.post_event_frames = max(0, int(post_event_frames))
        self.source_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in source_name)[:80]
        self.buffer: Deque[np.ndarray] = deque(maxlen=max(1, self.pre_event_frames))
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_path: Optional[Path] = None
        self.last_active_frame: Optional[int] = None
        self.segment_index = 0

    def _start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segment_index += 1
        self.current_path = make_unique_path(
            self.output_dir / f"{self.prefix}_{self.source_name}_{safe_timestamp()}_{self.segment_index:03d}.mp4"
        )
        self.writer = create_video_writer(self.current_path, self.fps, self.frame_size)
        for buffered_frame in self.buffer:
            self.writer.write(buffered_frame)

    def update(self, frame: np.ndarray, active: bool, frame_index: int) -> None:
        if active:
            self.last_active_frame = frame_index
            if self.writer is None:
                self._start()

        if self.writer is not None:
            self.writer.write(frame)
            if (
                not active
                and self.last_active_frame is not None
                and frame_index - self.last_active_frame > self.post_event_frames
            ):
                self.close()

        self.buffer.append(frame.copy())

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.current_path = None
            self.last_active_frame = None


class EventLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = make_unique_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "timestamp_sec",
                "frame_index",
                "track_id",
                "status",
                "raw_label",
                "model_confidence",
                "fall_score",
                "reason",
            ],
        )
        self.writer.writeheader()
        self.last_status: Dict[int, str] = {}

    def log_if_changed(
        self,
        timestamp_sec: float,
        frame_index: int,
        track_id: int,
        status: str,
        raw_label: str,
        model_confidence: float,
        fall_score: float,
        reason: str,
    ) -> None:
        if self.last_status.get(track_id) == status:
            return
        self.last_status[track_id] = status
        self.writer.writerow(
            {
                "timestamp_sec": f"{timestamp_sec:.3f}",
                "frame_index": int(frame_index),
                "track_id": int(track_id),
                "status": status,
                "raw_label": raw_label,
                "model_confidence": f"{model_confidence:.4f}",
                "fall_score": f"{fall_score:.4f}",
                "reason": reason,
            }
        )
        self.file.flush()

    def close(self) -> None:
        self.file.close()

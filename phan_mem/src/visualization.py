from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


STATUS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "NORMAL": (70, 220, 70),
    "WALKING_STANDING": (60, 210, 120),
    "SITTING": (255, 210, 60),
    "BENDING": (255, 150, 40),
    "LYING_INTENTIONAL": (255, 180, 40),
    "FALLING": (0, 180, 255),
    "FALLEN": (0, 0, 255),
    "FAINT": (120, 0, 255),
    "UNCONSCIOUS": (120, 0, 255),
    "OCCLUDED": (160, 160, 160),
}


def draw_track(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    track_id: int,
    status: str,
    confidence: float,
    fall_score: float,
) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    color = STATUS_COLORS.get(status, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {track_id} | {status} | p={confidence:.2f} s={fall_score:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    top = max(0, y1 - text_h - baseline - 6)
    cv2.rectangle(frame, (x1, top), (x1 + text_w + 8, top + text_h + baseline + 6), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, top + text_h + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_header(frame: np.ndarray, text: str) -> np.ndarray:
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (8, 8), (20 + text_w, 18 + text_h + baseline), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (14, 14 + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame

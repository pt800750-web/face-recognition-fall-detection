from __future__ import annotations

import re
from typing import Dict, List, Tuple


UNIFIED_LABELS: List[str] = [
    "NORMAL",
    "WALKING_STANDING",
    "SITTING",
    "BENDING",
    "LYING_INTENTIONAL",
    "FALLING",
    "FALLEN",
    "UNCONSCIOUS",
]

LABEL_TO_ID: Dict[str, int] = {name: index for index, name in enumerate(UNIFIED_LABELS)}
ID_TO_LABEL: Dict[int, str] = {index: name for name, index in LABEL_TO_ID.items()}

FALL_LABELS = {"FALLING", "FALLEN", "UNCONSCIOUS"}
NON_FALL_LABELS = {"NORMAL", "WALKING_STANDING", "SITTING", "BENDING", "LYING_INTENTIONAL"}


UP_FALL_ACTIVITY_MAP: Dict[int, str] = {
    1: "FALLING",
    2: "FALLING",
    3: "FALLING",
    4: "FALLING",
    5: "FALLING",
    6: "WALKING_STANDING",
    7: "WALKING_STANDING",
    8: "SITTING",
    9: "BENDING",
    10: "NORMAL",
    11: "LYING_INTENTIONAL",
}

UP_FALL_ACTIVITY_NAMES: Dict[int, str] = {
    1: "falling forward using hands",
    2: "falling forward using knees",
    3: "falling backward",
    4: "falling sideward",
    5: "falling sitting in empty chair",
    6: "walking",
    7: "standing",
    8: "sitting",
    9: "picking up an object",
    10: "jumping",
    11: "lying down",
}


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def label_id(label: str) -> int:
    normalized = str(label).upper()
    if normalized not in LABEL_TO_ID:
        raise KeyError(f"Unknown unified label: {label}")
    return LABEL_TO_ID[normalized]


def is_fall_label(label: str) -> bool:
    return str(label).upper() in FALL_LABELS


def map_raw_label(raw_label: str, dataset_name: str = "", context: str = "") -> str:
    text = normalize_text(f"{dataset_name} {context} {raw_label}")
    if not text:
        return "NORMAL"

    if any(token in text for token in ["unconscious", "syncope", "faint"]):
        return "UNCONSCIOUS"
    if any(token in text for token in ["fallen", "after fall", "post fall", "lying after"]):
        return "FALLEN"
    if any(token in text for token in ["fall", "falling", "slip", "collapse", "trip"]):
        return "FALLING"
    if any(token in text for token in ["sleep", "sleeping", "laying", "lying", "lie down", "rest", "bed", "sofa"]):
        return "LYING_INTENTIONAL"
    if any(token in text for token in ["sit", "sitting", "seated", "chair", "squat"]):
        return "SITTING"
    if any(token in text for token in ["bend", "bending", "pick", "picking", "crouch", "kneel", "push up", "pushup"]):
        return "BENDING"
    if any(token in text for token in ["walk", "walking", "stand", "standing", "run", "door"]):
        return "WALKING_STANDING"
    if any(token in text for token in ["jump", "jumping", "applaud", "normal", "adl"]):
        return "NORMAL"
    return "NORMAL"


def split_fall_segment(
    start_sec: float,
    end_sec: float,
    transition_seconds: float,
    impact_margin_seconds: float = 0.0,
) -> List[Tuple[str, float, float]]:
    start = float(start_sec)
    end = float(end_sec)
    if end <= start:
        return []
    transition = max(0.2, min(float(transition_seconds), end - start))
    falling_end = min(end, start + transition + max(0.0, float(impact_margin_seconds)))
    segments: List[Tuple[str, float, float]] = [("FALLING", start, falling_end)]
    if falling_end < end:
        segments.append(("FALLEN", falling_end, end))
    return segments


def make_segment(
    label: str,
    start_sec: float,
    end_sec: float,
    source_label: str,
    confidence: float = 1.0,
    is_pseudo: bool = False,
) -> Dict[str, object]:
    safe_label = label if label in LABEL_TO_ID else map_raw_label(label)
    return {
        "label": safe_label,
        "label_id": LABEL_TO_ID[safe_label],
        "start_sec": round(max(0.0, float(start_sec)), 3),
        "end_sec": round(max(0.0, float(end_sec)), 3),
        "source_label": str(source_label),
        "confidence": float(confidence),
        "is_pseudo": bool(is_pseudo),
    }


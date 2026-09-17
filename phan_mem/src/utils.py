from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_WRITER_CODEC_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    ".mp4": ("avc1", "H264", "X264", "mp4v"),
    ".mov": ("avc1", "H264", "X264", "mp4v"),
    ".mkv": ("X264", "avc1", "H264", "mp4v"),
    ".avi": ("MJPG", "XVID"),
    ".webm": ("VP90", "VP80"),
}
BROWSER_SAFE_VIDEO_CODECS = {"h264", "avc1", "vp80", "vp90", "vp8", "vp9"}


def project_path(*parts: str | os.PathLike[str]) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {path}")
    return config


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_project_dirs(config: Dict[str, Any]) -> None:
    default_keys = [
        "raw_dir",
        "processed_dir",
        "cache_dir",
        "feature_dir",
        "generated_video_dir",
        "manifest_dir",
        "checkpoints_dir",
        "outputs_dir",
        "logs_dir",
    ]
    for key in default_keys:
        if key in config.get("paths", {}):
            ensure_dir(config["paths"][key])
    for dataset_config in config.get("datasets", {}).values():
        if isinstance(dataset_config, dict) and "root_dir" in dataset_config:
            ensure_dir(dataset_config["root_dir"])


def setup_logging(config: Dict[str, Any], log_name: str) -> logging.Logger:
    ensure_project_dirs(config)
    log_level = getattr(logging, str(config["project"].get("log_level", "INFO")).upper(), logging.INFO)
    logs_dir = resolve_path(config["paths"]["logs_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{log_name}_{timestamp}.log"

    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("Logging to %s", log_path)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def select_device(device_name: str = "auto") -> str:
    normalized = str(device_name).lower().strip()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return normalized


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"JSONL file not found: {resolved}")
    rows: List[Dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {resolved}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Dict[str, Any] | List[Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as file:
        return json.load(file)


def md5_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    resolved = resolve_path(path)
    digest = hashlib.md5()
    with resolved.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_timestamp() -> str:
    return datetime.now().strftime("%d-%m-%Y_%H-%M")


def make_unique_path(path_value: str | Path) -> Path:
    resolved = resolve_path(path_value)
    if not resolved.exists():
        return resolved
    for index in range(1, 1000):
        candidate = resolved.with_name(f"{resolved.stem}_{index:02d}{resolved.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot allocate unique output path: {resolved}")


def make_video_id(video_path: str | Path, dataset_root: str | Path) -> str:
    video = resolve_path(video_path)
    root = resolve_path(dataset_root)
    try:
        relative = video.relative_to(root)
    except ValueError:
        relative = video.name
    text = str(relative).replace("\\", "/")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    stem = Path(text).stem.replace(" ", "_")
    return f"{stem}_{digest}"


def relative_to_project(path: str | Path) -> str:
    resolved = resolve_path(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def get_video_metadata(video_path: str | Path) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    resolved = resolve_path(video_path)
    if not resolved.exists():
        return False, {}, "file_not_found"
    cap = cv2.VideoCapture(str(resolved))
    if not cap.isOpened():
        cap.release()
        return False, {}, "cannot_open_video"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, _ = cap.read()
    cap.release()

    if not ok:
        return False, {}, "cannot_read_first_frame"
    if fps <= 0.0:
        fps = 30.0
    duration = float(frame_count / fps) if frame_count > 0 else 0.0
    return True, {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }, None


def get_video_fourcc(video_path: str | Path) -> Optional[str]:
    resolved = resolve_path(video_path)
    if not resolved.exists():
        return None
    cap = cv2.VideoCapture(str(resolved))
    if not cap.isOpened():
        cap.release()
        return None
    try:
        fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    finally:
        cap.release()
    text = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    normalized = text.replace("\x00", "").strip().lower()
    return normalized or None


def is_browser_playable_video(video_path: str | Path) -> bool:
    resolved = resolve_path(video_path)
    if resolved.suffix.lower() == ".webm":
        return True
    fourcc = get_video_fourcc(resolved)
    return fourcc in BROWSER_SAFE_VIDEO_CODECS


def download_with_resume(
    url: str,
    output_path: str | Path,
    expected_size: Optional[int],
    logger: logging.Logger,
    chunk_size: int = 1024 * 1024,
) -> None:
    resolved = resolve_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    existing_size = resolved.stat().st_size if resolved.exists() else 0
    headers: Dict[str, str] = {}
    mode = "wb"
    if existing_size > 0 and (expected_size is None or existing_size < expected_size):
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    if expected_size is not None and existing_size == expected_size:
        logger.info("Archive already exists with expected size: %s", resolved)
        return

    logger.info("Downloading %s", url)
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    if response.status_code == 200 and mode == "ab":
        logger.warning("Server ignored Range header. Restarting download from byte 0.")
        mode = "wb"
        existing_size = 0
    response.raise_for_status()

    total = expected_size if expected_size is not None else None
    progress = tqdm(
        total=total,
        initial=existing_size if mode == "ab" else 0,
        unit="B",
        unit_scale=True,
        desc="download",
    )
    with resolved.open(mode) as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            file.write(chunk)
            progress.update(len(chunk))
    progress.close()


def extract_zip(archive_path: str | Path, output_dir: str | Path, logger: logging.Logger) -> None:
    archive = resolve_path(archive_path)
    destination = resolve_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    logger.info("Extracting %s to %s", archive, destination)
    with zipfile.ZipFile(archive, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="extract", unit="file"):
            zf.extract(member, destination)


def parse_source(source: str) -> int | str:
    text = str(source).strip()
    if text.isdigit():
        return int(text)
    return text


def create_video_writer(
    path: str | Path,
    fps: float,
    frame_size: Tuple[int, int],
    codec_candidates: Optional[Sequence[str]] = None,
) -> cv2.VideoWriter:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    candidates = tuple(codec_candidates or VIDEO_WRITER_CODEC_CANDIDATES.get(resolved.suffix.lower(), ("mp4v",)))
    for codec in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(resolved), fourcc, float(fps), frame_size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Cannot open video writer: {resolved} codecs={','.join(candidates)}")


def transcode_video(
    source_path: str | Path,
    target_path: str | Path,
    codec_candidates: Optional[Sequence[str]] = None,
) -> Path:
    source = resolve_path(source_path)
    target = resolve_path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open source video for transcoding: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1.0 or fps > 240.0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    pending_frame: Optional[np.ndarray] = None

    try:
        if width <= 0 or height <= 0:
            ok, pending_frame = capture.read()
            if not ok or pending_frame is None:
                raise RuntimeError(f"Cannot read first frame from source video: {source}")
            height, width = pending_frame.shape[:2]

        writer = create_video_writer(target, fps, (width, height), codec_candidates=codec_candidates)
        wrote_frame = False
        try:
            if pending_frame is not None:
                writer.write(pending_frame)
                wrote_frame = True

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
                wrote_frame = True
        finally:
            writer.release()
    except Exception:
        if target.exists():
            target.unlink(missing_ok=True)
        raise
    finally:
        capture.release()

    if not wrote_frame or not target.exists() or target.stat().st_size <= 0:
        if target.exists():
            target.unlink(missing_ok=True)
        raise RuntimeError(f"Transcoding produced no playable output: {target}")
    return target


def checkpoint_payload_to_device(payload: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    return parser


def now_seconds() -> float:
    return time.time()

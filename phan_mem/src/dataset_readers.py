from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
from tqdm import tqdm

from .label_mapping import (
    LABEL_TO_ID,
    UP_FALL_ACTIVITY_MAP,
    UP_FALL_ACTIVITY_NAMES,
    make_segment,
    map_raw_label,
    split_fall_segment,
)
from .utils import get_video_metadata, make_video_id, relative_to_project, resolve_path


@dataclass
class ReaderResult:
    samples: List[Dict[str, object]]
    bad_files: List[Dict[str, str]]


def canonical_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(header).strip().lower()).strip("_")


def read_rows(path: Path) -> Tuple[List[Dict[str, str]], Optional[str]]:
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [{canonical_header(k): str(v) for k, v in item.items()} for item in payload if isinstance(item, dict)], None
            if isinstance(payload, dict):
                rows = payload.get("segments") or payload.get("annotations") or payload.get("data")
                if isinstance(rows, list):
                    return [{canonical_header(k): str(v) for k, v in item.items()} for item in rows if isinstance(item, dict)], None
            return [], "unsupported_json_annotation"

        with path.open("r", encoding="utf-8-sig", newline="") as file:
            sample = file.read(4096)
            file.seek(0)
            first_line = sample.splitlines()[0] if sample.splitlines() else ""
            delimiter = "," if "," in first_line else None
            if delimiter is None:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                reader = csv.DictReader(file, dialect=dialect)
            else:
                reader = csv.DictReader(file, delimiter=delimiter)
            if reader.fieldnames is None:
                return [], "missing_header"
            rows: List[Dict[str, str]] = []
            for row in reader:
                normalized = {
                    canonical_header(str(key)): str(value).strip()
                    for key, value in row.items()
                    if key is not None and value is not None
                }
                extra_values = row.get(None)
                if extra_values and reader.fieldnames:
                    last_column = canonical_header(str(reader.fieldnames[-1]))
                    joined_extra = ",".join(str(value).strip() for value in extra_values if value is not None)
                    if joined_extra:
                        normalized[last_column] = (normalized.get(last_column, "") + "," + joined_extra).strip(",")
                rows.append(normalized)
            return rows, None
    except Exception as exc:
        return [], str(exc)


def parse_time_ranges(text: str) -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    expression = str(text)
    for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(?:to|-|–|—)\s*(-?\d+(?:\.\d+)?)", expression, flags=re.I):
        start = float(match.group(1))
        end = float(match.group(2))
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return ranges


def parse_classes_field(
    classes_field: str,
    duration_sec: float,
    dataset_name: str,
    fall_transition_seconds: float,
    impact_margin_seconds: float,
) -> List[Dict[str, object]]:
    text = str(classes_field).replace("\n", " ").strip()
    if not text:
        return []
    segments: List[Dict[str, object]] = []
    pattern = re.compile(r"([A-Za-z0-9 _/&().-]+?)\s*\[([^\]]+)\]")
    for match in pattern.finditer(text):
        source_label = match.group(1).strip(" ;,")
        label = map_raw_label(source_label, dataset_name)
        ranges = parse_time_ranges(match.group(2))
        for start, end in ranges:
            start = max(0.0, min(start, duration_sec))
            end = max(0.0, min(end, duration_sec))
            if end <= start:
                continue
            if label == "FALLING":
                for split_label, split_start, split_end in split_fall_segment(
                    start, end, fall_transition_seconds, impact_margin_seconds
                ):
                    segments.append(make_segment(split_label, split_start, split_end, source_label, 1.0, False))
            else:
                segments.append(make_segment(label, start, end, source_label, 1.0, False))
    segments.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"])))
    return segments


def find_column(row: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in row:
            return candidate
    for key in row:
        for candidate in candidates:
            if candidate in key:
                return key
    return None


def annotation_candidates(video_path: Path, annotation_files: Sequence[Path]) -> List[Path]:
    candidates: List[Path] = []
    stem = video_path.stem.lower()
    parent = video_path.parent
    for annotation_file in annotation_files:
        lower_name = annotation_file.name.lower()
        if annotation_file.stem.lower() == stem or stem in lower_name:
            candidates.append(annotation_file)
        elif annotation_file.parent == parent:
            candidates.append(annotation_file)
        elif "annotation" in str(annotation_file.parent).lower() and annotation_file.parent.parent == parent.parent:
            candidates.append(annotation_file)
    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def infer_subject_from_path(path: Path, dataset_root: Path) -> str:
    try:
        parts = path.relative_to(dataset_root).parts
    except ValueError:
        parts = path.parts
    for part in parts:
        lower = part.lower()
        if "subject" in lower or re.fullmatch(r"s\d+", lower) or re.fullmatch(r"sa\d+", lower):
            return re.sub(r"[^A-Za-z0-9]+", "_", part).strip("_")
    for part in parts:
        match = re.search(r"(?:subject|sub|s|sa)[_\-\s]*(\d+)", part, flags=re.I)
        if match:
            return f"subject_{match.group(1)}"
    return "unknown_subject"


def infer_scene_from_path(path: Path, dataset_root: Path) -> str:
    try:
        parts = path.relative_to(dataset_root).parts
    except ValueError:
        parts = path.parts
    known = ["home", "coffee", "office", "lecture", "room", "camera", "cam", "adl", "fall"]
    scene_parts = [part for part in parts[:-1] if any(token in part.lower() for token in known)]
    return "_".join(scene_parts[-3:]) if scene_parts else "unknown_scene"


def make_fall_pseudo_segments(
    duration_sec: float,
    source_label: str,
    start_ratio: float,
    transition_ratio: float,
) -> List[Dict[str, object]]:
    duration = max(float(duration_sec), 0.1)
    fall_start = min(duration * float(start_ratio), max(0.0, duration - 0.4))
    fall_end = min(duration, fall_start + max(0.4, duration * float(transition_ratio)))
    segments = []
    if fall_start > 0.05:
        segments.append(make_segment("WALKING_STANDING", 0.0, fall_start, "pseudo_pre_fall_upright", 0.45, True))
    segments.append(make_segment("FALLING", fall_start, fall_end, source_label, 0.55, True))
    if fall_end < duration:
        segments.append(make_segment("FALLEN", fall_end, duration, "pseudo_after_fall_lying", 0.55, True))
    return segments


def make_single_state_segment(duration_sec: float, label: str, source_label: str, confidence: float, is_pseudo: bool) -> List[Dict[str, object]]:
    return [make_segment(label, 0.0, max(0.1, float(duration_sec)), source_label, confidence, is_pseudo)]


def write_image_sequence_video(image_files: Sequence[Path], output_path: Path, fps: float) -> Tuple[bool, Optional[str]]:
    if not image_files:
        return False, "empty_image_sequence"
    first = cv2.imread(str(image_files[0]))
    if first is None:
        return False, f"cannot_read_first_image:{image_files[0]}"
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return False, f"cannot_open_video_writer:{output_path}"
    written = 0
    for image_file in image_files:
        frame = cv2.imread(str(image_file))
        if frame is None:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
    writer.release()
    if written == 0:
        return False, "no_readable_images"
    return True, None


class BaseDatasetReader:
    dataset_name = "base"

    def __init__(self, root_dir: Path, config: Dict[str, object], logger: logging.Logger) -> None:
        self.root_dir = root_dir
        self.config = config
        self.logger = logger
        self.video_extensions = {ext.lower() for ext in config["dataset"]["video_extensions"]}  # type: ignore[index]
        self.image_extensions = {ext.lower() for ext in config["dataset"]["image_extensions"]}  # type: ignore[index]
        self.annotation_extensions = {ext.lower() for ext in config["dataset"]["annotation_extensions"]}  # type: ignore[index]
        self.prepare_config = config["prepare"]  # type: ignore[index]

    def scan(self) -> ReaderResult:
        raise NotImplementedError

    def _video_files(self) -> List[Path]:
        if not self.root_dir.exists():
            return []
        return sorted(path for path in self.root_dir.rglob("*") if path.is_file() and path.suffix.lower() in self.video_extensions)

    def _annotation_files(self) -> List[Path]:
        if not self.root_dir.exists():
            return []
        return sorted(path for path in self.root_dir.rglob("*") if path.is_file() and path.suffix.lower() in self.annotation_extensions)

    def _make_sample(
        self,
        video_path: Path,
        metadata: Dict[str, object],
        segments: List[Dict[str, object]],
        annotation_path: Optional[Path],
        label_source: str,
        extra: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        sample = {
            "dataset_name": self.dataset_name,
            "sample_id": f"{self.dataset_name}_{make_video_id(video_path, self.root_dir)}",
            "subject_id": infer_subject_from_path(video_path, self.root_dir),
            "scene": infer_scene_from_path(video_path, self.root_dir),
            "video_path": relative_to_project(video_path),
            "annotation_path": relative_to_project(annotation_path) if annotation_path else None,
            "label_source": label_source,
            "fps": float(metadata["fps"]),
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "frame_count": int(metadata["frame_count"]),
            "duration_sec": float(metadata["duration_sec"]),
            "segments": segments,
            "events": [segment for segment in segments if str(segment["label"]) in {"FALLING", "FALLEN"}],
        }
        if extra:
            sample.update(extra)
        return sample

    def _validate_video(self, video_path: Path) -> Tuple[bool, Dict[str, object], Optional[str]]:
        is_valid, metadata, error = get_video_metadata(video_path)
        if not is_valid:
            return False, metadata, error
        if float(metadata["duration_sec"]) < float(self.prepare_config["min_duration_sec"]):
            return False, metadata, "too_short_duration"
        if int(metadata["frame_count"]) < int(self.prepare_config["min_frames"]):
            return False, metadata, "too_few_frames"
        return True, metadata, None


class GMDCSA24Reader(BaseDatasetReader):
    dataset_name = "gmdcsa24"

    def _annotation_for_video(
        self,
        video_path: Path,
        annotation_files: Sequence[Path],
        duration_sec: float,
    ) -> Tuple[List[Dict[str, object]], Optional[Path], str]:
        candidates = []
        if video_path.parent.name.lower() in {"adl", "fall"}:
            for name in [f"{video_path.parent.name}.csv", f"{video_path.parent.name.lower()}.csv", f"{video_path.parent.name.upper()}.csv"]:
                candidate = video_path.parent.parent / name
                if candidate.exists():
                    candidates.append(candidate)
        candidates.extend(annotation_candidates(video_path, annotation_files))

        seen = set()
        for annotation_file in candidates:
            if annotation_file in seen:
                continue
            seen.add(annotation_file)
            rows, error = read_rows(annotation_file)
            if error:
                continue
            for row in rows:
                file_col = find_column(row, ["file_name", "filename", "video", "name"])
                classes_col = find_column(row, ["classes", "class", "labels", "annotation"])
                if file_col is None or classes_col is None:
                    continue
                row_file = Path(row[file_col]).name.lower()
                if row_file != video_path.name.lower() and Path(row_file).stem != video_path.stem.lower():
                    continue
                segments = parse_classes_field(
                    row.get(classes_col, ""),
                    duration_sec,
                    self.dataset_name,
                    float(self.prepare_config["fall_transition_seconds"]),
                    float(self.prepare_config["fall_impact_margin_seconds"]),
                )
                if segments:
                    return segments, annotation_file, "csv_segments"
        category = video_path.parent.name.lower()
        if category == "fall" and bool(self.prepare_config["allow_unannotated_fall"]):
            return make_fall_pseudo_segments(
                duration_sec,
                "pseudo_gmdcsa24_fall_clip",
                float(self.prepare_config["pseudo_fall_start_ratio"]),
                float(self.prepare_config["pseudo_fall_transition_ratio"]),
            ), None, "pseudo_from_fall_folder"
        if bool(self.prepare_config["allow_unannotated_adl_as_normal"]):
            label = map_raw_label(video_path.stem, self.dataset_name, str(video_path.parent))
            return make_single_state_segment(duration_sec, label, f"pseudo_{label.lower()}", 0.4, True), None, "pseudo_from_adl_folder"
        return [], None, "missing_annotation"

    def scan(self) -> ReaderResult:
        samples: List[Dict[str, object]] = []
        bad_files: List[Dict[str, str]] = []
        if not self.root_dir.exists():
            return ReaderResult(samples, [{"dataset": self.dataset_name, "path": str(self.root_dir), "error": "root_dir_not_found"}])
        annotation_files = self._annotation_files()
        for video_path in tqdm(self._video_files(), desc="scan gmdcsa24", unit="video"):
            valid, metadata, error = self._validate_video(video_path)
            if not valid:
                bad_files.append({"dataset": self.dataset_name, "path": str(video_path), "error": str(error)})
                continue
            segments, annotation_path, label_source = self._annotation_for_video(video_path, annotation_files, float(metadata["duration_sec"]))
            if not segments:
                bad_files.append({"dataset": self.dataset_name, "path": str(video_path), "error": label_source})
                continue
            samples.append(self._make_sample(video_path, metadata, segments, annotation_path, label_source))
        return ReaderResult(samples, bad_files)


class LE2IReader(BaseDatasetReader):
    dataset_name = "le2i"

    def _parse_le2i_annotation(self, annotation_path: Path, fps: float, duration_sec: float) -> List[Dict[str, object]]:
        text = annotation_path.read_text(encoding="utf-8", errors="ignore")
        ranges = parse_time_ranges(text)
        numeric_values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]
        if not ranges and len(numeric_values) >= 2:
            start, end = numeric_values[0], numeric_values[1]
            if start > duration_sec or end > duration_sec:
                start = start / max(fps, 1.0)
                end = end / max(fps, 1.0)
            ranges = [(start, end)]
        segments: List[Dict[str, object]] = []
        for start, end in ranges:
            start = max(0.0, min(start, duration_sec))
            end = max(0.0, min(end, duration_sec))
            if end <= start:
                continue
            if start > 0.05:
                segments.append(make_segment("WALKING_STANDING", 0.0, start, "le2i_pre_fall", 0.55, True))
            for label, split_start, split_end in split_fall_segment(
                start,
                end,
                float(self.prepare_config["fall_transition_seconds"]),
                float(self.prepare_config["fall_impact_margin_seconds"]),
            ):
                segments.append(make_segment(label, split_start, split_end, "le2i_annotation_fall", 0.75, False))
            if end < duration_sec:
                segments.append(make_segment("FALLEN", end, duration_sec, "le2i_after_fall", 0.65, True))
        return segments

    def _segments_for_video(
        self,
        video_path: Path,
        annotation_files: Sequence[Path],
        metadata: Dict[str, object],
    ) -> Tuple[List[Dict[str, object]], Optional[Path], str]:
        duration_sec = float(metadata["duration_sec"])
        fps = float(metadata["fps"])
        for candidate in annotation_candidates(video_path, annotation_files):
            try:
                segments = self._parse_le2i_annotation(candidate, fps, duration_sec)
            except Exception:
                continue
            if segments:
                return segments, candidate, "le2i_annotation_or_fall_frames"

        context = f"{video_path.parent} {video_path.stem}"
        label = map_raw_label(video_path.stem, self.dataset_name, context)
        lower_context = context.lower()
        if label == "FALLING" or "fall" in lower_context:
            return make_fall_pseudo_segments(
                duration_sec,
                "pseudo_le2i_fall_clip",
                float(self.prepare_config["pseudo_fall_start_ratio"]),
                float(self.prepare_config["pseudo_fall_transition_ratio"]),
            ), None, "pseudo_from_fall_name_or_folder"
        if any(token in lower_context for token in ["coffee", "home", "office", "lecture", "adl", "normal"]):
            inferred = label if label != "NORMAL" else "WALKING_STANDING"
            return make_single_state_segment(duration_sec, inferred, f"pseudo_le2i_{inferred.lower()}", 0.35, True), None, "pseudo_from_adl_context"
        return [], None, "annotation_not_found"

    def scan(self) -> ReaderResult:
        samples: List[Dict[str, object]] = []
        bad_files: List[Dict[str, str]] = []
        if not self.root_dir.exists():
            return ReaderResult(samples, [{"dataset": self.dataset_name, "path": str(self.root_dir), "error": "root_dir_not_found_manual_required"}])
        annotation_files = self._annotation_files()
        for video_path in tqdm(self._video_files(), desc="scan le2i", unit="video"):
            valid, metadata, error = self._validate_video(video_path)
            if not valid:
                bad_files.append({"dataset": self.dataset_name, "path": str(video_path), "error": str(error)})
                continue
            segments, annotation_path, label_source = self._segments_for_video(video_path, annotation_files, metadata)
            if not segments:
                bad_files.append({"dataset": self.dataset_name, "path": str(video_path), "error": label_source})
                continue
            samples.append(self._make_sample(video_path, metadata, segments, annotation_path, label_source))
        return ReaderResult(samples, bad_files)


class UPFallReader(BaseDatasetReader):
    dataset_name = "up_fall"

    def _activity_id(self, path: Path) -> Optional[int]:
        text = " ".join(path.parts + (path.stem,))
        patterns = [
            r"(?:activity|act|a)[_\-\s]*(\d{1,2})",
            r"(?:^|[^a-z])A(\d{1,2})(?:[^a-z]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                value = int(match.group(1))
                if 1 <= value <= 11:
                    return value
        for activity_id, activity_name in UP_FALL_ACTIVITY_NAMES.items():
            if activity_name.replace(" ", "").lower() in text.replace(" ", "").lower():
                return activity_id
        return None

    def _segments_from_activity(self, video_path: Path, duration_sec: float) -> Tuple[List[Dict[str, object]], str]:
        activity_id = self._activity_id(video_path)
        if activity_id is not None:
            label = UP_FALL_ACTIVITY_MAP[activity_id]
            source_label = f"UP-Fall activity {activity_id}: {UP_FALL_ACTIVITY_NAMES[activity_id]}"
            if label == "FALLING":
                return make_fall_pseudo_segments(
                    duration_sec,
                    source_label,
                    float(self.prepare_config["pseudo_fall_start_ratio"]),
                    float(self.prepare_config["pseudo_fall_transition_ratio"]),
                ), "pseudo_from_up_fall_activity_id"
            return make_single_state_segment(duration_sec, label, source_label, 0.75, True), "pseudo_from_up_fall_activity_id"
        inferred = map_raw_label(video_path.stem, self.dataset_name, str(video_path.parent))
        if inferred == "FALLING":
            return make_fall_pseudo_segments(
                duration_sec,
                "pseudo_up_fall_fall_keyword",
                float(self.prepare_config["pseudo_fall_start_ratio"]),
                float(self.prepare_config["pseudo_fall_transition_ratio"]),
            ), "pseudo_from_keyword"
        return make_single_state_segment(duration_sec, inferred, f"pseudo_up_fall_{inferred.lower()}", 0.35, True), "pseudo_from_keyword"

    def _extract_camera_zips(self, cache_root: Path) -> List[Path]:
        extracted_dirs: List[Path] = []
        if not self.root_dir.exists():
            return extracted_dirs
        zip_files = sorted(path for path in self.root_dir.rglob("*.zip") if path.is_file())
        for zip_path in zip_files:
            lower_name = zip_path.name.lower()
            if not any(token in lower_name for token in ["camera", "cam"]):
                continue
            target_dir = cache_root / re.sub(r"[^A-Za-z0-9_.-]+", "_", zip_path.stem)
            marker = target_dir / ".extracted"
            if marker.exists():
                extracted_dirs.append(target_dir)
                continue
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(target_dir)
                marker.write_text(str(zip_path), encoding="utf-8")
                extracted_dirs.append(target_dir)
            except Exception as exc:
                self.logger.warning("Could not extract UP-Fall camera zip %s: %s", zip_path, exc)
                shutil.rmtree(target_dir, ignore_errors=True)
        return extracted_dirs

    def _image_sequence_dirs(self, roots: Iterable[Path]) -> List[Path]:
        sequence_dirs: List[Path] = []
        min_frames = int(self.prepare_config["min_frames"])
        for root in roots:
            if not root.exists():
                continue
            for directory in [root] + [path for path in root.rglob("*") if path.is_dir()]:
                image_count = sum(1 for child in directory.iterdir() if child.is_file() and child.suffix.lower() in self.image_extensions)
                if image_count >= min_frames:
                    sequence_dirs.append(directory)
        unique: List[Path] = []
        seen = set()
        for directory in sequence_dirs:
            if directory not in seen:
                seen.add(directory)
                unique.append(directory)
        return unique

    def _generate_videos_from_images(self) -> Tuple[List[Path], List[Dict[str, str]]]:
        generated: List[Path] = []
        bad_files: List[Dict[str, str]] = []
        if not bool(self.prepare_config["create_videos_from_image_sequences"]):
            return generated, bad_files
        cache_root = resolve_path(self.config["paths"]["cache_dir"]) / "up_fall_extracted_images"  # type: ignore[index]
        generated_root = resolve_path(self.config["paths"]["generated_video_dir"]) / "up_fall"  # type: ignore[index]
        extracted_roots = self._extract_camera_zips(cache_root)
        roots = [self.root_dir] + extracted_roots
        for image_dir in tqdm(self._image_sequence_dirs(roots), desc="up_fall images->video", unit="seq"):
            image_files = sorted(
                [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in self.image_extensions],
                key=lambda path: [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name)],
            )
            try:
                relative_name = "_".join(image_dir.relative_to(self.root_dir).parts)
            except ValueError:
                relative_name = "_".join(image_dir.parts[-4:])
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", relative_name).strip("_") or image_dir.name
            output_path = generated_root / f"{safe_name}.mp4"
            if output_path.exists():
                generated.append(output_path)
                continue
            ok, error = write_image_sequence_video(image_files, output_path, float(self.prepare_config["generated_video_fps"]))
            if ok:
                generated.append(output_path)
            else:
                bad_files.append({"dataset": self.dataset_name, "path": str(image_dir), "error": str(error)})
        return generated, bad_files

    def scan(self) -> ReaderResult:
        samples: List[Dict[str, object]] = []
        bad_files: List[Dict[str, str]] = []
        if not self.root_dir.exists():
            return ReaderResult(samples, [{"dataset": self.dataset_name, "path": str(self.root_dir), "error": "root_dir_not_found_manual_required"}])
        generated_videos, generated_errors = self._generate_videos_from_images()
        bad_files.extend(generated_errors)
        video_files = sorted(set(self._video_files() + generated_videos))
        for video_path in tqdm(video_files, desc="scan up_fall", unit="video"):
            valid, metadata, error = self._validate_video(video_path)
            if not valid:
                bad_files.append({"dataset": self.dataset_name, "path": str(video_path), "error": str(error)})
                continue
            segments, label_source = self._segments_from_activity(video_path, float(metadata["duration_sec"]))
            extra = {}
            activity_id = self._activity_id(video_path)
            if activity_id is not None:
                extra["activity_id"] = activity_id
                extra["activity_name"] = UP_FALL_ACTIVITY_NAMES[activity_id]
            samples.append(self._make_sample(video_path, metadata, segments, None, label_source, extra))
        return ReaderResult(samples, bad_files)


def build_readers(config: Dict[str, object], logger: logging.Logger) -> List[BaseDatasetReader]:
    dataset_configs = config.get("datasets", {})
    enabled_names = dataset_configs.get("enabled", []) if isinstance(dataset_configs, dict) else []
    readers: List[BaseDatasetReader] = []
    for dataset_name in enabled_names:
        dataset_cfg = dataset_configs.get(dataset_name, {}) if isinstance(dataset_configs, dict) else {}
        if not isinstance(dataset_cfg, dict) or not bool(dataset_cfg.get("enabled", True)):
            continue
        root_dir = resolve_path(dataset_cfg["root_dir"])
        if dataset_name == "gmdcsa24":
            readers.append(GMDCSA24Reader(root_dir, config, logger))
        elif dataset_name == "le2i":
            readers.append(LE2IReader(root_dir, config, logger))
        elif dataset_name == "up_fall":
            readers.append(UPFallReader(root_dir, config, logger))
        else:
            logger.warning("Unknown dataset reader: %s", dataset_name)
    return readers


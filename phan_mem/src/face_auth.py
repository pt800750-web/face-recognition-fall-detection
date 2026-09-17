from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    import face_recognition
except ImportError:  # pragma: no cover - depends on local Windows setup.
    face_recognition = None

from .utils import PROJECT_ROOT


DEFAULT_FACE_TOLERANCE = 0.5


@dataclass
class FaceMatch:
    staff_id: str
    person_type: str
    person_label: str
    person_name: str
    distance: float


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def resolve_face_database_path() -> Path:
    configured = str(os.environ.get("FACE_AUTH_DATABASE", "") or "").strip()
    candidates: list[Path] = []
    if configured:
        path = Path(configured)
        candidates.append(path if path.is_absolute() else (PROJECT_ROOT / path))

    candidates.extend(
        [
            PROJECT_ROOT / "data" / "database.json",
            PROJECT_ROOT.parent / "data" / "database.json",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def decode_image_data_url(image_data: str) -> np.ndarray:
    text = str(image_data or "").strip()
    if "," in text and text.lower().startswith("data:image"):
        text = text.split(",", 1)[1]
    if not text:
        raise ValueError("Thiếu ảnh khuôn mặt.")

    try:
        raw = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise ValueError("Ảnh gửi lên không đúng định dạng base64.") from exc

    buffer = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Không đọc được ảnh khuôn mặt.")
    return frame


class FaceAuthenticator:
    def __init__(
        self,
        database_path: Path,
        tolerance: float = DEFAULT_FACE_TOLERANCE,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.tolerance = float(tolerance)
        self.logger = logger or logging.getLogger("face_auth")
        self.known_faces: list[Dict[str, Any]] = []
        self.last_loaded_mtime: Optional[float] = None
        self.reload()

    @classmethod
    def from_env(cls, logger: Optional[logging.Logger] = None) -> "FaceAuthenticator":
        return cls(
            database_path=resolve_face_database_path(),
            tolerance=env_float("FACE_AUTH_TOLERANCE", DEFAULT_FACE_TOLERANCE),
            logger=logger,
        )

    @property
    def available(self) -> bool:
        return face_recognition is not None

    def reload_if_changed(self) -> None:
        try:
            mtime = self.database_path.stat().st_mtime
        except OSError:
            mtime = None
        if mtime != self.last_loaded_mtime:
            self.reload()

    def reload(self) -> None:
        self.known_faces = []
        if not self.database_path.exists():
            self.last_loaded_mtime = None
            self.logger.warning("Face database not found: %s", self.database_path)
            return

        try:
            records = json.loads(self.database_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("Could not load face database %s: %s", self.database_path, exc)
            records = []

        if not isinstance(records, list):
            records = []

        for record in records:
            if not isinstance(record, dict):
                continue
            staff_info = record.get("staff_info", {}) or {}
            if not isinstance(staff_info, dict):
                continue
            if staff_info.get("active") is False:
                continue
            staff_id = str(record.get("staff_id") or staff_info.get("staff_id") or "").strip()
            self._add_known_face(record, staff_id, "staff", "Nhân viên giám sát", staff_info)

        try:
            self.last_loaded_mtime = self.database_path.stat().st_mtime
        except OSError:
            self.last_loaded_mtime = None

        self.logger.info("Loaded %d known face encodings from %s", len(self.known_faces), self.database_path)

    def _add_known_face(
        self,
        record: Dict[str, Any],
        staff_id: str,
        person_type: str,
        person_label: str,
        info: Dict[str, Any],
    ) -> None:
        encoding = info.get("face_encoding")
        if not encoding:
            return
        try:
            encoding_array = np.array(encoding, dtype=np.float64)
        except (TypeError, ValueError):
            return
        if encoding_array.shape != (128,):
            return

        self.known_faces.append(
            {
                "record": record,
                "staff_id": staff_id,
                "person_type": person_type,
                "person_label": person_label,
                "person_name": str(info.get("name", "") or "").strip(),
                "encoding": encoding_array,
            }
        )

    def detect_single_face(self, frame_bgr: np.ndarray) -> np.ndarray:
        if face_recognition is None:
            raise RuntimeError("Chưa cài thư viện face_recognition.")

        scale = 0.5
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_small = np.ascontiguousarray(rgb_small)

        locations = face_recognition.face_locations(rgb_small, model="hog")
        if not locations:
            raise ValueError("Chưa thấy khuôn mặt rõ trong khung hình.")
        if len(locations) > 1:
            raise ValueError("Đang thấy nhiều khuôn mặt. Vui lòng chỉ để một người trong khung hình.")

        encodings = face_recognition.face_encodings(rgb_small, locations)
        if not encodings:
            raise ValueError("Không trích xuất được mã hóa khuôn mặt.")
        return encodings[0]

    def find_best_match(self, encoding: np.ndarray) -> Optional[FaceMatch]:
        self.reload_if_changed()
        if not self.known_faces:
            return None

        known_encodings = [item["encoding"] for item in self.known_faces]
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])

        if best_distance > self.tolerance:
            return None

        item = self.known_faces[best_index]
        return FaceMatch(
            staff_id=str(item["staff_id"]),
            person_type=str(item["person_type"]),
            person_label=str(item["person_label"]),
            person_name=str(item["person_name"]),
            distance=best_distance,
        )

    def verify_image_data(self, image_data: str) -> Dict[str, Any]:
        if face_recognition is None:
            return {
                "ok": False,
                "error": "Máy chưa cài face_recognition nên chưa thể quét mặt.",
                "code": "missing_dependency",
            }

        self.reload_if_changed()
        if not self.known_faces:
            return {
                "ok": False,
                "error": "Chưa có dữ liệu khuôn mặt nhân viên. Hãy chạy quetmat.py để đăng ký nhân viên giám sát trước.",
                "code": "no_face_database",
            }

        frame = decode_image_data_url(image_data)
        encoding = self.detect_single_face(frame)
        match = self.find_best_match(encoding)
        if match is None:
            return {
                "ok": False,
                "error": "Khuôn mặt không khớp nhân viên giám sát được cấp quyền.",
                "code": "not_matched",
            }

        return {
            "ok": True,
            "match": {
                "staff_id": match.staff_id,
                "patient_id": match.staff_id,
                "person_type": match.person_type,
                "person_label": match.person_label,
                "person_name": match.person_name,
                "distance": round(match.distance, 4),
            },
        }

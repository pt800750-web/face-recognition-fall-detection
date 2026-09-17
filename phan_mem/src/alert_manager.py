from __future__ import annotations

import hashlib
import hmac
import html
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# FALLING is the live-camera warning state. Include it so a real fall can notify
# Telegram immediately instead of waiting for the later FALLEN confirmation.
DANGER_STATUSES = {"FALLING", "FALLEN", "FAINT", "UNCONSCIOUS"}

STATUS_TEXT_VI = {
    "NORMAL": "Bình thường",
    "WALKING_STANDING": "Đi/đứng",
    "SITTING": "Ngồi",
    "BENDING": "Cúi người",
    "LYING_INTENTIONAL": "Nằm chủ động",
    "FALLING": "Đang ngã",
    "FALLEN": "Đã ngã",
    "FAINT": "Ngất",
    "UNCONSCIOUS": "Bất tỉnh",
}


def load_dotenv_file(env_path: Path) -> None:
    """Load .env manually so the project does not need python-dotenv."""
    if not env_path.exists():
        return

    lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines()

    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue

        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            # Let .env be the source of truth when from_env() is called.
            os.environ[key] = value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def split_env_list(name: str) -> list[str]:
    value = os.environ.get(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def status_to_vi(status: str) -> str:
    return STATUS_TEXT_VI.get(str(status), str(status).replace("_", " ").title())


ALERT_SEVERITY_BY_STATUS: Dict[str, Dict[str, Any]] = {
    "FALLING": {"key": "warning", "label_vi": "Cảnh báo", "rank": 1},
    "FALLEN": {"key": "danger", "label_vi": "Nguy hiểm", "rank": 2},
    "FAINT": {"key": "emergency", "label_vi": "Khẩn cấp", "rank": 3},
    "UNCONSCIOUS": {"key": "emergency", "label_vi": "Khẩn cấp", "rank": 3},
}


def alert_severity(status: str) -> Dict[str, Any]:
    fallback = {"key": "warning", "label_vi": "Cảnh báo", "rank": 1}
    return ALERT_SEVERITY_BY_STATUS.get(str(status), fallback)


@dataclass
class AlertRecord:
    alert_id: str
    camera_id: int
    track_id: int
    status: str
    confidence: float
    fall_score: float
    reason: str
    timestamp_sec: float
    created_at: float
    image_path: Optional[Path]
    acknowledged: bool = False
    acknowledged_at: Optional[float] = None
    reminder_count: int = 0
    last_reminded_at: Optional[float] = None


@dataclass
class DangerFilterState:
    first_seen_at: float
    last_seen_at: float
    count: int = 0


class AlertManager:
    def __init__(
        self,
        output_root: Path,
        public_base_url: str = "",
        ack_secret: str = "",
        cooldown_seconds: int = 45,
        escalate_seconds: int = 60,
        enable_telegram: bool = False,
        telegram_bot_token: str = "",
        telegram_chat_ids: Optional[list[str]] = None,
        enable_telegram_reminders: bool = True,
        telegram_reminder_max_count: int = 3,
        telegram_reminder_interval_seconds: int = 60,
        alert_filter_enable: bool = True,
        min_danger_seconds: float = 0.8,
        min_danger_frames: int = 3,
        send_fallen_immediately: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.alert_dir = self.output_root / "alerts"
        self.alert_dir.mkdir(parents=True, exist_ok=True)

        self.public_base_url = str(public_base_url or "").rstrip("/")
        self.ack_secret = str(ack_secret or "change-this-secret")
        self.cooldown_seconds = max(1, int(cooldown_seconds))
        self.escalate_seconds = max(1, int(escalate_seconds))

        self.enable_telegram = bool(enable_telegram)
        self.telegram_bot_token = str(telegram_bot_token or "").strip()
        self.telegram_chat_ids = telegram_chat_ids or []
        self.enable_telegram_reminders = bool(enable_telegram_reminders)
        self.telegram_reminder_max_count = max(0, int(telegram_reminder_max_count))
        self.telegram_reminder_interval_seconds = max(1, int(telegram_reminder_interval_seconds))
        self.alert_filter_enable = bool(alert_filter_enable)
        self.min_danger_seconds = max(0.0, float(min_danger_seconds))
        self.min_danger_frames = max(1, int(min_danger_frames))
        self.send_fallen_immediately = bool(send_fallen_immediately)
        self.filter_stale_seconds = max(3.0, self.min_danger_seconds * 4.0)

        self.logger = logger or logging.getLogger("alert_manager")
        self.lock = threading.RLock()
        self.records: Dict[str, AlertRecord] = {}
        self.active_alerts: Dict[tuple[int, int], str] = {}
        self.last_alert_time: Dict[tuple[int, int, str], float] = {}
        self.danger_filter_states: Dict[tuple[int, int], DangerFilterState] = {}

    @classmethod
    def from_env(
        cls,
        output_root: Path,
        logger: Optional[logging.Logger] = None,
        env_path: Optional[Path] = None,
    ) -> "AlertManager":
        resolved_env_path = env_path or (PROJECT_ROOT / ".env")
        load_dotenv_file(resolved_env_path)

        manager = cls(
            output_root=output_root,
            public_base_url=os.environ.get("PUBLIC_BASE_URL", ""),
            ack_secret=os.environ.get("ALERT_ACK_SECRET", "change-this-secret"),
            cooldown_seconds=env_int("ALERT_COOLDOWN_SECONDS", 45),
            escalate_seconds=env_int("ALERT_ESCALATE_SECONDS", 60),
            enable_telegram=env_bool("ALERT_ENABLE_TELEGRAM", False),
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_ids=split_env_list("TELEGRAM_CHAT_IDS"),
            enable_telegram_reminders=env_bool("ALERT_ENABLE_TELEGRAM_REMINDERS", True),
            telegram_reminder_max_count=env_int("ALERT_REMINDER_MAX_COUNT", 3),
            telegram_reminder_interval_seconds=env_int("ALERT_REMINDER_INTERVAL_SECONDS", 60),
            alert_filter_enable=env_bool("ALERT_FILTER_ENABLE", True),
            min_danger_seconds=env_float("ALERT_MIN_DANGER_SECONDS", 0.8),
            min_danger_frames=env_int("ALERT_MIN_DANGER_FRAMES", 3),
            send_fallen_immediately=env_bool("ALERT_SEND_FALLEN_IMMEDIATELY", True),
            logger=logger,
        )
        manager.logger.info(
            "Telegram alerts enabled=%s chat_ids=%d env=%s filter=%s min_seconds=%.2f min_frames=%d",
            manager.enable_telegram,
            len(manager.telegram_chat_ids),
            resolved_env_path,
            manager.alert_filter_enable,
            manager.min_danger_seconds,
            manager.min_danger_frames,
        )
        return manager

    def handle_detection(
        self,
        camera_id: int,
        track_id: int,
        status: str,
        confidence: float,
        fall_score: float,
        reason: str,
        frame: np.ndarray,
        timestamp_sec: float,
        force: bool = False,
    ) -> Optional[str]:
        status = str(status)

        if status not in DANGER_STATUSES:
            return None

        now = time.time()
        active_key = (int(camera_id), int(track_id))

        if not force:
            with self.lock:
                active_alert_id = self.active_alerts.get(active_key)
                active_record = self.records.get(active_alert_id) if active_alert_id else None
                if active_record is not None and not active_record.acknowledged:
                    previous_status = active_record.status
                    if alert_severity(status)["rank"] >= alert_severity(previous_status)["rank"]:
                        active_record.status = status
                    active_record.confidence = float(confidence)
                    active_record.fall_score = max(active_record.fall_score, float(fall_score))
                    active_record.reason = str(reason or "")
                    active_record.timestamp_sec = float(timestamp_sec)
                    if active_record.status != previous_status:
                        self.logger.info(
                            "Alert %s status escalated from %s to %s without starting a duplicate reminder sequence.",
                            active_record.alert_id,
                            previous_status,
                            active_record.status,
                        )
                    return active_record.alert_id
                self.active_alerts.pop(active_key, None)

        if not force and not self._passes_false_alarm_filter(
            camera_id=int(camera_id),
            track_id=int(track_id),
            status=status,
            now=now,
        ):
            return None

        cooldown_key = (int(camera_id), int(track_id), status)

        if not force:
            with self.lock:
                last_time = self.last_alert_time.get(cooldown_key, 0.0)
                if now - last_time < self.cooldown_seconds:
                    return None
                self.last_alert_time[cooldown_key] = now

        alert_id = uuid.uuid4().hex[:12]

        alert_frame = self._make_alert_frame(
            frame=frame,
            camera_id=int(camera_id),
            track_id=int(track_id),
            status=status,
            confidence=float(confidence),
            fall_score=float(fall_score),
        )

        image_path = self._save_snapshot(
            alert_id=alert_id,
            camera_id=int(camera_id),
            frame=alert_frame,
        )

        record = AlertRecord(
            alert_id=alert_id,
            camera_id=int(camera_id),
            track_id=int(track_id),
            status=status,
            confidence=float(confidence),
            fall_score=float(fall_score),
            reason=str(reason or ""),
            timestamp_sec=float(timestamp_sec),
            created_at=now,
            image_path=image_path,
        )

        with self.lock:
            self.records[alert_id] = record
            self.active_alerts[active_key] = alert_id

        message = self._build_message(record)

        if self.enable_telegram:
            self._queue_telegram_alert(record, message)

        self._schedule_telegram_reminders(record.alert_id)

        return alert_id

    def _passes_false_alarm_filter(self, camera_id: int, track_id: int, status: str, now: float) -> bool:
        if not self.alert_filter_enable:
            return True

        if self.send_fallen_immediately and status in {"FALLEN", "FAINT", "UNCONSCIOUS"}:
            with self.lock:
                self.danger_filter_states.pop((camera_id, track_id), None)
            return True

        key = (camera_id, track_id)

        with self.lock:
            state = self.danger_filter_states.get(key)
            if state is None or now - state.last_seen_at > self.filter_stale_seconds:
                state = DangerFilterState(first_seen_at=now, last_seen_at=now, count=0)
                self.danger_filter_states[key] = state

            state.last_seen_at = now
            state.count += 1

            duration = now - state.first_seen_at
            passed = state.count >= self.min_danger_frames or duration >= self.min_danger_seconds

            if not passed:
                self.logger.info(
                    "Alert candidate filtered camera=%s track=%s status=%s count=%d duration=%.2fs",
                    camera_id,
                    track_id,
                    status,
                    state.count,
                    duration,
                )
                return False

            self.danger_filter_states.pop(key, None)
            self.logger.info(
                "Alert candidate passed filter camera=%s track=%s status=%s count=%d duration=%.2fs",
                camera_id,
                track_id,
                status,
                state.count,
                duration,
            )
            return True

    def acknowledge(self, alert_id: str, token: str) -> bool:
        alert_id = str(alert_id or "").strip()
        token = str(token or "").strip()

        if not alert_id or not token:
            return False

        expected = self._ack_token(alert_id)

        if not hmac.compare_digest(token, expected):
            return False

        with self.lock:
            record = self.records.get(alert_id)
            if record is None:
                return False

            record.acknowledged = True
            record.acknowledged_at = time.time()
            active_key = (record.camera_id, record.track_id)
            if self.active_alerts.get(active_key) == alert_id:
                self.active_alerts.pop(active_key, None)

        self.logger.info("Alert %s acknowledged.", alert_id)
        return True

    def get_record(self, alert_id: str) -> Optional[AlertRecord]:
        with self.lock:
            return self.records.get(alert_id)

    def list_records(self, limit: int = 30) -> list[Dict[str, Any]]:
        with self.lock:
            records = sorted(self.records.values(), key=lambda item: item.created_at, reverse=True)

        items: list[Dict[str, Any]] = []
        for record in records[: max(1, int(limit))]:
            image_available = bool(record.image_path and record.image_path.exists())
            severity = alert_severity(record.status)
            items.append(
                {
                    "alert_id": record.alert_id,
                    "camera_id": record.camera_id,
                    "track_id": record.track_id,
                    "status": record.status,
                    "status_vi": status_to_vi(record.status),
                    "severity": severity["key"],
                    "severity_vi": severity["label_vi"],
                    "severity_rank": severity["rank"],
                    "confidence": record.confidence,
                    "fall_score": record.fall_score,
                    "reason": record.reason,
                    "timestamp_sec": record.timestamp_sec,
                    "created_at": record.created_at,
                    "acknowledged": record.acknowledged,
                    "acknowledged_at": record.acknowledged_at,
                    "reminder_count": record.reminder_count,
                    "last_reminded_at": record.last_reminded_at,
                    "image_available": image_available,
                    "image_url": f"/alert_image?id={record.alert_id}" if image_available else "",
                }
            )

        return items

    def acknowledge_from_query(self, query: Dict[str, list[str]]) -> tuple[bool, str]:
        alert_id = (query.get("id") or [""])[0]
        token = (query.get("token") or [""])[0]

        if not alert_id or not token:
            return False, "Thieu ma canh bao hoac token xac nhan."

        ok = self.acknowledge(alert_id, token)
        if ok:
            return True, "Da xac nhan an toan. Cac tin nhac lai Telegram se dung."

        return False, "Khong the xac nhan canh bao. Lien ket co the da het hieu luc hoac khong dung token."

    def render_ack_html(self, ok: bool, message: str) -> str:
        title = "Da xac nhan" if ok else "Xac nhan that bai"
        color = "#0f9f6e" if ok else "#dc2626"
        safe_message = html.escape(message)
        return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      font-family: Arial, sans-serif;
      background: #f3f6fb;
      color: #102033;
    }}
    main {{
      width: min(520px, calc(100vw - 32px));
      padding: 28px;
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 18px 60px rgba(15, 23, 42, 0.12);
      border-top: 6px solid {color};
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 24px;
      color: {color};
    }}
    p {{
      margin: 0;
      font-size: 16px;
      line-height: 1.55;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{title}</h1>
    <p>{safe_message}</p>
  </main>
</body>
</html>"""

    def _make_alert_frame(
        self,
        frame: np.ndarray,
        camera_id: int,
        track_id: int,
        status: str,
        confidence: float,
        fall_score: float,
    ) -> np.ndarray:
        if frame is None:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            image = frame.copy()

        h, w = image.shape[:2]

        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

        cv2.rectangle(image, (12, 12), (w - 12, h - 12), (0, 0, 255), 3)

        cv2.putText(
            image,
            "CANH BAO NGA / NGAT",
            (36, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.35,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

        status_text = self._status_ascii(status)
        info_1 = f"CAM {camera_id + 1:02d} | ID {track_id} | {status_text}"
        info_2 = f"TIN CAY: {confidence:.2f} | DIEM NGA: {fall_score:.2f}"
        created_time = time.strftime("%Y-%m-%d %H:%M:%S")

        cv2.putText(
            image,
            info_1,
            (36, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            info_2,
            (36, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            created_time,
            (36, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return image

    def _save_snapshot(
        self,
        alert_id: str,
        camera_id: int,
        frame: np.ndarray,
    ) -> Optional[Path]:
        if frame is None:
            return None

        try:
            filename = f"alert_cam{camera_id + 1:02d}_{alert_id}.jpg"
            image_path = self.alert_dir / filename
            cv2.imwrite(str(image_path), frame)
            return image_path
        except Exception as exc:
            self.logger.warning("Could not save alert snapshot: %s", exc)
            return None

    def _build_message(self, record: AlertRecord) -> str:
        created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created_at))
        severity = alert_severity(record.status)

        lines = [
            "🚨 CẢNH BÁO NGUY HIỂM",
            "",
            f"Mức độ: {severity['label_vi']}",
            f"Vị trí: {self._camera_label(record.camera_id)}",
            f"Camera: CAM {record.camera_id + 1:02d}",
            f"Track ID: {record.track_id}",
            f"Trạng thái: {status_to_vi(record.status)}",
            f"Thời gian: {created_time}",
            f"Độ tin cậy: {record.confidence:.2f}",
            f"Điểm ngã: {record.fall_score:.2f}",
            f"Lý do: {record.reason or 'runtime'}",
        ]

        family_url = self._family_url()
        if family_url:
            lines.extend(["", f"Xem trực tiếp: {family_url}"])
        care_phone = self._care_phone()
        if care_phone:
            lines.append(f"Liên hệ nhân viên chăm sóc: {care_phone}")

        return "\n".join(lines)

    def _ack_token(self, alert_id: str) -> str:
        payload = str(alert_id).encode("utf-8")
        secret = self.ack_secret.encode("utf-8")
        return hmac.new(secret, payload, hashlib.sha256).hexdigest()

    def _ack_url(self, alert_id: str) -> str:
        if not self.public_base_url:
            return ""

        token = self._ack_token(alert_id)
        return f"{self.public_base_url}/ack_alert?id={alert_id}&token={token}"

    def _family_url(self) -> str:
        if not self.public_base_url:
            return ""
        if "://web." in self.public_base_url:
            return self.public_base_url.replace("://web.", "://family.", 1)
        return f"{self.public_base_url}/family"

    def _camera_label(self, camera_id: int) -> str:
        return os.environ.get(f"CAMERA_{int(camera_id)}_LABEL", f"CAM {int(camera_id) + 1:02d}").strip()

    def _care_phone(self) -> str:
        for name in ("FAMILY_CARE_PHONE", "CARE_PHONE", "STAFF_CARE_PHONE", "NURSING_CARE_PHONE", "HOTLINE_PHONE"):
            value = str(os.environ.get(name, "") or "").strip()
            if value:
                return value
        return ""

    def _telegram_chat_ids_for_camera(self, camera_id: int) -> list[str]:
        chat_ids = split_env_list(f"CAMERA_{int(camera_id)}_TELEGRAM_CHAT_IDS")
        if chat_ids:
            return chat_ids

        group_camera_ids: set[int] = set()
        for item in split_env_list("ALERT_GROUP_CAMERA_IDS"):
            try:
                group_camera_ids.add(int(item))
            except ValueError:
                self.logger.warning("Invalid camera id in ALERT_GROUP_CAMERA_IDS: %s", item)

        if int(camera_id) in group_camera_ids:
            group_chat_ids = split_env_list("TELEGRAM_GROUP_CHAT_IDS")
            if group_chat_ids:
                return group_chat_ids
            self.logger.warning(
                "Camera %s (%s) is configured for group Telegram routing, but TELEGRAM_GROUP_CHAT_IDS is empty.",
                int(camera_id) + 1,
                self._camera_label(camera_id),
            )
            return []

        if env_bool("ALERT_ROUTE_FALLBACK_TO_GLOBAL", False):
            return self.telegram_chat_ids

        self.logger.warning(
            "No Telegram target configured for camera %s (%s). Set CAMERA_%s_TELEGRAM_CHAT_IDS or TELEGRAM_GROUP_CHAT_IDS.",
            int(camera_id) + 1,
            self._camera_label(camera_id),
            int(camera_id),
        )
        return []

    def _queue_telegram_alert(self, record: AlertRecord, message: str) -> None:
        thread = threading.Thread(
            target=self._send_telegram_alert,
            args=(record, message),
            name=f"telegram-alert-{record.alert_id}",
            daemon=True,
        )
        thread.start()
        self.logger.info("Queued Telegram alert %s for background sending.", record.alert_id)

    def _send_telegram_alert(self, record: AlertRecord, message: str) -> None:
        if not self.telegram_bot_token:
            self.logger.warning("Telegram is enabled but TELEGRAM_BOT_TOKEN is empty.")
            return

        chat_ids = self._telegram_chat_ids_for_camera(record.camera_id)
        if not chat_ids:
            self.logger.warning("Telegram is enabled but no chat id is configured for camera %s.", record.camera_id + 1)
            return

        for chat_id in chat_ids:
            try:
                if record.image_path is not None and record.image_path.exists():
                    self._telegram_send_photo(chat_id, record, message)
                else:
                    self._telegram_send_message(chat_id, record, message)
                self.logger.info("Sent Telegram alert %s to %s.", record.alert_id, chat_id)
            except Exception as exc:
                self.logger.warning("Failed to send Telegram alert to %s: %s", chat_id, exc)

    def _telegram_reply_markup(self, record: AlertRecord) -> Dict[str, Any]:
        buttons = []

        ack_url = self._ack_url(record.alert_id)
        if ack_url:
            buttons.append(
                [
                    {
                        "text": "✅ Đã xác nhận an toàn",
                        "url": ack_url,
                    }
                ]
            )

        family_url = self._family_url()
        if family_url:
            buttons.append(
                [
                    {
                        "text": "📹 Mở camera trực tiếp",
                        "url": family_url,
                    }
                ]
            )

        return {"inline_keyboard": buttons} if buttons else {}

    def _telegram_send_message(self, chat_id: str, record: AlertRecord, message: str) -> None:
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"

        data: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": message,
        }

        reply_markup = self._telegram_reply_markup(record)
        if reply_markup:
            data["reply_markup"] = reply_markup

        response = requests.post(url, json=data, timeout=15)

        if response.status_code >= 400:
            raise RuntimeError(f"Telegram sendMessage failed: {response.status_code} {response.text}")

    def _telegram_send_photo(self, chat_id: str, record: AlertRecord, message: str) -> None:
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto"

        data: Dict[str, Any] = {
            "chat_id": chat_id,
            "caption": message,
        }

        reply_markup = self._telegram_reply_markup(record)
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)

        assert record.image_path is not None

        with record.image_path.open("rb") as file:
            files = {"photo": file}
            response = requests.post(url, data=data, files=files, timeout=30)

        if response.status_code >= 400:
            raise RuntimeError(f"Telegram sendPhoto failed: {response.status_code} {response.text}")

    def _schedule_telegram_reminders(self, alert_id: str) -> None:
        if (
            not self.enable_telegram
            or not self.enable_telegram_reminders
            or self.telegram_reminder_max_count <= 0
        ):
            return

        thread = threading.Thread(
            target=self._send_telegram_reminders_after_delay,
            args=(alert_id,),
            name=f"telegram-reminders-{alert_id}",
            daemon=True,
        )
        thread.start()

    def _send_telegram_reminders_after_delay(self, alert_id: str) -> None:
        time.sleep(self.escalate_seconds)

        for attempt_no in range(1, self.telegram_reminder_max_count + 1):
            with self.lock:
                record = self.records.get(alert_id)
                if record is None:
                    return

                if record.acknowledged:
                    self.logger.info("Alert %s acknowledged; Telegram reminders stopped.", alert_id)
                    return

                reminder_no = record.reminder_count + 1

            sent = self._send_telegram_reminder(record, reminder_no)
            if sent:
                with self.lock:
                    current_record = self.records.get(alert_id)
                    if current_record is record:
                        current_record.reminder_count = reminder_no
                        current_record.last_reminded_at = time.time()
            else:
                self.logger.warning(
                    "Telegram reminder attempt %d for alert %s was not counted because delivery failed.",
                    attempt_no,
                    alert_id,
                )

            if attempt_no < self.telegram_reminder_max_count:
                time.sleep(self.telegram_reminder_interval_seconds)

    def _status_ascii(self, status: str) -> str:
        mapping = {
            "FALLEN": "DA NGA",
            "FAINT": "NGAT",
            "UNCONSCIOUS": "BAT TINH",
            "FALLING": "DANG NGA",
        }
        return mapping.get(str(status), str(status).replace("_", " "))

    def _build_telegram_reminder_message(self, record: AlertRecord, reminder_no: int) -> str:
        status_text = status_to_vi(record.status)
        camera_text = self._camera_label(record.camera_id)
        severity = alert_severity(record.status)

        lines = [
            f"🚨 CẢNH BÁO CHƯA XÁC NHẬN LẦN {reminder_no}",
            "",
            f"Mức độ: {severity['label_vi']}",
            f"Vị trí: {camera_text}",
            f"Camera: CAM {record.camera_id + 1:02d}",
            f"Trạng thái: {status_text}",
            f"Độ tin cậy: {record.confidence:.2f}",
            f"Điểm ngã: {record.fall_score:.2f}",
            "",
            "Cảnh báo trước đó chưa được bấm xác nhận an toàn.",
            "Vui lòng kiểm tra ngay.",
        ]

        family_url = self._family_url()
        if family_url:
            lines.extend(["", f"Xem trực tiếp: {family_url}"])
        care_phone = self._care_phone()
        if care_phone:
            lines.append(f"Liên hệ nhân viên chăm sóc: {care_phone}")

        return "\n".join(lines)

    def _send_telegram_reminder(self, record: AlertRecord, reminder_no: int) -> bool:
        message = self._build_telegram_reminder_message(record, reminder_no)

        chat_ids = self._telegram_chat_ids_for_camera(record.camera_id)
        if not chat_ids:
            self.logger.warning("Telegram reminder skipped; no chat id is configured for camera %s.", record.camera_id + 1)
            return False

        sent_to_any_chat = False
        for chat_id in chat_ids:
            try:
                self._telegram_send_message(chat_id, record, message)
                sent_to_any_chat = True
                self.logger.info(
                    "Sent Telegram reminder %d for alert %s to %s.",
                    reminder_no,
                    record.alert_id,
                    chat_id,
                )
            except Exception as exc:
                self.logger.warning("Failed to send Telegram reminder to %s: %s", chat_id, exc)

        return sent_to_any_chat

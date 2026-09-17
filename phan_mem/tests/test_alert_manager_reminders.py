import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.alert_manager import AlertManager, AlertRecord


class AlertManagerReminderTests(unittest.TestCase):
    def test_one_active_reminder_sequence_per_tracked_person(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            manager = AlertManager(
                output_root=Path(output_dir),
                enable_telegram=False,
                alert_filter_enable=False,
            )
            frame = np.zeros((120, 160, 3), dtype=np.uint8)

            first_alert_id = manager.handle_detection(
                camera_id=0,
                track_id=7,
                status="FALLING",
                confidence=0.80,
                fall_score=0.72,
                reason="fall-started",
                frame=frame,
                timestamp_sec=1.0,
            )
            escalated_alert_id = manager.handle_detection(
                camera_id=0,
                track_id=7,
                status="FALLEN",
                confidence=0.91,
                fall_score=0.88,
                reason="fall-confirmed",
                frame=frame,
                timestamp_sec=2.0,
            )
            other_person_alert_id = manager.handle_detection(
                camera_id=0,
                track_id=8,
                status="FALLEN",
                confidence=0.90,
                fall_score=0.86,
                reason="other-person",
                frame=frame,
                timestamp_sec=2.0,
            )

            self.assertEqual(escalated_alert_id, first_alert_id)
            self.assertNotEqual(other_person_alert_id, first_alert_id)
            self.assertEqual(len(manager.records), 2)
            self.assertEqual(manager.records[first_alert_id].status, "FALLEN")

    def test_failed_delivery_does_not_advance_reminder_number(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            manager = AlertManager(
                output_root=Path(output_dir),
                enable_telegram=True,
                telegram_bot_token="test-token",
                telegram_chat_ids=["test-chat"],
                telegram_reminder_max_count=3,
                escalate_seconds=1,
                telegram_reminder_interval_seconds=1,
            )
            record = AlertRecord(
                alert_id="alert-1",
                camera_id=0,
                track_id=7,
                status="FALLEN",
                confidence=0.91,
                fall_score=0.88,
                reason="test",
                timestamp_sec=0.0,
                created_at=0.0,
                image_path=None,
            )
            manager.records[record.alert_id] = record

            attempted_numbers = []
            outcomes = iter([False, True, True])

            def fake_send(_record: AlertRecord, reminder_no: int) -> bool:
                attempted_numbers.append(reminder_no)
                return next(outcomes)

            manager._send_telegram_reminder = fake_send

            with patch("src.alert_manager.time.sleep", return_value=None):
                manager._send_telegram_reminders_after_delay(record.alert_id)

            self.assertEqual(attempted_numbers, [1, 1, 2])
            self.assertEqual(record.reminder_count, 2)
            self.assertIsNotNone(record.last_reminded_at)

    def test_successful_deliveries_are_numbered_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            manager = AlertManager(
                output_root=Path(output_dir),
                telegram_reminder_max_count=3,
                escalate_seconds=1,
                telegram_reminder_interval_seconds=1,
            )
            record = AlertRecord(
                alert_id="alert-2",
                camera_id=1,
                track_id=8,
                status="FALLING",
                confidence=0.85,
                fall_score=0.80,
                reason="test",
                timestamp_sec=0.0,
                created_at=0.0,
                image_path=None,
            )
            manager.records[record.alert_id] = record

            sent_numbers = []

            def fake_send(_record: AlertRecord, reminder_no: int) -> bool:
                sent_numbers.append(reminder_no)
                return True

            manager._send_telegram_reminder = fake_send

            with patch("src.alert_manager.time.sleep", return_value=None):
                manager._send_telegram_reminders_after_delay(record.alert_id)

            self.assertEqual(sent_numbers, [1, 2, 3])
            self.assertEqual(record.reminder_count, 3)


if __name__ == "__main__":
    unittest.main()

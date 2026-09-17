from pathlib import Path
import time

import cv2
import numpy as np

from src.alert_manager import AlertManager
from src.utils import setup_logging, load_config, ensure_project_dirs, resolve_path


def main() -> None:
    config = load_config("configs/config.yaml")
    ensure_project_dirs(config)
    logger = setup_logging(config, "test_alert")
    manager = AlertManager.from_env(output_root=resolve_path(config["paths"]["outputs_dir"]), logger=logger)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (2, 8, 6)
    cv2.rectangle(frame, (320, 180), (960, 620), (0, 0, 255), 4)
    cv2.putText(frame, "TEST CANH BAO NGA", (360, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "Neu ban nhan duoc anh nay la Telegram OK", (300, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (55, 255, 155), 2, cv2.LINE_AA)

    alert_id = manager.handle_detection(
        camera_id=0,
        track_id=1,
        status="FALLEN",
        confidence=0.91,
        fall_score=0.88,
        reason="test_alert",
        frame=frame,
        timestamp_sec=0.0,
    )
    print(f"Da tao canh bao test: {alert_id}")
    print("Kiem tra Telegram. Neu khong bam xac nhan thi bot se nhac lai theo cau hinh trong .env.")
    if manager.enable_telegram_reminders:
        time.sleep(manager.escalate_seconds + 5)
    else:
        time.sleep(2)


if __name__ == "__main__":
    main()

from pathlib import Path
import hashlib
import hmac
import json
import time

import requests


def load_env(path=".env"):
    env = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


env = load_env()

token = env.get("TELEGRAM_BOT_TOKEN", "")
chat_id = env.get("TELEGRAM_CHAT_IDS", "").split(",")[0].strip()
base_url = env.get("PUBLIC_BASE_URL", "").rstrip("/")
secret = env.get("ALERT_ACK_SECRET", "change-this-secret")

alert_id = "test-alert"
ack_token = hmac.new(
    secret.encode("utf-8"),
    alert_id.encode("utf-8"),
    hashlib.sha256,
).hexdigest()

ack_url = f"{base_url}/ack_alert?id={alert_id}&token={ack_token}"

message = (
    "🚨 CẢNH BÁO NGUY HIỂM\n\n"
    "Camera: CAM 01\n"
    "Trạng thái: Đã ngã\n"
    f"Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    "Độ tin cậy: 0.91\n"
    "Điểm ngã: 0.88\n\n"
    f"Xem trực tiếp: {base_url}"
)

keyboard = {
    "inline_keyboard": [
        [
            {
                "text": "✅ Đã xác nhận an toàn",
                "url": ack_url,
            }
        ],
        [
            {
                "text": "📹 Mở camera trực tiếp",
                "url": base_url,
            }
        ],
    ]
}

response = requests.post(
    f"https://api.telegram.org/bot{token}/sendMessage",
    data={
        "chat_id": chat_id,
        "text": message,
        "reply_markup": json.dumps(keyboard, ensure_ascii=False),
    },
    timeout=20,
)

print(response.status_code)
print(response.text)
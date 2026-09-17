from __future__ import annotations

import argparse
import hmac
import json
import logging
import mimetypes
import os
import secrets
import threading
import time
from http.cookies import SimpleCookie
from collections import defaultdict, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch

from inference import predict_probabilities
from src.clip_writer import EventClipRecorder, EventLogger
from src.fall_logic import FallDecision, FallStateMachine
from src.face_auth import FaceAuthenticator
from src.features import YoloPersonPoseDetector, compute_feature_vector, compute_motion_inside_box
from src.alert_manager import AlertManager, DANGER_STATUSES
from src.models import load_model_from_checkpoint
from src.postprocess import ProbabilitySmoother
from src.tracker import PersonTracker
from src.utils import (
    create_video_writer,
    ensure_project_dirs,
    get_video_metadata,
    is_browser_playable_video,
    load_config,
    make_unique_path,
    parse_source,
    resolve_path,
    safe_timestamp,
    select_device,
    setup_logging,
    transcode_video,
)
from src.visualization import draw_header, draw_track


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg", ".webm"}
WEB_VIDEO_CACHE_DIRNAME = ".web_cache"
WEB_MP4_CODECS = ("mp4v",)
# Do not use VP90/VP80 with OpenCV VideoWriter on Windows; many builds cannot
# encode WebM and print noisy FFmpeg errors. We transcode only to MP4/mp4v.
WEB_WEBM_CODECS: tuple[str, ...] = ()
VIDEO_READY_GRACE_SECONDS = 3.0
AUTH_COOKIE_NAME = "fall_auth_session"
AUTH_SESSION_TTL_SECONDS = 8 * 60 * 60


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

# OpenCV Hershey font does not render Vietnamese accents well, so the label
# drawn directly on camera frames uses uppercase ASCII Vietnamese. The HTML UI
# still uses full accented Vietnamese above.
STATUS_TEXT_FRAME = {
    "NORMAL": "BINH THUONG",
    "WALKING_STANDING": "DI/DUNG",
    "SITTING": "NGOI",
    "BENDING": "CUI NGUOI",
    "LYING_INTENTIONAL": "NAM CHU DONG",
    "FALLING": "DANG NGA",
    "FALLEN": "DA NGA",
    "FAINT": "NGAT",
    "UNCONSCIOUS": "BAT TINH",
}


def status_to_vi(status: str) -> str:
    return STATUS_TEXT_VI.get(str(status), str(status).replace("_", " ").title())


def status_to_frame_text(status: str) -> str:
    return STATUS_TEXT_FRAME.get(str(status), str(status).replace("_", " ").upper())


STATUS_COLOR_BGR = {
    "NORMAL": (80, 220, 80),
    "WALKING_STANDING": (80, 220, 80),
    "SITTING": (255, 190, 60),
    "BENDING": (0, 215, 255),
    "LYING_INTENTIONAL": (255, 80, 200),
    "FALLING": (0, 215, 255),
    "FALLEN": (40, 40, 255),
    "FAINT": (170, 40, 255),
    "UNCONSCIOUS": (170, 40, 255),
}


def status_to_color(status: str) -> tuple[int, int, int]:
    return STATUS_COLOR_BGR.get(str(status), (255, 255, 255))


def draw_track_colored(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    track_id: int,
    status: str,
    confidence: float,
    fall_score: float,
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    color = status_to_color(status)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    label = f"ID {track_id} | {status_to_frame_text(status)} | {confidence:.2f} | S:{fall_score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_y1 = max(0, y1 - text_h - baseline - 8)
    label_y2 = min(height - 1, label_y1 + text_h + baseline + 8)
    label_x2 = min(width - 1, x1 + text_w + 12)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, label, (x1 + 6, label_y2 - baseline - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# Windows/OpenCV fix: avoid H264/OpenH264 because many OpenCV wheels require
# a matching openh264 DLL. This writer uses mp4v first, then falls back to AVI
# codecs that are usually available on Windows.
def create_safe_video_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    safe_fps = float(fps) if fps and float(fps) > 0 else 30.0
    width, height = int(frame_size[0]), int(frame_size[1])

    candidates = [
        (target.with_suffix(".mp4"), "mp4v"),
        (target.with_suffix(".avi"), "XVID"),
        (target.with_suffix(".avi"), "MJPG"),
    ]

    for output_path, codec in candidates:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            safe_fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(f"Cannot create video writer for {target}")


# Override the imported helper used below in this file. Also patch src.clip_writer
# so EventClipRecorder uses the same non-H264 writer when saving event clips.
create_video_writer = create_safe_video_writer
try:
    import src.clip_writer as _clip_writer_module
    _clip_writer_module.create_video_writer = create_safe_video_writer
except Exception:
    pass


LOGIN_HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Đăng nhập giám sát</title>
  <style>
    :root { color-scheme: dark; --bg:#07110f; --panel:#0d1b17; --line:rgba(53,238,190,.22); --text:#e9fff8; --muted:#9bc8ba; --green:#37ff9a; --danger:#ff334e; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; display:grid; place-items:center; font-family:Arial, sans-serif; background:radial-gradient(circle at top left, rgba(55,255,154,.13), transparent 34%), var(--bg); color:var(--text); }
    main { width:min(440px, calc(100vw - 28px)); border:1px solid var(--line); border-radius:10px; background:rgba(13,27,23,.94); padding:26px; box-shadow:0 24px 80px rgba(0,0,0,.42); }
    h1 { margin:0 0 8px; font-size:28px; }
    p { margin:0 0 22px; color:var(--muted); line-height:1.5; }
    label { display:grid; gap:7px; margin:0 0 14px; color:var(--muted); font-size:13px; font-weight:800; text-transform:uppercase; }
    input { min-height:44px; border:1px solid rgba(53,238,190,.22); border-radius:8px; padding:10px 12px; color:var(--text); background:#06100e; font:inherit; }
    button { width:100%; min-height:46px; border:0; border-radius:8px; color:#00120d; background:var(--green); font:inherit; font-weight:900; cursor:pointer; }
    button:disabled { opacity:.62; cursor:wait; }
    .error { min-height:20px; margin-top:14px; color:var(--danger); font-size:14px; font-weight:700; }
  </style>
</head>
<body>
  <main>
    <h1>Đăng nhập ca giám sát</h1>
    <p>Nhân viên cần đăng nhập trước, sau đó xác thực khuôn mặt để vào hệ thống quản lý té ngã.</p>
    <form id="login-form">
      <label>Tài khoản <input id="username" autocomplete="username" required></label>
      <label>Mật khẩu <input id="password" type="password" autocomplete="current-password" required></label>
      <button id="login-button" type="submit">Tiếp tục quét mặt</button>
      <div class="error" id="login-error"></div>
    </form>
  </main>
  <script>
    const form = document.getElementById("login-form");
    const button = document.getElementById("login-button");
    const errorEl = document.getElementById("login-error");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      errorEl.textContent = "";
      button.disabled = true;
      button.textContent = "Đang kiểm tra";
      try {
        const response = await fetch("/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            username: document.getElementById("username").value,
            password: document.getElementById("password").value,
          }),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "Không đăng nhập được.");
        window.location.href = data.next || "/face_auth";
      } catch (error) {
        errorEl.textContent = error.message || "Không đăng nhập được.";
      } finally {
        button.disabled = false;
        button.textContent = "Tiếp tục quét mặt";
      }
    });
  </script>
</body>
</html>"""


FACE_AUTH_HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quét mặt xác thực</title>
  <style>
    :root { color-scheme: dark; --bg:#07110f; --panel:#0d1b17; --line:rgba(53,238,190,.22); --text:#e9fff8; --muted:#9bc8ba; --green:#37ff9a; --cyan:#19d3ff; --danger:#ff334e; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; display:grid; place-items:center; font-family:Arial, sans-serif; background:radial-gradient(circle at top right, rgba(25,211,255,.13), transparent 32%), var(--bg); color:var(--text); }
    main { width:min(760px, calc(100vw - 28px)); border:1px solid var(--line); border-radius:10px; background:rgba(13,27,23,.94); padding:22px; box-shadow:0 24px 80px rgba(0,0,0,.42); }
    h1 { margin:0 0 8px; font-size:28px; }
    p { margin:0 0 18px; color:var(--muted); line-height:1.5; }
    .camera { position:relative; overflow:hidden; border:1px solid rgba(25,211,255,.25); border-radius:8px; background:#020807; aspect-ratio:16/9; }
    video { width:100%; height:100%; object-fit:cover; transform:scaleX(-1); display:block; }
    .scan-line { position:absolute; left:5%; right:5%; height:3px; top:22%; background:var(--cyan); box-shadow:0 0 18px var(--cyan); animation:scan 1.6s linear infinite; opacity:.9; }
    @keyframes scan { from { top:18%; } to { top:82%; } }
    .actions { display:grid; grid-template-columns:1fr auto; gap:10px; margin-top:14px; }
    button, a { min-height:44px; border-radius:8px; font:inherit; font-weight:900; text-align:center; text-decoration:none; cursor:pointer; }
    button { border:0; color:#00120d; background:var(--green); }
    a { display:grid; place-items:center; border:1px solid rgba(53,238,190,.22); color:var(--text); background:rgba(0,0,0,.32); padding:0 14px; }
    button:disabled { opacity:.62; cursor:wait; }
    .status { min-height:22px; margin-top:12px; color:var(--muted); font-weight:800; }
    .status.error { color:var(--danger); }
    .status.ok { color:var(--green); }
    canvas { display:none; }
  </style>
</head>
<body>
  <main>
    <h1>Quét mặt nhân viên</h1>
    <p>Đặt khuôn mặt rõ trong khung hình. Hệ thống sẽ so khớp với dữ liệu từ file quét mặt trước khi mở trang quản lý.</p>
    <div class="camera">
      <video id="video" autoplay playsinline muted></video>
      <div class="scan-line"></div>
    </div>
    <div class="actions">
      <button id="scan-button" type="button">Quét và vào hệ thống</button>
      <a href="/auth/logout">Đăng xuất</a>
    </div>
    <div class="status" id="face-status">Đang mở camera trình duyệt...</div>
    <canvas id="canvas" width="960" height="540"></canvas>
  </main>
  <script>
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const button = document.getElementById("scan-button");
    const statusEl = document.getElementById("face-status");
    let stream = null;

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540, facingMode: "user" }, audio: false });
        video.srcObject = stream;
        statusEl.textContent = "Camera đã sẵn sàng. Bấm quét để xác thực.";
      } catch (error) {
        statusEl.textContent = "Không mở được camera trình duyệt. Hãy cấp quyền camera rồi tải lại trang.";
        statusEl.className = "status error";
        button.disabled = true;
      }
    }

    function captureImage() {
      const ctx = canvas.getContext("2d");
      const width = video.videoWidth || 960;
      const height = video.videoHeight || 540;
      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(video, 0, 0, width, height);
      return canvas.toDataURL("image/jpeg", 0.86);
    }

    button.addEventListener("click", async () => {
      statusEl.className = "status";
      statusEl.textContent = "Đang so khớp khuôn mặt...";
      button.disabled = true;
      try {
        const response = await fetch("/auth/face", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: captureImage() }),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "Xác thực khuôn mặt thất bại.");
        statusEl.className = "status ok";
        statusEl.textContent = `Đã xác thực ${data.match?.person_name || "nhân viên"}. Đang mở hệ thống...`;
        window.setTimeout(() => { window.location.href = data.next || "/dashboard"; }, 550);
      } catch (error) {
        statusEl.className = "status error";
        statusEl.textContent = error.message || "Xác thực khuôn mặt thất bại.";
        button.disabled = false;
      }
    });

    startCamera();
  </script>
</body>
</html>"""
FAMILY_HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trang người thân | Theo dõi an toàn</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #020403;
      --surface: #07110d;
      --surface-soft: #0b1a14;
      --surface-blue: rgba(25, 211, 255, 0.10);
      --text: #eefcf7;
      --muted: #8fb4ab;
      --line: rgba(55, 255, 154, 0.22);
      --line-strong: rgba(25, 211, 255, 0.42);
      --green: #37ff9a;
      --blue: #19d3ff;
      --amber: #ffcf4a;
      --red: #ff293d;
      --shadow: rgba(0, 0, 0, 0.58);
    }

    * { box-sizing: border-box; letter-spacing: 0; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        linear-gradient(rgba(55, 255, 154, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(25, 211, 255, 0.03) 1px, transparent 1px),
        var(--bg);
      background-size: 44px 44px;
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }
    button, input { font: inherit; }

    .family-page {
      width: min(1440px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 34px;
    }

    .family-header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      margin-bottom: 18px;
    }

    .eyebrow {
      margin: 0 0 8px;
      color: var(--green);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }
    h1, h2, h3, p { margin-top: 0; }
    h1 {
      margin-bottom: 10px;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.08;
    }
    .subtitle {
      max-width: 760px;
      margin-bottom: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.55;
    }

    .live-summary {
      min-width: 270px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: var(--surface);
      box-shadow: var(--shadow);
    }
    .live-summary span {
      display: block;
      margin-bottom: 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .live-summary strong {
      display: flex;
      align-items: center;
      gap: 9px;
      font-size: 18px;
    }
    .call-care-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 38px;
      margin-top: 12px;
      border: 1px solid var(--line-strong);
      border-radius: 7px;
      padding: 9px 12px;
      color: var(--text);
      background: rgba(25, 211, 255, 0.12);
      font-size: 13px;
      font-weight: 900;
      text-decoration: none;
    }
    .call-care-button:hover {
      border-color: var(--blue);
      background: rgba(25, 211, 255, 0.18);
    }
    .call-care-button[hidden] {
      display: none;
    }
    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 0 5px rgba(22, 132, 91, 0.12);
      flex: 0 0 auto;
    }
    .live-summary.danger .status-dot {
      background: var(--red);
      box-shadow: 0 0 0 5px rgba(217, 48, 79, 0.12);
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .summary-item, .panel, .camera-tile {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
    }
    .summary-item { padding: 14px; }
    .summary-label {
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .summary-value {
      margin-bottom: 0;
      font-size: 28px;
      font-weight: 900;
      line-height: 1;
    }
    .summary-value.good { color: var(--green); }
    .summary-value.warn { color: var(--amber); }
    .summary-value.danger { color: var(--red); }

    .family-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 18px;
      align-items: start;
    }
    .camera-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .camera-tile { overflow: hidden; }
    .camera-head, .camera-foot {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 11px 12px;
    }
    .camera-name { font-weight: 900; }
    .camera-state {
      border-radius: 999px;
      padding: 5px 9px;
      color: var(--green);
      background: rgba(22, 132, 91, 0.10);
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
    }
    .camera-tile.warning .camera-state {
      color: #6c4600;
      background: rgba(183, 121, 31, 0.16);
    }
    .camera-tile.danger .camera-state {
      color: var(--red);
      background: rgba(217, 48, 79, 0.12);
    }
    .camera-view {
      position: relative;
      aspect-ratio: 16 / 9;
      background: #0b1117;
    }
    .camera-view img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #0b1117;
    }
    .camera-overlay {
      position: absolute;
      left: 10px;
      bottom: 10px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .camera-chip {
      border: 1px solid rgba(255, 255, 255, 0.20);
      border-radius: 6px;
      padding: 6px 8px;
      color: #fff;
      background: rgba(0, 0, 0, 0.56);
      font-size: 12px;
      font-weight: 800;
    }
    .camera-foot {
      color: var(--muted);
      font-size: 13px;
    }

    .family-side {
      display: grid;
      gap: 14px;
    }
    .panel { padding: 15px; }
    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
    }
    .section-title h2 {
      margin-bottom: 0;
      font-size: 18px;
      line-height: 1.2;
    }
    .section-title span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .alert-list, .video-list, .chat-messages {
      display: grid;
      gap: 10px;
    }
    .alert-item {
      display: grid;
      grid-template-columns: 70px minmax(0, 1fr);
      gap: 10px;
      border: 1px solid var(--line);
      border-left: 4px solid var(--amber);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }
    .alert-item.danger, .alert-item.emergency { border-left-color: var(--red); }
    .alert-thumb, .alert-placeholder {
      width: 70px;
      aspect-ratio: 1 / 1;
      border: 1px solid var(--line);
      border-radius: 7px;
      object-fit: cover;
      background: var(--surface-soft);
    }
    .alert-placeholder {
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 11px;
      font-weight: 900;
    }
    .alert-main { min-width: 0; }
    .alert-title {
      margin-bottom: 5px;
      font-weight: 900;
    }
    .alert-meta {
      margin-bottom: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }

    .video-list button, .question-chip, .chat-send {
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      color: var(--text);
      background: #fff;
      cursor: pointer;
    }
    .video-list button {
      display: grid;
      gap: 4px;
      width: 100%;
      padding: 10px;
      text-align: left;
    }
    .video-list button:hover, .question-chip:hover {
      border-color: var(--blue);
      background: var(--surface-blue);
    }
    .video-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 900;
    }
    .video-meta {
      color: var(--muted);
      font-size: 12px;
    }
    .replay-player {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 10px;
      background: #0b1117;
      object-fit: contain;
    }

    .question-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }
    .question-chip {
      padding: 8px 10px;
      color: var(--blue);
      font-size: 13px;
      font-weight: 800;
    }
    .chat-messages {
      max-height: 280px;
      overflow: auto;
      padding-right: 2px;
    }
    .chat-row {
      border-radius: 8px;
      padding: 10px 11px;
      background: var(--surface-soft);
      color: var(--text);
      font-size: 14px;
      line-height: 1.45;
    }
    .chat-row.user { background: var(--surface-blue); }
    .chat-input-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      margin-top: 10px;
    }
    .chat-input-row input {
      min-width: 0;
      min-height: 42px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      padding: 0 11px;
      color: var(--text);
      background: #fff;
      outline: none;
    }
    .chat-input-row input:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(47, 111, 237, 0.10);
    }
    .chat-send {
      min-height: 42px;
      padding: 0 16px;
      color: #fff;
      background: var(--blue);
      border-color: var(--blue);
      font-weight: 900;
    }
    .empty {
      margin-bottom: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }
    [hidden] { display: none !important; }

    @media (max-width: 1120px) {
      .family-layout { grid-template-columns: 1fr; }
      .family-side { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 820px) {
      .family-page {
        width: min(100% - 20px, 760px);
        padding-top: 16px;
      }
      .family-header, .summary-grid, .camera-grid, .family-side {
        grid-template-columns: 1fr;
      }
      .live-summary { min-width: 0; }
      .chat-input-row { grid-template-columns: 1fr; }
    }


    .live-summary,
    .summary-item,
    .panel,
    .camera-tile {
      background: rgba(5, 15, 12, 0.88);
      box-shadow: 0 18px 52px var(--shadow), inset 0 0 22px rgba(25, 211, 255, 0.035);
    }

    .alert-item,
    .video-list button,
    .question-chip,
    .chat-send,
    .chat-input-row input {
      color: var(--text);
      background: rgba(0, 0, 0, 0.34);
    }

    .video-list button:hover,
    .question-chip:hover {
      border-color: var(--green);
      background: rgba(25, 211, 255, 0.10);
    }

    .chat-fab {
      position: fixed;
      right: 22px;
      bottom: 22px;
      z-index: 40;
      width: 58px;
      height: 58px;
      border: 1px solid rgba(55, 255, 154, 0.55);
      border-radius: 50%;
      color: #00120d;
      background: var(--green);
      box-shadow: 0 18px 42px rgba(0, 0, 0, 0.42), 0 0 22px rgba(55, 255, 154, 0.28);
      font-size: 24px;
      font-weight: 900;
      cursor: pointer;
    }

    .family-side .panel:nth-child(3) {
      position: fixed;
      right: 22px;
      bottom: 92px;
      z-index: 41;
      width: min(390px, calc(100vw - 28px));
      max-height: min(620px, calc(100vh - 120px));
      overflow: auto;
      background: rgba(5, 15, 12, 0.96);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.55);
    }

    .family-side .panel:nth-child(3)[hidden] {
      display: none !important;
    }

    .chat-widget-close {
      width: 34px;
      height: 34px;
      border: 1px solid rgba(25, 211, 255, 0.28);
      border-radius: 8px;
      color: var(--text);
      background: rgba(0, 0, 0, 0.34);
      font-size: 22px;
      line-height: 1;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <main class="family-page">
    <header class="family-header">
      <div>
        <p class="eyebrow">Cổng thông tin người thân</p>
        <h1>Theo dõi an toàn của người thân</h1>
        <p class="subtitle">Trang này chỉ hiển thị thông tin cần thiết cho gia đình: camera trực tiếp, cảnh báo gần đây, video sự cố và phần hỏi nhanh tình trạng hiện tại.</p>
      </div>
      <div class="live-summary" id="live-summary">
        <span>Tình trạng hiện tại</span>
        <strong><i class="status-dot"></i><em id="overall-status">Đang tải dữ liệu</em></strong>
        <a class="call-care-button" id="call-care-button" href="#" hidden>Gọi nhân viên chăm sóc</a>
      </div>
    </header>

    <section class="summary-grid" aria-label="Tổng quan tình trạng">
      <article class="summary-item"><p class="summary-label">Camera đang hoạt động</p><p class="summary-value good" id="active-cameras">0</p></article>
      <article class="summary-item"><p class="summary-label">Người trong khung hình</p><p class="summary-value" id="people-count">0</p></article>
      <article class="summary-item"><p class="summary-label">Cảnh báo hiện tại</p><p class="summary-value good" id="danger-count">0</p></article>
      <article class="summary-item"><p class="summary-label">Cập nhật lần cuối</p><p class="summary-value" id="last-updated">--:--</p></article>
    </section>

    <section class="family-layout">
      <div class="camera-grid" id="camera-grid">
        <article class="camera-tile" data-camera="0"><div class="camera-head"><span class="camera-name">Camera 1</span><span class="camera-state">Đang chờ</span></div><div class="camera-view"><img data-stream="/video_feed?camera=0" src="" alt="Camera 1"><div class="camera-overlay"><span class="camera-chip people-chip">0 người</span><span class="camera-chip fps-chip">0.0 FPS</span></div></div><div class="camera-foot"><span class="source-chip">Nguồn: --</span><span class="time-chip">t=0.0s</span></div></article>
        <article class="camera-tile" data-camera="1"><div class="camera-head"><span class="camera-name">Camera 2</span><span class="camera-state">Đang chờ</span></div><div class="camera-view"><img data-stream="/video_feed?camera=1" src="" alt="Camera 2"><div class="camera-overlay"><span class="camera-chip people-chip">0 người</span><span class="camera-chip fps-chip">0.0 FPS</span></div></div><div class="camera-foot"><span class="source-chip">Nguồn: --</span><span class="time-chip">t=0.0s</span></div></article>
        <article class="camera-tile" data-camera="2"><div class="camera-head"><span class="camera-name">Camera 3</span><span class="camera-state">Đang chờ</span></div><div class="camera-view"><img data-stream="/video_feed?camera=2" src="" alt="Camera 3"><div class="camera-overlay"><span class="camera-chip people-chip">0 người</span><span class="camera-chip fps-chip">0.0 FPS</span></div></div><div class="camera-foot"><span class="source-chip">Nguồn: --</span><span class="time-chip">t=0.0s</span></div></article>
        <article class="camera-tile" data-camera="3"><div class="camera-head"><span class="camera-name">Camera 4</span><span class="camera-state">Đang chờ</span></div><div class="camera-view"><img data-stream="/video_feed?camera=3" src="" alt="Camera 4"><div class="camera-overlay"><span class="camera-chip people-chip">0 người</span><span class="camera-chip fps-chip">0.0 FPS</span></div></div><div class="camera-foot"><span class="source-chip">Nguồn: --</span><span class="time-chip">t=0.0s</span></div></article>
      </div>

      <aside class="family-side">
        <section class="panel"><div class="section-title"><h2>Cảnh báo gần đây</h2><span id="alert-count">0 sự kiện</span></div><div class="alert-list" id="alert-list"><p class="empty">Đang tải cảnh báo...</p></div></section>
        <section class="panel"><div class="section-title"><h2>Video sự cố</h2><span id="video-count">0 video</span></div><img class="replay-player" id="replay-player" alt="Video sự cố" hidden><div class="video-list" id="video-list"><p class="empty">Đang tải video...</p></div></section>
        <section class="panel">
          <div class="section-title"><h2>Hỏi nhanh</h2><button class="chat-widget-close" type="button" id="chat-close" aria-label="Đóng chatbox">×</button></div>
          <div class="question-row">
            <button class="question-chip" type="button" data-question="Tình trạng hiện tại?">Tình trạng hiện tại?</button>
            <button class="question-chip" type="button" data-question="Cảnh báo xảy ra lúc nào?">Cảnh báo lúc nào?</button>
            <button class="question-chip" type="button" data-question="Ngã ở phòng nào?">Ngã ở phòng nào?</button>
            <button class="question-chip" type="button" data-question="Có video sự cố không?">Có video sự cố?</button>
            <button class="question-chip" type="button" data-question="Nhân viên đã xác nhận chưa?">Đã xác nhận chưa?</button>
            <button class="question-chip" type="button" data-question="Tôi cần liên hệ nhân viên chăm sóc">Liên hệ nhân viên</button>
          </div>
          <div class="chat-messages" id="chat-box"><div class="chat-row">Bạn có thể chọn câu hỏi nhanh hoặc nhập câu hỏi về thời gian cảnh báo, phòng/camera, trạng thái, video sự cố và số điện thoại nhân viên chăm sóc.</div></div>
          <div class="chat-input-row"><input id="chat-input" placeholder="Nhập câu hỏi của bạn"><button class="chat-send" type="button" id="chat-send">Gửi</button></div>
        </section>
      </aside>
    </section>
    <button class="chat-fab" id="chat-fab" type="button" aria-label="Mở chatbox">💬</button>
  </main>

  <script>
    const dangerStates = new Set(["FALLEN", "FAINT", "UNCONSCIOUS"]);
    const warningStates = new Set(["FALLING"]);
    const statusLabelMap = {
      NORMAL: "Bình thường",
      WALKING_STANDING: "Đi/đứng",
      SITTING: "Ngồi",
      BENDING: "Cúi người",
      LYING_INTENTIONAL: "Nằm chủ động",
      FALLING: "Đang ngã",
      FALLEN: "Đã ngã",
      FAINT: "Ngất",
      UNCONSCIOUS: "Bất tỉnh",
    };

    let latestStatus = null;
    let latestAlerts = [];
    let latestVideos = [];
    const familyToken = new URLSearchParams(window.location.search).get("token") || "";
    let carePhone = "09xx xxx xxx";
    let cameraLabelMap = {
      0: "Phòng 101",
      1: "Phòng 102",
      2: "Khu sinh hoạt chung",
      3: "Hành lang / khu sinh hoạt",
    };
    const intentKeywords = {
      alertTime: ["luc nao", "may gio", "khi nao", "thoi gian", "hoi may gio", "bao nhieu gio", "xay ra luc nao", "vua roi luc nao"],
      fallStatus: ["nga", "te", "bi nga", "bi te", "canh bao", "su co", "ngat", "bat tinh"],
      location: ["o dau", "phong nao", "camera nao", "vi tri", "khu nao", "cho nao", "noi nao"],
      video: ["video", "clip", "xem lai", "coi lai", "doan ghi", "ghi hinh", "luc nga", "luc te"],
      currentStatus: ["hien tai", "bay gio", "trang thai", "co on khong", "binh thuong khong", "dang the nao", "co an toan khong"],
      acknowledgement: ["xac nhan", "da xac nhan", "chua xac nhan", "xu ly", "kiem tra", "da den chua", "co ai den", "da biet chua", "da ho tro chua"],
      staffContact: ["lien he", "nhan vien", "dieu duong", "goi ai", "goi cho ai", "so dien thoai", "hotline", "cham soc", "nguoi phu trach"],
      healthRisk: ["bi thuong", "nguy hiem", "cap cuu", "co sao khong", "dau khong", "chay mau", "benh vien"],
      outOfScopeCare: ["an com", "uong thuoc", "thuoc", "ngu chua", "huyet ap", "doi phong", "tham nuoi", "an uong"],
    };

    const liveSummaryEl = document.getElementById("live-summary");
    const overallStatusEl = document.getElementById("overall-status");
    const activeCamerasEl = document.getElementById("active-cameras");
    const peopleCountEl = document.getElementById("people-count");
    const dangerCountEl = document.getElementById("danger-count");
    const lastUpdatedEl = document.getElementById("last-updated");
    const alertCountEl = document.getElementById("alert-count");
    const alertListEl = document.getElementById("alert-list");
    const videoCountEl = document.getElementById("video-count");
    const videoListEl = document.getElementById("video-list");
    const replayPlayerEl = document.getElementById("replay-player");
    const callCareButtonEl = document.getElementById("call-care-button");
    const chatBoxEl = document.getElementById("chat-box");
    const chatInputEl = document.getElementById("chat-input");
    const chatSendEl = document.getElementById("chat-send");
    const chatFabEl = document.getElementById("chat-fab");
    const chatCloseEl = document.getElementById("chat-close");
    const chatWidgetEl = chatBoxEl.closest(".panel");

    function withToken(url) {
      if (!familyToken) return url;
      const separator = url.includes("?") ? "&" : "?";
      return `${url}${separator}token=${encodeURIComponent(familyToken)}`;
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function statusLabel(status) {
      const key = String(status || "NORMAL");
      return statusLabelMap[key] || key.replaceAll("_", " ").toLowerCase();
    }

    function severityText(severity) {
      const key = String(severity || "warning");
      if (key === "emergency") return "Khẩn cấp";
      if (key === "danger") return "Nguy hiểm";
      return "Cảnh báo";
    }

    function formatTime(seconds) {
      const value = Number(seconds || 0);
      if (!value) return "--";
      return new Date(value * 1000).toLocaleString("vi-VN", { hour12: false });
    }

    function formatShortTime(date = new Date()) {
      return date.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    }

    function normalizeText(value) {
      return String(value || "")
        .toLowerCase()
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .replace(/đ/g, "d")
        .replace(/[^a-z0-9\s]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
    }

    function hasAny(text, keywords) {
      return keywords.some((keyword) => text.includes(keyword));
    }

    function cameraLabel(cameraId) {
      return cameraLabelMap[cameraId] || `Camera ${cameraId + 1}`;
    }

    function phoneHref(phone) {
      const digits = String(phone || "").replace(/[^\d+]/g, "");
      return digits && !digits.includes("x") ? `tel:${digits}` : "";
    }

    function setupCareContact() {
      if (!callCareButtonEl) return;
      const href = phoneHref(carePhone);
      callCareButtonEl.textContent = href ? `Gọi nhân viên: ${carePhone}` : `Số nhân viên: ${carePhone}`;
      callCareButtonEl.hidden = false;
      if (href) {
        callCareButtonEl.href = href;
      } else {
        callCareButtonEl.removeAttribute("href");
      }
    }

    function renderCameraLabels() {
      document.querySelectorAll(".camera-tile").forEach((tile) => {
        const cameraId = Number(tile.dataset.camera || 0);
        const label = cameraLabel(cameraId);
        const nameEl = tile.querySelector(".camera-name");
        const imageEl = tile.querySelector("img");
        if (nameEl) nameEl.textContent = label;
        if (imageEl) imageEl.alt = label;
      });
    }

    async function loadFamilyConfig() {
      try {
        const config = await fetchJson("/family_config");
        if (config.care_phone) carePhone = String(config.care_phone);
        if (config.camera_labels && typeof config.camera_labels === "object") {
          cameraLabelMap = Object.fromEntries(Object.entries(config.camera_labels).map(([key, value]) => [Number(key), String(value)]));
        }
      } catch (error) {
        console.warn("Không tải được cấu hình người thân:", error);
      }
      setupCareContact();
      renderCameraLabels();
    }

    function setupCameraStreams() {
      document.querySelectorAll("img[data-stream]").forEach((image) => {
        image.src = withToken(image.dataset.stream || "");
      });
    }

    function cameraTile(cameraId) {
      return document.querySelector(`[data-camera="${cameraId}"]`);
    }

    async function fetchJson(url) {
      const response = await fetch(withToken(url), { cache: "no-store" });
      const data = await response.json();
      if (!response.ok || data.ok === false) throw new Error(data.error || "Không tải được dữ liệu.");
      return data;
    }

    async function loadFamilyData() {
      try {
        const [statusData, alertData, videoData] = await Promise.all([fetchJson("/status"), fetchJson("/alerts"), fetchJson("/videos")]);
        latestStatus = statusData;
        latestAlerts = Array.isArray(alertData.alerts) ? alertData.alerts : [];
        latestVideos = (videoData.videos || []).filter((video) => {
          const path = String(video.path || "").toLowerCase();
          return path.includes("fall") || path.includes("faint") || path.includes("unconscious");
        });
        renderStatus(statusData);
        renderAlerts();
        renderVideos();
      } catch (error) {
        overallStatusEl.textContent = "Không tải được dữ liệu";
        alertListEl.innerHTML = `<p class="empty">${escapeHtml(error.message || "Không tải được dữ liệu.")}</p>`;
      }
    }

    function renderStatus(data) {
      const cameras = data.cameras || [];
      const activeCount = Number(data.active_cameras || cameras.filter((camera) => camera.running).length);
      const totalPeople = cameras.reduce((total, camera) => total + Number(camera.people || 0), 0);
      const dangerCount = cameras.filter((camera) => camera.fall_alert || camera.faint_alert).length;
      liveSummaryEl.classList.toggle("danger", dangerCount > 0);
      overallStatusEl.textContent = dangerCount > 0 ? `Có cảnh báo trên ${dangerCount} camera` : activeCount > 0 ? "Đang theo dõi ổn định" : "Đang chờ nguồn camera";
      activeCamerasEl.textContent = activeCount;
      peopleCountEl.textContent = totalPeople;
      dangerCountEl.textContent = dangerCount;
      dangerCountEl.className = `summary-value ${dangerCount > 0 ? "danger" : "good"}`;
      lastUpdatedEl.textContent = formatShortTime();
      cameras.forEach(renderCamera);
    }

    function renderCamera(camera) {
      const cameraId = Number(camera.camera_id || 0);
      const tile = cameraTile(cameraId);
      if (!tile) return;
      const label = camera.camera_label || cameraLabel(cameraId);
      const running = Boolean(camera.running);
      const fallAlert = Boolean(camera.fall_alert);
      const faintAlert = Boolean(camera.faint_alert);
      const tracks = camera.tracks || [];
      const warning = !fallAlert && tracks.some((track) => warningStates.has(track.status));
      const dangerTrack = tracks.find((track) => dangerStates.has(track.status));
      const warningTrack = tracks.find((track) => warningStates.has(track.status));
      const dominantStatus = dangerTrack?.status || warningTrack?.status || tracks[0]?.status || "NORMAL";
      tile.classList.toggle("warning", warning);
      tile.classList.toggle("danger", fallAlert || faintAlert);
      tile.querySelector(".camera-name").textContent = label;
      tile.querySelector(".camera-state").textContent = faintAlert ? "Ngất" : fallAlert ? "Cảnh báo" : warning ? "Đang ngã" : running ? statusLabel(dominantStatus) : "Đang chờ";
      tile.querySelector(".people-chip").textContent = `${Number(camera.people || 0)} người`;
      tile.querySelector(".fps-chip").textContent = `${Number(camera.processing_fps || 0).toFixed(1)} FPS`;
      tile.querySelector(".source-chip").textContent = `Nguồn: ${camera.source || "--"}`;
      tile.querySelector(".time-chip").textContent = `t=${Number(camera.timestamp_sec || 0).toFixed(1)}s`;
    }

    function renderAlerts() {
      alertCountEl.textContent = `${latestAlerts.length} sự kiện`;
      if (!latestAlerts.length) {
        alertListEl.innerHTML = '<p class="empty">Chưa có cảnh báo nào được ghi nhận.</p>';
        return;
      }
      alertListEl.innerHTML = latestAlerts.slice(0, 6).map((item) => {
        const severity = String(item.severity || "warning");
        const statusText = item.status_vi || statusLabel(item.status);
        const cameraId = Number(item.camera_id || 0);
        const label = item.camera_label || cameraLabel(cameraId);
        const thumb = item.image_available
          ? `<a href="${escapeHtml(item.image_url)}" target="_blank" rel="noopener"><img class="alert-thumb" src="${escapeHtml(item.image_url)}" alt="Ảnh cảnh báo"></a>`
          : '<div class="alert-placeholder">Không ảnh</div>';
        return `<article class="alert-item ${escapeHtml(severity)}">${thumb}<div class="alert-main"><p class="alert-title">${escapeHtml(label)} - ${escapeHtml(statusText)}</p><p class="alert-meta">Camera ${cameraId + 1} · ${escapeHtml(severityText(severity))} · ${escapeHtml(formatTime(item.created_at))}</p><p class="alert-meta">${item.acknowledged ? "Đã xác nhận an toàn" : "Đang chờ nhân viên kiểm tra"}</p></div></article>`;
      }).join("");
    }

    function renderVideos() {
      videoCountEl.textContent = `${latestVideos.length} video`;
      if (!latestVideos.length) {
        videoListEl.innerHTML = '<p class="empty">Chưa có video sự cố được lưu.</p>';
        replayPlayerEl.hidden = true;
        replayPlayerEl.removeAttribute("src");
        return;
      }
      videoListEl.innerHTML = latestVideos.slice(0, 8).map((video, index) => `<button type="button" onclick="playFamilyVideo(${index})"><span class="video-name">${escapeHtml(video.name || video.path)}</span><span class="video-meta">${escapeHtml(formatTime(video.modified_at))}</span></button>`).join("");
    }

    function playFamilyVideo(index) {
      const video = latestVideos[index];
      if (!video) return;
      replayPlayerEl.hidden = false;
      replayPlayerEl.src = withToken(`/replay_video?path=${encodeURIComponent(video.path)}`);
    }
    function latestAlertSummary() {
      if (!latestAlerts.length) {
        return "Hiện hệ thống chưa ghi nhận cảnh báo té ngã nào. Trang vẫn đang theo dõi các camera và sẽ cập nhật tự động nếu có sự cố mới.";
      }

      const latest = latestAlerts[0];
      const cameraId = Number(latest.camera_id || 0);
      const label = latest.camera_label || cameraLabel(cameraId);
      const timeText = formatTime(latest.created_at);
      const statusText = latest.status_vi || statusLabel(latest.status);
      const ackText = latest.acknowledged ? "Cảnh báo này đã được xác nhận an toàn." : "Cảnh báo này hiện đang chờ nhân viên kiểm tra và xác nhận an toàn.";

      return `Cảnh báo gần nhất được ghi nhận lúc ${timeText} tại ${label} - Camera ${cameraId + 1}. Trạng thái hệ thống là ${statusText}. ${ackText} Bạn có thể xem camera trực tiếp và mục video sự cố trên trang này.`;
    }

    function currentStatusSummary() {
      const cameras = latestStatus?.cameras || [];
      const activeCount = Number(latestStatus?.active_cameras || cameras.filter((camera) => camera.running).length || 0);
      const dangerCamera = cameras.find((camera) => camera.fall_alert || camera.faint_alert || (camera.tracks || []).some((track) => dangerStates.has(track.status)));

      if (dangerCamera || latestStatus?.fall_alert || latestStatus?.faint_alert) {
        return latestAlertSummary();
      }

      if (activeCount > 0) {
        return `Hiện hệ thống đang theo dõi ${activeCount} camera và chưa ghi nhận cảnh báo nguy hiểm mới. Nếu bạn muốn kiểm tra kỹ hơn, hãy xem trực tiếp camera phòng hoặc khu vực tương ứng trên trang.`;
      }

      return "Hiện hệ thống đang chờ nguồn camera, bạn vui lòng thử tải lại trang sau ít phút.";
    }

    function acknowledgementSummary() {
      if (!latestAlerts.length) {
        return "Hiện chưa có cảnh báo nào cần xác nhận. Nếu có cảnh báo mới, trạng thái xác nhận sẽ được hiển thị trong mục Cảnh báo gần đây.";
      }

      const latest = latestAlerts[0];
      const cameraId = Number(latest.camera_id || 0);
      const label = latest.camera_label || cameraLabel(cameraId);
      const timeText = formatTime(latest.created_at);
      const ackText = latest.acknowledged ? "Đã xác nhận an toàn" : "Đang chờ nhân viên kiểm tra và xác nhận an toàn";

      return `Trạng thái xác nhận mới nhất: ${ackText}. Cảnh báo này ở ${label} - Camera ${cameraId + 1}, ghi nhận lúc ${timeText}.`;
    }

    function videoSummary() {
      if (latestVideos.length) {
        return `Hiện có ${latestVideos.length} video sự cố đã lưu. Bạn có thể chọn video trong mục “Video sự cố” để xem lại. ${latestAlertSummary()}`;
      }

      return "Hiện chưa có video sự cố được lưu. Nếu cảnh báo vừa xảy ra, hệ thống có thể cần thêm thời gian để ghi xong video.";
    }

    function staffContactSummary() {
      return `Bạn có thể liên hệ nhân viên chăm sóc qua số ${carePhone}. Khi gọi, bạn nên nói rõ tên người thân, số phòng và thời điểm nhận cảnh báo để nhân viên kiểm tra nhanh hơn.`;
    }

    function answerQuestion(question) {
      const text = normalizeText(question);

      if (hasAny(text, intentKeywords.acknowledgement)) {
        return acknowledgementSummary();
      }

      if (hasAny(text, intentKeywords.staffContact)) {
        return staffContactSummary();
      }

      if (hasAny(text, intentKeywords.healthRisk)) {
        return `Mình có thể cung cấp thời gian, vị trí và trạng thái cảnh báo từ hệ thống camera, nhưng không thể kết luận tình trạng sức khỏe thực tế. Để xác nhận chính xác, vui lòng liên hệ nhân viên chăm sóc qua số ${carePhone}.`;
      }

      if (hasAny(text, intentKeywords.video)) {
        return videoSummary();
      }

      if (hasAny(text, intentKeywords.alertTime) || hasAny(text, intentKeywords.location) || hasAny(text, intentKeywords.fallStatus)) {
        return latestAlertSummary();
      }

      if (hasAny(text, intentKeywords.currentStatus)) {
        return currentStatusSummary();
      }

      if (text.includes("camera") || text.includes("hoat dong")) {
        const activeCount = Number(latestStatus?.active_cameras || 0);
        const labels = Object.entries(cameraLabelMap).map(([id, label]) => `Camera ${Number(id) + 1}: ${label}`).join("; ");
        return `Hiện có ${activeCount} camera đang hoạt động. Danh sách khu vực đang theo dõi: ${labels}.`;
      }

      if (hasAny(text, intentKeywords.outOfScopeCare)) {
        return `Mình chưa có dữ liệu về nội dung này trên hệ thống camera. Bạn có thể liên hệ nhân viên chăm sóc qua số ${carePhone} để xác nhận chính xác.`;
      }

      return `Mình chưa hiểu rõ câu hỏi này. Bạn có thể hỏi về tình trạng hiện tại, thời gian cảnh báo, vị trí té ngã, video sự cố, trạng thái xác nhận hoặc số điện thoại nhân viên chăm sóc.`;
    }

    function sendQuestion(questionFromButton = "") {
      const question = String(questionFromButton || chatInputEl.value || "").trim();
      if (!question) return;
      chatBoxEl.innerHTML += `<div class="chat-row user"><strong>Người thân:</strong> ${escapeHtml(question)}</div>`;
      chatBoxEl.innerHTML += `<div class="chat-row"><strong>Chatbox:</strong> ${escapeHtml(answerQuestion(question))}</div>`;
      chatInputEl.value = "";
      chatBoxEl.scrollTop = chatBoxEl.scrollHeight;
    }

    chatWidgetEl.hidden = true;
    chatFabEl.addEventListener("click", () => {
      chatWidgetEl.hidden = !chatWidgetEl.hidden;
      if (!chatWidgetEl.hidden) chatInputEl.focus();
    });
    chatCloseEl.addEventListener("click", () => {
      chatWidgetEl.hidden = true;
    });
    document.querySelectorAll(".question-chip").forEach((button) => button.addEventListener("click", () => sendQuestion(button.dataset.question || button.textContent)));
    chatSendEl.addEventListener("click", () => sendQuestion());
    chatInputEl.addEventListener("keydown", (event) => { if (event.key === "Enter") sendQuestion(); });
    setupCameraStreams();
    loadFamilyConfig().then(loadFamilyData);
    window.setInterval(loadFamilyData, 3000);
  </script>
</body>
</html>
"""
INDEX_HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Giám sát té ngã đa camera</title>
  <style>
    :root {
      --bg: #020403;
      --surface: #07110d;
      --surface-2: #0b1a14;
      --line: rgba(55, 255, 154, 0.22);
      --line-strong: rgba(25, 211, 255, 0.42);
      --text: #eefcf7;
      --muted: #8fb4ab;
      --green: #37ff9a;
      --cyan: #19d3ff;
      --danger: #ff293d;
      --warning: #ffcf4a;
      --normal: #37ff9a;
      --sitting: #4bbcff;
      --bending: #ffb020;
      --lying: #d65cff;
      --shadow: rgba(0, 0, 0, 0.58);
    }

    * {
      box-sizing: border-box;
      letter-spacing: 0;
    }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        linear-gradient(rgba(55, 255, 154, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(25, 211, 255, 0.03) 1px, transparent 1px),
        var(--bg);
      background-size: 44px 44px;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      overflow-x: hidden;
    }

    .page {
      width: min(1540px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 30px;
    }

    .topbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: center;
      margin-bottom: 18px;
    }

    .eyebrow {
      margin: 0 0 8px;
      color: var(--green);
      font-size: 13px;
      font-weight: 800;
      text-transform: uppercase;
    }

    h1 {
      margin: 0;
      font-size: 32px;
      line-height: 1.1;
    }

    .subtitle {
      margin: 10px 0 0;
      max-width: 820px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.55;
    }

    .system-pill,
    .panel,
    .stat-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(5, 15, 12, 0.88);
      box-shadow: 0 18px 52px var(--shadow), inset 0 0 22px rgba(25, 211, 255, 0.035);
    }

    .system-pill {
      min-width: 238px;
      padding: 14px 16px;
    }

    .system-pill span {
      display: block;
      margin-bottom: 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .system-pill strong {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 18px;
      font-style: normal;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 16px var(--green);
      flex: 0 0 auto;
    }

    body.fall-alert .dot {
      background: var(--danger);
      box-shadow: 0 0 16px var(--danger);
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      align-items: start;
    }

    .camera-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }

    .camera-tile {
      position: relative;
      overflow: hidden;
      border: 2px solid rgba(25, 211, 255, 0.34);
      border-radius: 8px;
      padding: 10px;
      background: linear-gradient(145deg, rgba(7, 18, 14, 0.96), rgba(0, 0, 0, 0.96));
      box-shadow: 0 20px 64px rgba(0, 0, 0, 0.54), 0 0 28px rgba(25, 211, 255, 0.11);
      transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
    }

    .camera-tile.alert {
      border-color: var(--danger);
      box-shadow: 0 24px 72px rgba(0, 0, 0, 0.66), 0 0 0 3px rgba(255, 41, 61, 0.24), 0 0 38px rgba(255, 41, 61, 0.40);
    }

    .camera-tile.warn {
      border-color: var(--warning);
      box-shadow: 0 24px 72px rgba(0, 0, 0, 0.60), 0 0 34px rgba(255, 207, 74, 0.24);
    }

    .camera-tile.idle {
      border-color: rgba(143, 180, 171, 0.22);
      opacity: 0.86;
    }

    .camera-tile.status-normal {
      border-color: rgba(55, 255, 154, 0.68);
      box-shadow: 0 20px 64px rgba(0, 0, 0, 0.54), 0 0 28px rgba(55, 255, 154, 0.16);
    }

    .camera-tile.status-sitting {
      border-color: var(--sitting);
      box-shadow: 0 20px 64px rgba(0, 0, 0, 0.54), 0 0 34px rgba(75, 188, 255, 0.30);
    }

    .camera-tile.status-bending,
    .camera-tile.status-falling {
      border-color: var(--warning);
      box-shadow: 0 24px 72px rgba(0, 0, 0, 0.60), 0 0 34px rgba(255, 207, 74, 0.30);
    }

    .camera-tile.status-lying {
      border-color: var(--lying);
      box-shadow: 0 24px 72px rgba(0, 0, 0, 0.60), 0 0 34px rgba(214, 92, 255, 0.32);
    }

    .camera-tile.status-fallen,
    .camera-tile.status-faint,
    .camera-tile.status-unconscious {
      border-color: var(--danger);
      box-shadow: 0 24px 72px rgba(0, 0, 0, 0.66), 0 0 0 3px rgba(255, 41, 61, 0.24), 0 0 38px rgba(255, 41, 61, 0.40);
    }


    .tile-top,
    .tile-footer,
    .source-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .tile-top {
      padding: 0 2px 9px;
      color: var(--muted);
      font-size: 13px;
    }

    .live-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
      font-weight: 900;
    }

    .live-chip::before {
      content: "";
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 14px var(--green);
    }

    .camera-tile.idle .live-chip::before {
      background: var(--muted);
      box-shadow: none;
    }

    .camera-tile.alert .live-chip::before {
      background: var(--danger);
      box-shadow: 0 0 14px var(--danger);
    }

    .tile-state {
      border: 1px solid rgba(55, 255, 154, 0.25);
      border-radius: 6px;
      padding: 5px 8px;
      color: var(--green);
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
    }

    .camera-tile.alert .tile-state {
      border-color: rgba(255, 41, 61, 0.54);
      color: #fff;
      background: rgba(255, 41, 61, 0.20);
    }

    .video-wrap {
      position: relative;
      overflow: hidden;
      border-radius: 6px;
      aspect-ratio: 16 / 9;
      background: #000;
    }

    .video-wrap img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #000;
    }

    .readout {
      position: absolute;
      top: 10px;
      left: 10px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .metric {
      min-width: 74px;
      border: 1px solid rgba(25, 211, 255, 0.24);
      border-radius: 6px;
      padding: 7px 9px;
      background: rgba(0, 0, 0, 0.58);
      backdrop-filter: blur(8px);
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 10px;
      font-weight: 800;
      text-transform: uppercase;
    }

    .metric strong {
      display: block;
      margin-top: 3px;
      color: var(--text);
      font-size: 16px;
      line-height: 1;
    }

    .tile-footer {
      padding: 9px 2px 0;
      color: var(--muted);
      font-size: 12px;
    }

    .source-text {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .side {
      display: grid;
      gap: 14px;
    }

    .panel,
    .stat-card {
      padding: 15px;
    }

    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
    }

    .section-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .section-title h2 {
      margin: 0;
      font-size: 17px;
    }

    .section-title span {
      color: var(--muted);
      font-size: 12px;
    }

    .section-toggle {
      border: 1px solid rgba(25, 211, 255, 0.28);
      border-radius: 999px;
      padding: 6px 11px;
      color: var(--text);
      background: rgba(0, 0, 0, 0.34);
      font: inherit;
      font-size: 12px;
      font-weight: 900;
      line-height: 1;
      cursor: pointer;
      transition: background 160ms ease, border-color 160ms ease, color 160ms ease, transform 160ms ease;
    }

    .section-toggle:hover {
      transform: translateY(-1px);
      border-color: rgba(55, 255, 154, 0.42);
    }

    .section-toggle[aria-expanded="true"] {
      color: #00120d;
      background: var(--green);
      border-color: rgba(55, 255, 154, 0.72);
    }

    .source-list {
      display: grid;
      gap: 10px;
    }

    .source-form {
      display: grid;
      gap: 7px;
      border-top: 1px solid rgba(25, 211, 255, 0.12);
      padding-top: 10px;
    }

    .source-form:first-child {
      border-top: 0;
      padding-top: 0;
    }

    .source-label {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--text);
      font-size: 13px;
      font-weight: 900;
    }

    .source-label span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .source-input {
      width: 100%;
      min-width: 0;
      border: 1px solid rgba(25, 211, 255, 0.28);
      border-radius: 8px;
      padding: 10px 11px;
      color: var(--text);
      background: rgba(0, 0, 0, 0.36);
      font: inherit;
      outline: none;
    }

    .source-input:focus {
      border-color: rgba(55, 255, 154, 0.72);
      box-shadow: 0 0 0 3px rgba(55, 255, 154, 0.10);
    }

    .source-button {
      flex: 0 0 auto;
      border: 0;
      border-radius: 8px;
      padding: 9px 12px;
      color: #00120d;
      background: var(--green);
      font: inherit;
      font-weight: 900;
      cursor: pointer;
    }

    .source-button:disabled {
      cursor: wait;
      opacity: 0.62;
    }

    .source-message {
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }

    .panel-button,
    .video-item {
      width: 100%;
      border: 1px solid rgba(25, 211, 255, 0.28);
      border-radius: 8px;
      color: var(--text);
      background: rgba(0, 0, 0, 0.34);
      font: inherit;
      text-align: left;
      cursor: pointer;
    }

    .panel-button {
      padding: 10px 12px;
      color: #00120d;
      background: var(--green);
      font-weight: 900;
      text-align: center;
    }

    .panel-button.secondary {
      color: var(--text);
      background: rgba(0, 0, 0, 0.34);
    }

    .video-actions {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      margin-bottom: 10px;
    }

    .video-actions .panel-button.secondary {
      width: auto;
      padding-inline: 11px;
    }

    .video-library {
      display: grid;
      gap: 10px;
    }

    .replay-player {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      border: 1px solid rgba(25, 211, 255, 0.28);
      border-radius: 8px;
      background: #000;
      object-fit: contain;
    }

    .video-list {
      display: grid;
      gap: 8px;
      max-height: 290px;
      overflow: auto;
      padding-right: 2px;
    }

    .video-item {
      display: grid;
      gap: 5px;
      padding: 10px;
    }

    .video-item:hover,
    .video-item.active {
      border-color: rgba(55, 255, 154, 0.72);
      box-shadow: 0 0 0 3px rgba(55, 255, 154, 0.08);
    }

    .video-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
      font-weight: 900;
    }

    .video-meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }

    [hidden] {
      display: none !important;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .stat-label {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
    }

    .stat-value {
      margin: 0;
      font-size: 27px;
      font-weight: 900;
      line-height: 1;
    }

    .good {
      color: var(--green);
    }

    .danger {
      color: var(--danger);
    }

    .warn {
      color: var(--warning);
    }

    .track-list {
      display: grid;
      gap: 10px;
      max-height: 360px;
      overflow: auto;
      padding-right: 2px;
    }

    .alert-history-list {
      display: grid;
      gap: 10px;
      max-height: 430px;
      overflow: auto;
      padding-right: 2px;
    }

    .alert-summary-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 0 0 10px;
    }

    .alert-summary-grid div {
      border: 1px solid rgba(25, 211, 255, 0.16);
      border-radius: 8px;
      padding: 8px;
      background: rgba(255, 255, 255, 0.035);
    }

    .alert-summary-grid span {
      display: block;
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }

    .alert-summary-grid strong {
      color: var(--cyan);
      font-size: 20px;
      line-height: 1;
    }

    .alert-history-tools {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }

    .alert-history-tools label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }

    .alert-history-tools select {
      width: 100%;
      min-height: 34px;
      border: 1px solid rgba(25, 211, 255, 0.22);
      border-radius: 7px;
      padding: 6px 8px;
      color: var(--text);
      background: rgba(0, 0, 0, 0.34);
      font: inherit;
      text-transform: none;
    }

    .alert-history-item {
      display: grid;
      grid-template-columns: 76px minmax(0, 1fr);
      gap: 11px;
      border: 1px solid rgba(25, 211, 255, 0.18);
      border-left: 4px solid var(--danger);
      border-radius: 8px;
      padding: 10px;
      background:
        linear-gradient(135deg, rgba(255, 41, 61, 0.13), rgba(25, 211, 255, 0.04)),
        rgba(0, 0, 0, 0.30);
    }

    .alert-history-item.acknowledged {
      border-left-color: var(--green);
      background:
        linear-gradient(135deg, rgba(55, 255, 154, 0.12), rgba(25, 211, 255, 0.04)),
        rgba(0, 0, 0, 0.26);
    }

    .alert-history-item.severity-warning {
      border-left-color: var(--warning);
    }

    .alert-history-item.severity-danger {
      border-left-color: var(--danger);
    }

    .alert-history-item.severity-emergency {
      border-left-color: #ff4fd8;
      background:
        linear-gradient(135deg, rgba(255, 79, 216, 0.16), rgba(255, 41, 61, 0.10)),
        rgba(0, 0, 0, 0.32);
    }

    .alert-thumb {
      width: 76px;
      aspect-ratio: 1 / 1;
      border: 1px solid rgba(25, 211, 255, 0.24);
      border-radius: 7px;
      object-fit: cover;
      background: rgba(0, 0, 0, 0.45);
    }

    .alert-thumb-placeholder {
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 11px;
      font-weight: 900;
    }

    .alert-history-main {
      min-width: 0;
      display: grid;
      gap: 7px;
    }

    .alert-history-top,
    .alert-history-meta,
    .alert-history-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .alert-history-title {
      min-width: 0;
      font-weight: 900;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .alert-badge {
      flex: 0 0 auto;
      border-radius: 999px;
      padding: 4px 8px;
      color: #fff;
      background: var(--danger);
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
    }

    .alert-badge.safe {
      color: #00120d;
      background: var(--green);
    }

    .alert-badge.severity-warning {
      color: #231700;
      background: var(--warning);
    }

    .alert-badge.severity-danger {
      background: var(--danger);
    }

    .alert-badge.severity-emergency {
      background: #ff4fd8;
    }

    .alert-history-meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .alert-history-actions a {
      color: var(--cyan);
      font-size: 12px;
      font-weight: 900;
      text-decoration: none;
    }

    .alert-history-actions a:hover {
      color: var(--green);
    }

    .track-item {
      border: 1px solid rgba(25, 211, 255, 0.18);
      border-radius: 8px;
      padding: 11px;
      background: rgba(0, 0, 0, 0.28);
    }

    .track-item.status-normal {
      border-color: rgba(55, 255, 154, 0.46);
      background: rgba(55, 255, 154, 0.08);
    }

    .track-item.status-sitting {
      border-color: rgba(75, 188, 255, 0.62);
      background: rgba(75, 188, 255, 0.10);
    }

    .track-item.status-bending,
    .track-item.status-falling {
      border-color: rgba(255, 207, 74, 0.72);
      background: rgba(255, 207, 74, 0.10);
    }

    .track-item.status-lying {
      border-color: rgba(214, 92, 255, 0.72);
      background: rgba(214, 92, 255, 0.10);
    }

    .track-item.status-fallen,
    .track-item.status-faint,
    .track-item.status-unconscious {
      border-color: rgba(255, 41, 61, 0.72);
      background: rgba(255, 41, 61, 0.12);
    }


    .track-top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 7px;
    }

    .track-id {
      font-weight: 900;
    }

    .status-tag {
      border-radius: 6px;
      padding: 4px 8px;
      color: #00120d;
      background: var(--green);
      font-size: 12px;
      font-weight: 900;
    }

    .status-tag.alert {
      color: #fff;
      background: var(--danger);
    }

    .status-tag.warn {
      color: #00120d;
      background: var(--warning);
    }

    .track-meta,
    .replay-status,
    .empty,
    .footer-note {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    .footer-note {
      margin-top: 14px;
    }

    @media (max-width: 1180px) {
      .layout {
        grid-template-columns: 1fr;
      }

      .side {
        grid-template-columns: 1fr 1fr;
      }
    }

    @media (max-width: 840px) {
      .camera-grid,
      .side,
      .topbar {
        grid-template-columns: 1fr;
      }

      .page {
        width: min(100% - 20px, 840px);
        padding-top: 18px;
      }

      h1 {
        font-size: 26px;
      }

      .alert-history-item {
        grid-template-columns: 64px minmax(0, 1fr);
      }

      .alert-thumb {
        width: 64px;
      }

      .alert-history-tools {
        grid-template-columns: 1fr;
      }

    }
  </style>
</head>
<body>
  <main class="page">
    <header class="topbar">
      <div>
        <p class="eyebrow">Hệ thống an toàn AI</p>
        <h1>Giám sát té ngã 4 camera</h1>
        <p class="subtitle">Nhập nguồn camera cho từng khu vực. Mỗi camera có luồng xử lý riêng, tự đồng bộ khung hình mới để giảm độ trễ khi giám sát.</p>
      </div>
      <div class="system-pill">
        <span>Trạng thái hệ thống</span>
        <strong><i class="dot"></i><em id="system-status">Đang khởi động</em></strong>
      </div>
    </header>

    <section class="layout">
      <div class="camera-grid" id="camera-grid">
        <article class="camera-tile idle" data-camera="0">
          <div class="tile-top"><span class="live-chip">CAM 01</span><span class="tile-state">CHỜ NGUỒN</span></div>
          <div class="video-wrap">
            <img id="stream-0" src="/video_feed?camera=0" alt="Camera 1">
            <div class="readout"><div class="metric"><span>Người</span><strong class="people">0</strong></div><div class="metric"><span>FPS</span><strong class="fps">0.0</strong></div></div>
          </div>
          <div class="tile-footer"><span class="source-text">Nguồn: --</span><span class="time-text">t=0.0s</span></div>
        </article>
        <article class="camera-tile idle" data-camera="1">
          <div class="tile-top"><span class="live-chip">CAM 02</span><span class="tile-state">CHỜ NGUỒN</span></div>
          <div class="video-wrap">
            <img id="stream-1" src="/video_feed?camera=1" alt="Camera 2">
            <div class="readout"><div class="metric"><span>Người</span><strong class="people">0</strong></div><div class="metric"><span>FPS</span><strong class="fps">0.0</strong></div></div>
          </div>
          <div class="tile-footer"><span class="source-text">Nguồn: --</span><span class="time-text">t=0.0s</span></div>
        </article>
        <article class="camera-tile idle" data-camera="2">
          <div class="tile-top"><span class="live-chip">CAM 03</span><span class="tile-state">CHỜ NGUỒN</span></div>
          <div class="video-wrap">
            <img id="stream-2" src="/video_feed?camera=2" alt="Camera 3">
            <div class="readout"><div class="metric"><span>Người</span><strong class="people">0</strong></div><div class="metric"><span>FPS</span><strong class="fps">0.0</strong></div></div>
          </div>
          <div class="tile-footer"><span class="source-text">Nguồn: --</span><span class="time-text">t=0.0s</span></div>
        </article>
        <article class="camera-tile idle" data-camera="3">
          <div class="tile-top"><span class="live-chip">CAM 04</span><span class="tile-state">CHỜ NGUỒN</span></div>
          <div class="video-wrap">
            <img id="stream-3" src="/video_feed?camera=3" alt="Camera 4">
            <div class="readout"><div class="metric"><span>Người</span><strong class="people">0</strong></div><div class="metric"><span>FPS</span><strong class="fps">0.0</strong></div></div>
          </div>
          <div class="tile-footer"><span class="source-text">Nguồn: --</span><span class="time-text">t=0.0s</span></div>
        </article>
      </div>

      <aside class="side">
        <section class="panel">
          <div class="section-title">
            <h2>Nguồn camera</h2>
            <div class="section-actions">
              <span id="source-panel-meta">IP stream</span>
              <button class="section-toggle" id="source-toggle" type="button" aria-expanded="true" aria-controls="source-list">Ẩn</button>
            </div>
          </div>
          <div class="source-list" id="source-list"></div>
        </section>

        <section class="stats">
          <div class="stat-card"><p class="stat-label">Camera đang chạy</p><p class="stat-value good" id="active-count">0</p></div>
          <div class="stat-card"><p class="stat-label">Tổng số người</p><p class="stat-value good" id="people-count">0</p></div>
          <div class="stat-card"><p class="stat-label">Cảnh báo</p><p class="stat-value good" id="fall-count">0</p></div>
          <div class="stat-card"><p class="stat-label">FPS trung bình</p><p class="stat-value" id="fps-value">0.0</p></div>
        </section>

        <section class="panel">
          <div class="section-title">
            <h2>Lịch sử cảnh báo</h2>
            <div class="section-actions">
              <span id="alert-history-count">0 sự kiện</span>
              <button class="section-toggle" id="demo-alert-button" type="button">Demo</button>
            </div>
          </div>
          <div class="alert-summary-grid">
            <div><span>Tổng</span><strong id="alert-total-count">0</strong></div>
            <div><span>Chờ xác nhận</span><strong id="alert-pending-count">0</strong></div>
            <div><span>Đã xác nhận</span><strong id="alert-ack-count">0</strong></div>
            <div><span>Khẩn cấp</span><strong id="alert-emergency-count">0</strong></div>
          </div>
          <div class="alert-history-tools">
            <label>
              Trạng thái
              <select id="alert-status-filter">
                <option value="all">Tất cả</option>
                <option value="FALLING">Đang ngã</option>
                <option value="FALLEN">Đã ngã</option>
                <option value="FAINT">Ngất</option>
                <option value="UNCONSCIOUS">Bất tỉnh</option>
              </select>
            </label>
            <label>
              Mức độ
              <select id="alert-severity-filter">
                <option value="all">Tất cả</option>
                <option value="warning">Cảnh báo</option>
                <option value="danger">Nguy hiểm</option>
                <option value="emergency">Khẩn cấp</option>
              </select>
            </label>
            <label>
              Xác nhận
              <select id="alert-ack-filter">
                <option value="all">Tất cả</option>
                <option value="pending">Chưa xác nhận</option>
                <option value="acknowledged">Đã xác nhận</option>
              </select>
            </label>
          </div>
          <div class="alert-history-list" id="alert-history-list">
            <p class="empty">Chưa có cảnh báo nào.</p>
          </div>
        </section>

        <section class="panel">
          <div class="section-title">
            <h2>Video đã lưu</h2>
            <span id="video-count">0 video</span>
          </div>
          <div class="video-actions">
            <button class="panel-button" id="video-toggle" type="button">Xem lại video</button>
            <button class="panel-button secondary" id="video-refresh" type="button">Tải lại</button>
          </div>
          <div class="video-library" id="video-library" hidden>
            <img class="replay-player" id="replay-player" alt="Video đã lưu" hidden>
            <p class="replay-status" id="replay-status"></p>
            <div class="video-list" id="video-list">
              <p class="empty">Chưa có video trong outputs.</p>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="section-title">
            <h2>Đối tượng theo dõi</h2>
            <span id="track-count">0 track</span>
          </div>
          <div class="track-list" id="track-list">
            <p class="empty">Chưa có người trong khung hình.</p>
          </div>
        </section>
      </aside>
    </section>

    <p class="footer-note">Khuyến nghị: dùng 480p hoặc 720p, 10-15 FPS để giảm tải cho YOLO Pose và luồng MJPEG.</p>
  </main>

  <script>
    const cameraCount = 4;
    const dangerStates = new Set(["FALLEN", "FAINT", "UNCONSCIOUS"]);
    const warningStates = new Set(["FALLING"]);
    const statusLabelMap = {
      NORMAL: "Bình thường",
      WALKING_STANDING: "Đi/đứng",
      SITTING: "Ngồi",
      BENDING: "Cúi người",
      LYING_INTENTIONAL: "Nằm chủ động",
      FALLING: "Đang ngã",
      FALLEN: "Đã ngã",
      FAINT: "Ngất",
      UNCONSCIOUS: "Bất tỉnh",
    };

    function statusLabel(status) {
      const key = String(status || "");
      return statusLabelMap[key] || key.replaceAll("_", " ").toLowerCase();
    }

    function statusClass(status) {
      const key = String(status || "NORMAL");
      if (key === "SITTING") return "status-sitting";
      if (key === "BENDING") return "status-bending";
      if (key === "LYING_INTENTIONAL") return "status-lying";
      if (key === "FALLING") return "status-falling";
      if (key === "FALLEN") return "status-fallen";
      if (key === "FAINT") return "status-faint";
      if (key === "UNCONSCIOUS") return "status-unconscious";
      return "status-normal";
    }

    function severityLabel(severity) {
      const key = String(severity || "warning");
      if (key === "emergency") return "Khẩn cấp";
      if (key === "danger") return "Nguy hiểm";
      return "Cảnh báo";
    }

    function severityClass(severity) {
      const key = String(severity || "warning");
      if (key === "emergency") return "severity-emergency";
      if (key === "danger") return "severity-danger";
      return "severity-warning";
    }


    const statusEl = document.getElementById("system-status");
    const sourceListEl = document.getElementById("source-list");
    const sourceToggleEl = document.getElementById("source-toggle");
    const sourcePanelMetaEl = document.getElementById("source-panel-meta");
    const activeEl = document.getElementById("active-count");
    const peopleEl = document.getElementById("people-count");
    const fallEl = document.getElementById("fall-count");
    const fpsEl = document.getElementById("fps-value");
    const alertHistoryCountEl = document.getElementById("alert-history-count");
    const alertTotalCountEl = document.getElementById("alert-total-count");
    const alertPendingCountEl = document.getElementById("alert-pending-count");
    const alertAckCountEl = document.getElementById("alert-ack-count");
    const alertEmergencyCountEl = document.getElementById("alert-emergency-count");
    const alertStatusFilterEl = document.getElementById("alert-status-filter");
    const alertSeverityFilterEl = document.getElementById("alert-severity-filter");
    const alertAckFilterEl = document.getElementById("alert-ack-filter");
    const demoAlertButtonEl = document.getElementById("demo-alert-button");
    const alertHistoryListEl = document.getElementById("alert-history-list");
    const trackCountEl = document.getElementById("track-count");
    const trackListEl = document.getElementById("track-list");
    const videoCountEl = document.getElementById("video-count");
    const videoToggleEl = document.getElementById("video-toggle");
    const videoRefreshEl = document.getElementById("video-refresh");
    const videoLibraryEl = document.getElementById("video-library");
    const videoListEl = document.getElementById("video-list");
    const replayPlayerEl = document.getElementById("replay-player");
    const replayStatusEl = document.getElementById("replay-status");
    const sourcePanelStorageKey = "fall-ui-source-panel-open";
    let videosLoaded = false;
    let currentVideoPath = "";
    let latestAlertHistory = [];

    function cameraTile(cameraId) {
      return document.querySelector(`[data-camera="${cameraId}"]`);
    }

    function renderSourceForms() {
      sourceListEl.innerHTML = Array.from({ length: cameraCount }, (_, cameraId) => `
        <form class="source-form" data-camera="${cameraId}">
          <label class="source-label" for="source-input-${cameraId}">
            Camera ${String(cameraId + 1).padStart(2, "0")}
            <span id="source-message-${cameraId}">Sẵn sàng</span>
          </label>
          <input
            class="source-input"
            id="source-input-${cameraId}"
            type="text"
            autocomplete="off"
            placeholder="http://192.168.1.20:8080/video"
          >
          <div class="source-actions">
            <button class="source-button" id="source-button-${cameraId}" type="submit">Áp dụng</button>
            <span class="source-message">RTSP, IP Webcam, DroidCam hoặc webcam index.</span>
          </div>
        </form>
      `).join("");

      document.querySelectorAll(".source-form").forEach((form) => {
        form.addEventListener("submit", (event) => {
          event.preventDefault();
          const cameraId = Number(form.dataset.camera || 0);
          const input = document.getElementById(`source-input-${cameraId}`);
          const source = input.value.trim();
          if (!source) {
            setSourceMessage(cameraId, "Nhập URL trước");
            return;
          }
          changeSource(cameraId, source);
        });
      });
    }

    function setSourceMessage(cameraId, message) {
      const el = document.getElementById(`source-message-${cameraId}`);
      if (el) {
        el.textContent = message;
      }
    }

    function setSourcePanelOpen(isOpen) {
      sourceListEl.hidden = !isOpen;
      sourceToggleEl.textContent = isOpen ? "An" : "Hiện";
      sourceToggleEl.setAttribute("aria-expanded", String(isOpen));
      sourcePanelMetaEl.textContent = isOpen ? "IP stream" : "Đã ẩn";
      try {
        window.localStorage.setItem(sourcePanelStorageKey, isOpen ? "1" : "0");
      } catch (error) {
      }
    }

    function loadSourcePanelPreference() {
      try {
        return window.localStorage.getItem(sourcePanelStorageKey) !== "0";
      } catch (error) {
        return true;
      }
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[char]));
    }

    function formatBytes(bytes) {
      const size = Number(bytes || 0);
      if (size < 1024) {
        return `${size} B`;
      }
      const units = ["KB", "MB", "GB"];
      let value = size / 1024;
      let unitIndex = 0;
      while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024;
        unitIndex += 1;
      }
      return `${value.toFixed(value >= 10 ? 1 : 2)} ${units[unitIndex]}`;
    }

    function formatModified(seconds) {
      const value = Number(seconds || 0);
      if (!value) {
        return "--";
      }
      return new Date(value * 1000).toLocaleString("vi-VN", { hour12: false });
    }

    function outputVideoUrl(path) {
      return `/replay_video?path=${encodeURIComponent(path)}&ts=${Date.now()}`;
    }

    function renderVideos(videos) {
      videoCountEl.textContent = `${videos.length} video`;
      videoListEl.innerHTML = "";
      if (videos.length === 0) {
        videoListEl.innerHTML = '<p class="empty">Chưa có video trong outputs.</p>';
        replayPlayerEl.hidden = true;
        replayPlayerEl.removeAttribute("src");
        replayStatusEl.textContent = "";
        currentVideoPath = "";
        return;
      }

      videos.forEach((video) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = `video-item ${video.path === currentVideoPath ? "active" : ""}`;
        item.dataset.path = video.path;
        const folder = video.folder ? `${video.folder} / ` : "";
        item.innerHTML = `
          <span class="video-name">${escapeHtml(video.name)}</span>
          <span class="video-meta">${escapeHtml(folder)}${formatBytes(video.size_bytes)} - ${formatModified(video.modified_at)}</span>
        `;
        item.addEventListener("click", () => playSavedVideo(video));
        videoListEl.appendChild(item);
      });
    }

    async function loadVideos(force = false) {
      if (videosLoaded && !force) {
        return;
      }
      videoCountEl.textContent = "Đang tải";
      try {
        const response = await fetch("/videos", { cache: "no-store" });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data.error || "Không đọc được outputs.");
        }
        renderVideos(data.videos || []);
        videosLoaded = true;
      } catch (error) {
        videoCountEl.textContent = "Loi";
        videoListEl.innerHTML = `<p class="empty">${escapeHtml(error.message || "Không đọc được outputs.")}</p>`;
      }
    }

    function playSavedVideo(video) {
      currentVideoPath = video.path;
      replayPlayerEl.hidden = false;
      replayStatusEl.textContent = "Đang tải video đã lưu...";
      replayPlayerEl.src = outputVideoUrl(video.path);
      document.querySelectorAll(".video-item").forEach((item) => {
        item.classList.toggle("active", item.dataset.path === currentVideoPath);
      });
    }

    replayPlayerEl.addEventListener("load", () => {
      replayStatusEl.textContent = "";
    });

    replayPlayerEl.addEventListener("error", () => {
      replayStatusEl.textContent = "Không hiển thị được video này. Hãy chờ clip ghi xong rồi bấm Tải lại.";
    });

    function renderCamera(data) {
      const cameraId = Number(data.camera_id || 0);
      const tile = cameraTile(cameraId);
      if (!tile) {
        return;
      }

      const running = Boolean(data.running);
      const fallAlert = Boolean(data.fall_alert);
      const faintAlert = Boolean(data.faint_alert);
      const tracks = data.tracks || [];
      const warning = !fallAlert && tracks.some((track) => warningStates.has(track.status));
      const dangerTrack = tracks.find((track) => dangerStates.has(track.status));
      const warningTrack = tracks.find((track) => warningStates.has(track.status));
      const dominantStatus = dangerTrack?.status || warningTrack?.status || tracks[0]?.status || "NORMAL";
      const people = Number(data.people || 0);
      const fps = Number(data.processing_fps || 0);
      const source = data.source || "";

      tile.classList.toggle("idle", !running);
      tile.classList.toggle("alert", fallAlert);
      tile.classList.toggle("warn", warning);
      ["status-normal", "status-sitting", "status-bending", "status-lying", "status-falling", "status-fallen", "status-faint", "status-unconscious"].forEach((name) => tile.classList.remove(name));
      if (running) {
        tile.classList.add(statusClass(dominantStatus));
      }
      tile.querySelector(".tile-state").textContent = faintAlert ? "NGẤT" : fallAlert ? "CẢNH BÁO NGÃ" : warning ? "ĐANG NGÃ" : running ? "ĐANG CHẠY" : "CHỜ NGUỒN";
      tile.querySelector(".people").textContent = people;
      tile.querySelector(".fps").textContent = fps.toFixed(1);
      tile.querySelector(".source-text").textContent = `Nguồn: ${source || "--"}`;
      tile.querySelector(".time-text").textContent = `t=${Number(data.timestamp_sec || 0).toFixed(1)}s`;

      const input = document.getElementById(`source-input-${cameraId}`);
      if (input && source && document.activeElement !== input) {
        input.value = source;
      }
    }

    function renderTracks(cameras) {
      const tracks = [];
      cameras.forEach((camera) => {
        (camera.tracks || []).forEach((track) => {
          tracks.push({ ...track, camera_id: camera.camera_id });
        });
      });

      trackCountEl.textContent = `${tracks.length} track`;
      if (tracks.length === 0) {
        trackListEl.innerHTML = '<p class="empty">Chưa có người trong khung hình.</p>';
        return;
      }

      trackListEl.innerHTML = tracks.map((track) => {
        const alert = dangerStates.has(track.status);
        const warn = warningStates.has(track.status);
        const score = Number(track.fall_score || 0).toFixed(2);
        const confidence = Number(track.confidence || 0).toFixed(2);
        return `
          <article class="track-item ${statusClass(track.status)}">
            <div class="track-top">
              <span class="track-id">CAM ${String(Number(track.camera_id) + 1).padStart(2, "0")} - ID ${track.track_id}</span>
              <span class="status-tag ${alert ? "alert" : warn ? "warn" : ""}">${escapeHtml(track.status_vi || statusLabel(track.status))}</span>
            </div>
            <p class="track-meta">Tin cậy: ${confidence} - Điểm ngã: ${score}</p>
            <p class="track-meta">Lý do: ${track.reason || "runtime"}</p>
          </article>
        `;
      }).join("");
    }

    function updateAlertSummary(alerts, filteredAlerts) {
      const pending = alerts.filter((item) => !Boolean(item.acknowledged)).length;
      const acknowledged = alerts.length - pending;
      const emergency = alerts.filter((item) => String(item.severity || "") === "emergency").length;
      alertHistoryCountEl.textContent = `${filteredAlerts.length}/${alerts.length} sự kiện`;
      alertTotalCountEl.textContent = alerts.length;
      alertPendingCountEl.textContent = pending;
      alertAckCountEl.textContent = acknowledged;
      alertEmergencyCountEl.textContent = emergency;
    }

    function filteredAlertHistory(alerts) {
      const statusFilter = alertStatusFilterEl.value || "all";
      const severityFilter = alertSeverityFilterEl.value || "all";
      const ackFilter = alertAckFilterEl.value || "all";

      return alerts.filter((item) => {
        const acknowledged = Boolean(item.acknowledged);
        if (statusFilter !== "all" && String(item.status || "") !== statusFilter) {
          return false;
        }
        if (severityFilter !== "all" && String(item.severity || "warning") !== severityFilter) {
          return false;
        }
        if (ackFilter === "pending" && acknowledged) {
          return false;
        }
        if (ackFilter === "acknowledged" && !acknowledged) {
          return false;
        }
        return true;
      });
    }

    function renderAlertHistory(alerts) {
      latestAlertHistory = Array.isArray(alerts) ? alerts : [];
      const visibleAlerts = filteredAlertHistory(latestAlertHistory);
      updateAlertSummary(latestAlertHistory, visibleAlerts);

      if (latestAlertHistory.length === 0) {
        alertHistoryListEl.innerHTML = '<p class="empty">Chưa có cảnh báo nào.</p>';
        return;
      }
      if (visibleAlerts.length === 0) {
        alertHistoryListEl.innerHTML = '<p class="empty">Không có cảnh báo phù hợp bộ lọc.</p>';
        return;
      }

      alertHistoryListEl.innerHTML = visibleAlerts.map((item) => {
        const acknowledged = Boolean(item.acknowledged);
        const statusText = item.status_vi || statusLabel(item.status);
        const severity = item.severity || "warning";
        const severityText = item.severity_vi || severityLabel(severity);
        const severityCss = severityClass(severity);
        const confidence = Number(item.confidence || 0).toFixed(2);
        const fallScore = Number(item.fall_score || 0).toFixed(2);
        const reminderCount = Number(item.reminder_count || 0);
        const createdAt = formatModified(item.created_at);
        const acknowledgedAt = item.acknowledged_at ? formatModified(item.acknowledged_at) : "";
        const thumb = item.image_available
          ? `<a href="${escapeHtml(item.image_url)}" target="_blank" rel="noopener"><img class="alert-thumb" src="${escapeHtml(item.image_url)}" alt="Ảnh cảnh báo"></a>`
          : '<div class="alert-thumb alert-thumb-placeholder">NO IMG</div>';
        const ackMeta = acknowledged ? `Đã xác nhận ${acknowledgedAt ? `lúc ${acknowledgedAt}` : ""}` : "Chưa xác nhận";
        const imageLink = item.image_available
          ? `<a href="${escapeHtml(item.image_url)}" target="_blank" rel="noopener">Mở ảnh</a>`
          : "";
        const liveLink = `${window.location.origin}/`;

        return `
          <article class="alert-history-item ${severityCss} ${acknowledged ? "acknowledged" : ""}">
            ${thumb}
            <div class="alert-history-main">
              <div class="alert-history-top">
                <span class="alert-history-title">CAM ${String(Number(item.camera_id) + 1).padStart(2, "0")} - ${escapeHtml(statusText)}</span>
                <span class="alert-badge ${severityCss}">${escapeHtml(severityText)}</span>
                <span class="alert-badge ${acknowledged ? "safe" : ""}">${acknowledged ? "Đã xác nhận" : "Đang chờ"}</span>
              </div>
              <div class="alert-history-meta">
                <span>${escapeHtml(createdAt)}</span>
                <span>ID ${escapeHtml(item.track_id ?? "--")}</span>
                <span>Tin cậy ${confidence}</span>
                <span>Điểm ngã ${fallScore}</span>
              </div>
              <div class="alert-history-meta">
                <span>${escapeHtml(ackMeta)}</span>
                <span>Nhắc lại ${reminderCount} lần</span>
              </div>
              <div class="alert-history-actions">
                ${imageLink}
                <a href="${escapeHtml(liveLink)}" target="_blank" rel="noopener">Mở trực tiếp</a>
              </div>
            </div>
          </article>
        `;
      }).join("");
    }

    [alertStatusFilterEl, alertSeverityFilterEl, alertAckFilterEl].forEach((select) => {
      select.addEventListener("change", () => renderAlertHistory(latestAlertHistory));
    });

    demoAlertButtonEl.addEventListener("click", sendDemoAlert);

    async function refreshAlerts() {
      try {
        const response = await fetch("/alerts", { cache: "no-store" });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data.error || "Không đọc được lịch sử cảnh báo.");
        }
        renderAlertHistory(data.alerts || []);
      } catch (error) {
        alertHistoryCountEl.textContent = "Lỗi";
        alertHistoryListEl.innerHTML = `<p class="empty">${escapeHtml(error.message || "Không đọc được lịch sử cảnh báo.")}</p>`;
      }
    }

    async function sendDemoAlert() {
      demoAlertButtonEl.disabled = true;
      const previousText = demoAlertButtonEl.textContent;
      demoAlertButtonEl.textContent = "Đang gửi";
      try {
        const response = await fetch("/demo_alert", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ camera_id: 0, status: "FALLEN" }),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data.error || "Không tạo được cảnh báo demo.");
        }
        demoAlertButtonEl.textContent = "Đã gửi";
        await refreshAlerts();
      } catch (error) {
        demoAlertButtonEl.textContent = "Lỗi";
        alertHistoryListEl.innerHTML = `<p class="empty">${escapeHtml(error.message || "Không tạo được cảnh báo demo.")}</p>`;
      } finally {
        window.setTimeout(() => {
          demoAlertButtonEl.disabled = false;
          demoAlertButtonEl.textContent = previousText;
        }, 1200);
      }
    }

    function renderDashboard(data) {
      const cameras = data.cameras || [];
      cameras.forEach(renderCamera);
      renderTracks(cameras);

      const fallCount = cameras.filter((camera) => camera.fall_alert || camera.faint_alert).length;
      const totalPeople = cameras.reduce((total, camera) => total + Number(camera.people || 0), 0);
      const activeCount = Number(data.active_cameras || cameras.filter((camera) => camera.running).length);
      const avgFps = Number(data.processing_fps || 0);

      document.body.classList.toggle("fall-alert", fallCount > 0);
      statusEl.textContent = fallCount > 0 ? `Cảnh báo trên ${fallCount} camera` : activeCount > 0 ? `Đang theo dõi ${activeCount} camera` : "Đang chờ nguồn";
      activeEl.textContent = activeCount;
      peopleEl.textContent = totalPeople;
      fallEl.textContent = fallCount;
      fpsEl.textContent = avgFps.toFixed(1);
      fallEl.className = `stat-value ${fallCount > 0 ? "danger" : "good"}`;
    }

    async function refreshStatus() {
      try {
        const response = await fetch("/status", { cache: "no-store" });
        const data = await response.json();
        renderDashboard(data);
      } catch (error) {
        document.body.classList.remove("fall-alert");
        statusEl.textContent = "Mất kết nối";
      }
    }

    async function changeSource(cameraId, source) {
      const button = document.getElementById(`source-button-${cameraId}`);
      button.disabled = true;
      setSourceMessage(cameraId, "Đang kết nối");
      try {
        const response = await fetch("/source", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ camera_id: cameraId, source }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.ok) {
          throw new Error(data.error || "Không đổi được nguồn camera.");
        }
        setSourceMessage(cameraId, "Đang chạy");
        document.getElementById(`stream-${cameraId}`).src = `/video_feed?camera=${cameraId}&ts=${Date.now()}`;
        await refreshStatus();
      } catch (error) {
        setSourceMessage(cameraId, error.message || "Lỗi kết nối");
      } finally {
        button.disabled = false;
      }
    }

    videoToggleEl.addEventListener("click", async () => {
      const shouldOpen = videoLibraryEl.hidden;
      videoLibraryEl.hidden = !shouldOpen;
      videoToggleEl.textContent = shouldOpen ? "Ẩn danh sách video" : "Xem lại video";
      if (shouldOpen) {
        await loadVideos(true);
      }
    });

    sourceToggleEl.addEventListener("click", () => {
      setSourcePanelOpen(sourceListEl.hidden);
    });

    videoRefreshEl.addEventListener("click", async () => {
      videoLibraryEl.hidden = false;
      videoToggleEl.textContent = "Ẩn danh sách video";
      await loadVideos(true);
    });

    renderSourceForms();
    setSourcePanelOpen(loadSourcePanelPreference());
    loadVideos(true);
    refreshStatus();
    refreshAlerts();
    window.setInterval(refreshStatus, 700);
    window.setInterval(refreshAlerts, 3000);
    window.setInterval(() => loadVideos(true), 5000);
  </script>
</body>
</html>
"""




def mp4_has_moov_atom(path: Path) -> bool:
    """Return True when an MP4/MOV file looks finalized enough to read.

    While OpenCV is still writing an MP4, the metadata atom (moov) is often not
    present yet. Calling FFmpeg/OpenCV metadata readers on that unfinished file
    causes repeated "moov atom not found" messages. This lightweight check lets
    the web UI skip unfinished or corrupt MP4 files before FFmpeg touches them.
    """
    try:
        size = path.stat().st_size
        if size <= 0:
            return False
        window = min(size, 1024 * 1024)
        with path.open("rb") as file:
            head = file.read(window)
            if size > window:
                file.seek(max(0, size - window))
                tail = file.read(window)
            else:
                tail = b""
        return b"moov" in head or b"moov" in tail
    except OSError:
        return False


def is_probably_ready_video(path: Path) -> bool:
    """Avoid probing files that are still being written or already corrupt."""
    try:
        stat = path.stat()
    except OSError:
        return False
    if stat.st_size <= 0:
        return False
    # Skip files that were modified moments ago; they may still be recording.
    if time.time() - float(stat.st_mtime) < VIDEO_READY_GRACE_SECONDS:
        return False
    if path.suffix.lower() in {".mp4", ".mov", ".m4v"} and not mp4_has_moov_atom(path):
        return False
    return True


def source_name(source: str) -> str:
    if str(source).isdigit():
        return f"webcam_{source}"
    parsed = urlparse(str(source))
    if parsed.scheme and parsed.netloc:
        stream_name = parsed.path.rstrip("/").rsplit("/", 1)[-1] or parsed.netloc
        name = f"{parsed.netloc}_{stream_name}"
    else:
        name = Path(str(source)).stem or "source"
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)[:80] or "source"


def looks_like_local_video_path(source: str) -> bool:
    text = str(source).strip()
    lower = text.lower()
    if text.startswith((".", "/", "\\")):
        return True
    if len(text) >= 3 and text[1] == ":" and text[2] in {"\\", "/"}:
        return True
    return lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"))


def is_live_source(source: str) -> bool:
    text = str(source).strip()
    return text.isdigit() or ("://" in text and not looks_like_local_video_path(text))


def open_video_capture(source: str, low_latency: bool) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(parse_source(source))
    if low_latency:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def resize_frame_for_web(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = float(max_width) / float(width)
    target_size = (max_width, max(1, int(round(height * scale))))
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def placeholder_jpeg(text: str, jpeg_quality: int) -> bytes:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (2, 8, 6)
    cv2.rectangle(frame, (36, 36), (1244, 684), (26, 217, 255), 2)
    cv2.putText(frame, text, (72, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (56, 255, 156), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 40, 95))])
    return encoded.tobytes() if ok else b""


def normalize_stream_source(source: Any) -> str:
    text = str(source or "").strip().strip("\"'")
    if not text:
        raise ValueError("Source is empty.")
    if text.isdigit():
        return text

    lower = text.lower()
    stream_prefixes = ("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://")
    if lower.startswith(stream_prefixes) or "://" in text or looks_like_local_video_path(text):
        return text

    candidate = f"http://{text}"
    parsed = urlparse(candidate)
    if parsed.netloc and parsed.path in {"", "/"} and not parsed.query:
        return candidate.rstrip("/") + "/video"
    return candidate


class LatestFrameCapture:
    def __init__(self, source: str, low_latency: bool, logger: logging.Logger) -> None:
        self.source = str(source)
        self.logger = logger
        self.cap = open_video_capture(self.source, low_latency=low_latency)
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.thread: Optional[threading.Thread] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_seq = -1
        self.latest_time = time.time()

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def get(self, prop_id: int) -> float:
        return float(self.cap.get(prop_id) or 0.0)

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, name="latest-frame-capture", daemon=True)
        self.thread.start()

    def read_latest(self, last_seq: int, timeout: float = 2.0) -> tuple[bool, Optional[np.ndarray], int, float]:
        deadline = time.time() + timeout
        with self.condition:
            while self.latest_seq <= last_seq and not self.stop_event.is_set():
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=remaining)
            if self.latest_frame is None or self.latest_seq <= last_seq:
                return False, None, last_seq, time.time()
            return True, self.latest_frame, self.latest_seq, self.latest_time

    def release(self) -> None:
        self.stop_event.set()
        try:
            self.cap.release()
        except Exception:
            pass
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                ok, frame = self.cap.read()
            except Exception as exc:
                self.logger.debug("Live capture read failed: %s", exc)
                ok, frame = False, None

            if ok and frame is not None:
                with self.condition:
                    self.latest_frame = frame
                    self.latest_seq += 1
                    self.latest_time = time.time()
                    self.condition.notify_all()
            else:
                time.sleep(0.02)


class InferenceStream:
    def __init__(
        self,
        config: Dict[str, Any],
        source: str,
        checkpoint: Optional[str],
        jpeg_quality: int,
        logger: logging.Logger,
        camera_id: int = 0,
        alert_manager: Optional[AlertManager] = None,
    ) -> None:
        self.config = config
        self.source = str(source)
        self.checkpoint = checkpoint
        self.jpeg_quality = int(np.clip(jpeg_quality, 40, 95))
        self.logger = logger
        self.camera_id = int(camera_id)
        self.alert_manager = alert_manager
        self.lock = threading.RLock()
        self.frame_ready = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.latest_jpeg = self._placeholder("Đang khởi động camera...")
        self.status: Dict[str, Any] = {
            "running": False,
            "source": self.source,
            "people": 0,
            "fall_alert": False,
            "faint_alert": False,
            "timestamp_sec": 0.0,
            "processing_fps": 0.0,
            "tracks": [],
            "error": "",
            "updated_at": time.time(),
        }

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, name="fall-web-inference", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.frame_ready:
            self.frame_ready.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=5.0)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.status)

    def wait_for_frame(self, timeout: float = 2.0) -> bytes:
        with self.frame_ready:
            self.frame_ready.wait(timeout=timeout)
            return self.latest_jpeg

    def _placeholder(self, text: str) -> bytes:
        return placeholder_jpeg(text, self.jpeg_quality)

    def _set_error(self, message: str) -> None:
        self.logger.error(message)
        placeholder = self._placeholder(message[:80])
        with self.frame_ready:
            self.latest_jpeg = placeholder
            self.status.update(
                {
                    "running": False,
                    "people": 0,
                    "fall_alert": False,
                    "faint_alert": False,
                    "tracks": [],
                    "error": message,
                    "updated_at": time.time(),
                }
            )
            self.frame_ready.notify_all()

    def _publish(self, frame: np.ndarray, payload: Dict[str, Any]) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        with self.frame_ready:
            self.latest_jpeg = encoded.tobytes()
            self.status.update(payload)
            self.status["updated_at"] = time.time()
            self.frame_ready.notify_all()

    def _load_model(self, device: torch.device) -> Optional[torch.nn.Module]:
        checkpoint_value = self.checkpoint or str(self.config["inference"]["checkpoint"])
        checkpoint_path = resolve_path(checkpoint_value)
        if checkpoint_path.exists():
            model, payload = load_model_from_checkpoint(str(checkpoint_path), device, fallback_config=self.config)
            self.logger.info("Loaded checkpoint %s at epoch %s", checkpoint_path, payload.get("epoch", "unknown"))
            return model
        if bool(self.config["inference"].get("allow_heuristic_without_checkpoint", False)):
            self.logger.warning("Checkpoint not found: %s. Running heuristic-only inference.", checkpoint_path)
            return None
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    def _run(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        live_capture: Optional[LatestFrameCapture] = None
        annotated_writer: Optional[cv2.VideoWriter] = None
        event_logger: Optional[EventLogger] = None
        person_recorder: Optional[EventClipRecorder] = None
        fall_recorder: Optional[EventClipRecorder] = None
        faint_recorder: Optional[EventClipRecorder] = None
        try:
            ensure_project_dirs(self.config)
            detector_cfg = self.config["detector"]
            device = torch.device(select_device(str(detector_cfg["device"])))

            detector = YoloPersonPoseDetector(
                model_name=str(detector_cfg["model"]),
                confidence=float(detector_cfg["confidence"]),
                iou=float(detector_cfg["iou"]),
                image_size=int(detector_cfg["image_size"]),
                max_detections=int(detector_cfg["max_detections"]),
                device=str(detector_cfg["device"]),
                logger=self.logger,
            )
            model = self._load_model(device)

            tracker_cfg = self.config["tracker"]
            tracker = PersonTracker(
                iou_threshold=float(tracker_cfg["iou_threshold"]),
                max_center_distance=float(tracker_cfg["max_center_distance"]),
                max_missing_frames=int(tracker_cfg["max_missing_frames"]),
                min_hits=int(tracker_cfg["min_hits"]),
                smoothing_alpha=float(tracker_cfg["smoothing_alpha"]),
            )
            state_machine = FallStateMachine(self.config["fall_logic"])
            smoother = ProbabilitySmoother(window_size=5, alpha=0.70)
            sequence_length = int(self.config["features"]["sequence_length"])
            model_sample_fps = max(1.0, float(self.config["features"]["sample_fps"]))
            histories: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=sequence_length))
            last_decisions: Dict[int, FallDecision] = {}

            web_cfg = self.config.get("web", {})
            if not isinstance(web_cfg, dict):
                web_cfg = {}
            max_frame_width = int(web_cfg.get("max_frame_width", 960) or 0)
            web_save_outputs = bool(web_cfg.get("save_outputs", False))
            drop_stale_frames = bool(web_cfg.get("drop_stale_frames", True))
            live_source = is_live_source(self.source)

            if live_source and drop_stale_frames:
                live_capture = LatestFrameCapture(self.source, low_latency=True, logger=self.logger)
                if not live_capture.is_opened():
                    raise RuntimeError(f"Cannot open source: {self.source}")
                live_capture.start()
                fps = float(live_capture.get(cv2.CAP_PROP_FPS) or self.config["inference"]["output_fps"])
            else:
                cap = open_video_capture(self.source, low_latency=live_source)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open source: {self.source}")
                fps = float(cap.get(cv2.CAP_PROP_FPS) or self.config["inference"]["output_fps"])

            if fps <= 1.0 or fps > 120.0:
                fps = float(self.config["inference"]["output_fps"])
            temporal_stride = max(1, int(round(fps / model_sample_fps)))
            self.logger.info(
                "Web inference source=%s fps=%.2f stride=%d live=%s drop_stale=%s max_width=%d",
                self.source,
                fps,
                temporal_stride,
                live_source,
                bool(live_capture is not None),
                max_frame_width,
            )

            output_root = resolve_path(self.config["paths"]["outputs_dir"])
            src_name = source_name(self.source)
            timestamp = safe_timestamp()
            frame_size: Optional[tuple[int, int]] = None
            pre_frames = int(round(float(self.config["inference"]["pre_event_seconds"]) * fps))
            post_frames = int(round(float(self.config["inference"]["post_event_seconds"]) * fps))
            no_person_post_frames = int(round(float(self.config["inference"]["clip_no_person_timeout_sec"]) * fps))

            previous_sample_gray: Optional[np.ndarray] = None
            frame_index = 0
            last_capture_seq = -1
            stream_start_time = time.time()
            last_temporal_timestamp = -1.0 / model_sample_fps
            last_publish_time = time.time()

            while not self.stop_event.is_set():
                if live_capture is not None:
                    ok, frame, capture_seq, capture_time = live_capture.read_latest(last_capture_seq, timeout=2.0)
                    if not ok or frame is None:
                        time.sleep(0.05)
                        continue
                    last_capture_seq = capture_seq
                    timestamp_sec = max(0.0, capture_time - stream_start_time)
                else:
                    if cap is None:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        if str(self.source).isdigit():
                            time.sleep(0.05)
                            continue
                        break
                    timestamp_sec = float(frame_index / fps)

                frame = resize_frame_for_web(frame, max_frame_width)

                if frame_size is None:
                    frame_h, frame_w = frame.shape[:2]
                    frame_size = (frame_w, frame_h)
                    if web_save_outputs and bool(self.config["inference"]["save_annotated"]):
                        try:
                            annotated_writer = create_video_writer(
                                make_unique_path(output_root / f"annotated_{src_name}_{timestamp}.mp4"),
                                fps,
                                frame_size,
                            )
                        except RuntimeError as exc:
                            self.logger.warning("Disabling annotated video recording: %s", exc)
                            annotated_writer = None
                    if bool(self.config["inference"]["save_event_log"]):
                        try:
                            event_logger = EventLogger(output_root / "events" / f"events_{src_name}_{timestamp}.csv")
                        except OSError as exc:
                            self.logger.warning("Disabling event log for this stream: %s", exc)
                            event_logger = None
                    if web_save_outputs and bool(self.config["inference"]["save_person_clips"]):
                        person_recorder = EventClipRecorder(
                            output_root / "person_clips",
                            "person",
                            fps,
                            frame_size,
                            0,
                            no_person_post_frames,
                            src_name,
                        )
                    if web_save_outputs and bool(self.config["inference"]["save_fall_clips"]):
                        fall_recorder = EventClipRecorder(
                            output_root / "fall_clips",
                            "fall",
                            fps,
                            frame_size,
                            pre_frames,
                            post_frames,
                            src_name,
                        )
                    if web_save_outputs and bool(self.config["inference"].get("save_faint_clips", self.config["inference"].get("save_unconscious_clips", True))):
                        faint_recorder = EventClipRecorder(
                            output_root / "faint_clips",
                            "faint",
                            fps,
                            frame_size,
                            pre_frames,
                            post_frames,
                            src_name,
                        )
                    if web_save_outputs:
                        self.logger.info("Web recording outputs enabled under %s", output_root)
                    elif event_logger is not None:
                        self.logger.info("Web video recording disabled; event log enabled under %s", output_root)
                    else:
                        self.logger.info("Web outputs disabled for realtime streaming")

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                observations = detector.detect(frame, timestamp_sec)
                tracks = tracker.update(observations, frame_index)
                active_state_ids = {track.track_id for track in tracker.all_tracks()}
                state_machine.remove_missing_tracks(active_state_ids)
                smoother.keep_only(active_state_ids)
                for stale_track_id in list(last_decisions.keys()):
                    if stale_track_id not in active_state_ids:
                        del last_decisions[stale_track_id]

                annotated = frame.copy()
                any_fall = False
                any_faint = False
                tracks_payload = []
                danger_tracks = []
                if live_source:
                    should_update_temporal = (timestamp_sec - last_temporal_timestamp) >= (1.0 / model_sample_fps)
                else:
                    should_update_temporal = frame_index % temporal_stride == 0

                for track in tracks:
                    if track.observation is None:
                        continue
                    if should_update_temporal:
                        motion = compute_motion_inside_box(previous_sample_gray, gray, track.observation.bbox_xyxy)
                        feature = compute_feature_vector(track.observation, track.last_feature, motion)
                        track.last_feature = feature
                        histories[track.track_id].append(feature)
                        probabilities = predict_probabilities(
                            model=model,
                            history=histories[track.track_id],
                            sequence_length=sequence_length,
                            device=device,
                            feature=feature,
                            fall_logic_cfg=self.config["fall_logic"],
                        )
                        probabilities = smoother.update(track.track_id, probabilities)
                        decision = state_machine.update(track.track_id, feature, probabilities, timestamp_sec)
                        last_decisions[track.track_id] = decision
                        if event_logger is not None:
                            event_logger.log_if_changed(
                                timestamp_sec=timestamp_sec,
                                frame_index=frame_index,
                                track_id=track.track_id,
                                status=decision.status,
                                raw_label=decision.raw_label,
                                model_confidence=decision.model_confidence,
                                fall_score=decision.fall_score,
                                reason=decision.reason,
                            )
                    else:
                        decision = last_decisions.get(track.track_id)
                        if decision is None:
                            continue

                    is_danger = decision.status in DANGER_STATUSES
                    any_fall = any_fall or is_danger
                    any_faint = any_faint or decision.is_unconscious_alert
                    if is_danger:
                        danger_tracks.append(
                            {
                                "track_id": int(track.track_id),
                                "status": decision.status,
                                "confidence": float(decision.model_confidence),
                                "fall_score": float(decision.fall_score),
                                "reason": decision.reason,
                            }
                        )
                    draw_track_colored(
                        annotated,
                        track.bbox_xyxy,
                        track.track_id,
                        decision.status,
                        decision.model_confidence,
                        decision.fall_score,
                    )
                    tracks_payload.append(
                        {
                            "track_id": track.track_id,
                            "status": decision.status,
                            "status_vi": status_to_vi(decision.status),
                            "raw_label": decision.raw_label,
                            "confidence": round(float(decision.model_confidence), 4),
                            "fall_score": round(float(decision.fall_score), 4),
                            "reason": decision.reason,
                        }
                    )

                header = f"NGUOI={len(tracks)} NGA={int(any_fall)} NGAT={int(any_faint)} t={timestamp_sec:.1f}s"
                draw_header(annotated, header)

                if self.alert_manager is not None:
                    for danger in danger_tracks:
                        self.alert_manager.handle_detection(
                            camera_id=self.camera_id,
                            track_id=int(danger["track_id"]),
                            status=str(danger["status"]),
                            confidence=float(danger["confidence"]),
                            fall_score=float(danger["fall_score"]),
                            reason=str(danger["reason"]),
                            frame=annotated,
                            timestamp_sec=timestamp_sec,
                        )

                if annotated_writer is not None:
                    annotated_writer.write(annotated)
                if person_recorder is not None:
                    person_recorder.update(annotated, len(tracks) > 0, frame_index)
                if fall_recorder is not None:
                    fall_recorder.update(annotated, any_fall, frame_index)
                if faint_recorder is not None:
                    faint_recorder.update(annotated, any_faint, frame_index)

                now = time.time()
                elapsed = max(now - last_publish_time, 1e-6)
                processing_fps = 1.0 / elapsed
                last_publish_time = now
                payload = {
                    "running": True,
                    "source": self.source,
                    "people": len(tracks),
                    "fall_alert": bool(any_fall),
                    "faint_alert": bool(any_faint),
                    "timestamp_sec": round(timestamp_sec, 3),
                    "processing_fps": round(processing_fps, 2),
                    "tracks": tracks_payload,
                    "error": "",
                }
                self._publish(annotated, payload)

                if should_update_temporal:
                    previous_sample_gray = gray
                    last_temporal_timestamp = timestamp_sec
                frame_index += 1
        except Exception as exc:
            self._set_error(str(exc))
        finally:
            if live_capture is not None:
                live_capture.release()
            if cap is not None:
                cap.release()
            if annotated_writer is not None:
                annotated_writer.release()
            if event_logger is not None:
                event_logger.close()
            if person_recorder is not None:
                person_recorder.close()
            if fall_recorder is not None:
                fall_recorder.close()
            if faint_recorder is not None:
                faint_recorder.close()
            with self.frame_ready:
                self.status["running"] = False
                self.status["updated_at"] = time.time()
                self.frame_ready.notify_all()


class FallWebHandler(BaseHTTPRequestHandler):
    server_version = "FallDetectionWeb/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if self._is_family_host() and parsed.path in {"/", "/family"}:
            self._send_html(FAMILY_HTML)
            return

        if parsed.path in {"/", "/login"}:
            session = self._current_session()
            if self._session_verified(session):
                self._redirect("/dashboard")
            elif self._session_logged_in(session):
                self._redirect("/face_auth")
            else:
                self._send_html(LOGIN_HTML)
        elif parsed.path == "/face_auth":
            session = self._current_session()
            if self._session_verified(session):
                self._redirect("/dashboard")
            elif self._session_logged_in(session):
                self._send_html(FACE_AUTH_HTML)
            else:
                self._redirect("/login")
        elif parsed.path == "/auth/logout":
            self._logout()
        elif parsed.path == "/dashboard":
            if not self._session_verified(self._current_session()):
                self._redirect("/login")
                return
            self._send_html(INDEX_HTML)
        elif parsed.path == "/family":
            if not self._require_family_or_verified(parsed):
                self.send_error(HTTPStatus.FORBIDDEN, "Invalid family access link.")
                return
            self._send_html(FAMILY_HTML)
        elif parsed.path == "/family_config":
            if not self._require_family_or_verified(parsed):
                return
            self._send_json(self.server.family_config())  # type: ignore[attr-defined]
        elif parsed.path == "/videos":
            if not self._require_family_or_verified(parsed):
                return
            self._send_json(self.server.list_output_videos())  # type: ignore[attr-defined]
        elif parsed.path == "/alerts":
            if not self._require_family_or_verified(parsed):
                return
            alerts = self.server.alert_manager.list_records()  # type: ignore[attr-defined]
            enriched_alerts = []
            for item in alerts:
                alert_item = dict(item)
                alert_item["camera_label"] = self.server.camera_label(int(alert_item.get("camera_id") or 0))  # type: ignore[attr-defined]
                enriched_alerts.append(alert_item)
            self._send_json({"ok": True, "alerts": enriched_alerts})
        elif parsed.path == "/alert_image":
            if not self._require_family_or_verified(parsed):
                return
            self._serve_alert_image(parsed)
        elif parsed.path == "/replay_video":
            if not self._require_family_or_verified(parsed):
                return
            self._stream_saved_video(parsed)
        elif parsed.path == "/ack_alert":
            ok, message = self.server.alert_manager.acknowledge_from_query(parse_qs(parsed.query))  # type: ignore[attr-defined]
            self._send_html(self.server.alert_manager.render_ack_html(ok, message))  # type: ignore[attr-defined]
        elif parsed.path == "/output_video":
            if not self._require_family_or_verified(parsed):
                return
            self._serve_output_video(parsed)
        elif parsed.path == "/status":
            if not self._require_family_or_verified(parsed):
                return
            try:
                query = parse_qs(parsed.query)
                if {"camera", "camera_id", "id"} & set(query.keys()):
                    camera_id = self._camera_id(query=query)
                    self._send_json(self.server.snapshot_camera(camera_id))  # type: ignore[attr-defined]
                else:
                    self._send_json(self.server.snapshot_all())  # type: ignore[attr-defined]
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        elif parsed.path == "/video_feed":
            if not self._require_family_or_verified(parsed):
                return
            try:
                camera_id = self._camera_id(query=parse_qs(parsed.query))
            except ValueError as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._stream_video(camera_id)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/auth/login":
            self._login()
        elif parsed.path == "/auth/face":
            self._verify_face()
        elif parsed.path == "/source":
            if not self._require_verified():
                return
            self._change_source(parsed)
        elif parsed.path == "/demo_alert":
            if not self._require_verified():
                return
            self._create_demo_alert()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        self.server.logger.debug("%s - %s", self.address_string(), format % args)  # type: ignore[attr-defined]

    def _cookie_token(self) -> str:
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        morsel = cookie.get(AUTH_COOKIE_NAME)
        return morsel.value if morsel is not None else ""

    def _current_session(self) -> Optional[Dict[str, Any]]:
        return self.server.get_auth_session(self._cookie_token())  # type: ignore[attr-defined]

    def _session_logged_in(self, session: Optional[Dict[str, Any]]) -> bool:
        return bool(session and session.get("login_ok"))

    def _session_verified(self, session: Optional[Dict[str, Any]]) -> bool:
        return bool(session and session.get("login_ok") and session.get("face_ok"))

    def _auth_cookie_header(self, token: str, max_age: int = AUTH_SESSION_TTL_SECONDS) -> str:
        return f"{AUTH_COOKIE_NAME}={token}; Path=/; Max-Age={max_age}; SameSite=Lax; HttpOnly"

    def _require_verified(self) -> bool:
        if self._session_verified(self._current_session()):
            return True
        self._send_json({"ok": False, "error": "Bạn cần đăng nhập và quét mặt trước."}, HTTPStatus.UNAUTHORIZED)
        return False

    def _is_family_host(self) -> bool:
        host = self.headers.get("Host", "").split(":", 1)[0].lower()
        return host == "family.khanhtoan.click"

    def _family_token_valid(self, parsed: Optional[Any] = None) -> bool:
        expected = str(getattr(self.server, "family_access_token", "") or "")  # type: ignore[attr-defined]
        if not expected:
            return False
        query = parse_qs((parsed or urlparse(self.path)).query)
        token_values = query.get("token") or query.get("family_token") or []
        token = str(token_values[-1] if token_values else "")
        return bool(token) and hmac.compare_digest(token, expected)

    def _require_family_or_verified(self, parsed: Optional[Any] = None) -> bool:
        if self._is_family_host() or self._session_verified(self._current_session()) or self._family_token_valid(parsed):
            return True
        self._send_json({"ok": False, "error": "Link nguoi than khong hop le."}, HTTPStatus.UNAUTHORIZED)
        return False

    def _login(self) -> None:
        try:
            payload = self._read_payload()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        username = str(payload.get("username", "") or "").strip()
        password = str(payload.get("password", "") or "")
        if not self.server.check_login(username, password):  # type: ignore[attr-defined]
            self._send_json({"ok": False, "error": "Sai tài khoản hoặc mật khẩu."}, HTTPStatus.UNAUTHORIZED)
            return

        token = self.server.create_auth_session(username)  # type: ignore[attr-defined]
        self._send_json(
            {"ok": True, "next": "/face_auth"},
            headers={"Set-Cookie": self._auth_cookie_header(token)},
        )

    def _verify_face(self) -> None:
        session = self._current_session()
        if not self._session_logged_in(session):
            self._send_json({"ok": False, "error": "Bạn cần đăng nhập trước."}, HTTPStatus.UNAUTHORIZED)
            return

        try:
            payload = self._read_payload()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        result = self.server.verify_face_image(str(payload.get("image", "") or ""))  # type: ignore[attr-defined]
        if not result.get("ok"):
            status = HTTPStatus.BAD_REQUEST if result.get("code") != "missing_dependency" else HTTPStatus.SERVICE_UNAVAILABLE
            self._send_json(result, status)
            return

        token = self._cookie_token()
        self.server.mark_face_verified(token, result.get("match") or {})  # type: ignore[attr-defined]
        self._send_json({"ok": True, "next": "/dashboard", "match": result.get("match") or {}})

    def _logout(self) -> None:
        token = self._cookie_token()
        if token:
            self.server.destroy_auth_session(token)  # type: ignore[attr-defined]
        self._redirect("/login", headers={"Set-Cookie": self._auth_cookie_header("", max_age=0)})

    def _send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(
        self,
        payload: Dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str, headers: Optional[Dict[str, str]] = None) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _read_payload(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length.") from exc
        if length <= 0:
            return {}
        if length > 6 * 1024 * 1024:
            raise ValueError("Request body is too large.")

        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type == "application/json":
            payload = json.loads(raw or "{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object.")
            return payload
        if content_type == "application/x-www-form-urlencoded":
            parsed = parse_qs(raw, keep_blank_values=True)
            return {key: values[-1] if values else "" for key, values in parsed.items()}

        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        return {"source": raw.strip()}

    def _camera_id(
        self,
        payload: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, list[str]]] = None,
    ) -> int:
        raw_value: Any = None
        if payload is not None:
            raw_value = payload.get("camera_id", payload.get("camera", payload.get("id")))
        if (raw_value is None or raw_value == "") and query is not None:
            values = query.get("camera") or query.get("camera_id") or query.get("id")
            if values:
                raw_value = values[-1]
        if raw_value is None or raw_value == "":
            return 0
        try:
            camera_id = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Camera id must be a number.") from exc
        camera_count = int(self.server.camera_count)  # type: ignore[attr-defined]
        if camera_id < 0 or camera_id >= camera_count:
            raise ValueError(f"Camera id must be between 0 and {camera_count - 1}.")
        return camera_id

    def _create_demo_alert(self) -> None:
        try:
            payload = self._read_payload()
            camera_id = self._camera_id(payload=payload)
            status = str(payload.get("status", "FALLEN") or "FALLEN").strip().upper()
            if status not in DANGER_STATUSES:
                raise ValueError("Demo status must be FALLING, FALLEN, FAINT, or UNCONSCIOUS.")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        frame = self._make_demo_alert_frame(camera_id=camera_id, status=status)
        alert_id = self.server.alert_manager.handle_detection(  # type: ignore[attr-defined]
            camera_id=camera_id,
            track_id=999,
            status=status,
            confidence=0.91,
            fall_score=0.88,
            reason="demo_alert",
            frame=frame,
            timestamp_sec=0.0,
            force=True,
        )

        if not alert_id:
            self._send_json({"ok": False, "error": "Could not create demo alert."}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(
            {
                "ok": True,
                "alert_id": alert_id,
                "camera_id": camera_id,
                "status": status,
                "message": "Demo alert created.",
            }
        )

    def _make_demo_alert_frame(self, camera_id: int, status: str) -> np.ndarray:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (8, 18, 16)

        color = status_to_color(status)
        cv2.rectangle(frame, (46, 46), (1234, 674), color, 5)
        cv2.rectangle(frame, (86, 128), (1194, 590), (18, 34, 30), -1)
        cv2.rectangle(frame, (86, 128), (1194, 590), color, 2)

        lines = [
            "DEMO CANH BAO NGA",
            f"CAM {camera_id + 1:02d} | ID 999 | {status_to_frame_text(status)}",
            "Tin cậy: 0.91 | Điểm ngã: 0.88",
            time.strftime("Thoi gian: %Y-%m-%d %H:%M:%S"),
            "Nguon: Che do demo do an",
        ]

        y = 220
        for index, text in enumerate(lines):
            scale = 1.45 if index == 0 else 0.92
            thickness = 3 if index == 0 else 2
            text_color = color if index == 0 else (245, 255, 255)
            cv2.putText(
                frame,
                text,
                (130, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )
            y += 82 if index == 0 else 58

        return frame

    def _change_source(self, parsed: Any) -> None:
        try:
            payload = self._read_payload()
            camera_id = self._camera_id(payload=payload, query=parse_qs(parsed.query))
            source = normalize_stream_source(payload.get("source", ""))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        try:
            selected_source = self.server.switch_source(camera_id, source)  # type: ignore[attr-defined]
        except Exception as exc:
            self.server.logger.exception("Failed to switch source")  # type: ignore[attr-defined]
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json({"ok": True, "camera_id": camera_id, "source": selected_source})

    def _serve_alert_image(self, parsed: Any) -> None:
        query = parse_qs(parsed.query)
        values = query.get("id") or query.get("alert_id")
        if not values:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing alert id.")
            return

        alert_id = str(values[-1]).strip()
        record = self.server.alert_manager.get_record(alert_id)  # type: ignore[attr-defined]
        if record is None or record.image_path is None or not record.image_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Alert image not found.")
            return

        image_path = record.image_path.resolve()
        alert_root = self.server.alert_manager.alert_dir.resolve()  # type: ignore[attr-defined]
        try:
            image_path.relative_to(alert_root)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Invalid alert image path.")
            return

        payload = image_path.read_bytes()
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _stream_saved_video(self, parsed: Any) -> None:
        query = parse_qs(parsed.query)
        values = query.get("path") or query.get("file")
        if not values:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing video path.")
            return

        try:
            video_path = self.server.resolve_output_video(values[-1])  # type: ignore[attr-defined]
        except ValueError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            return

        if not is_probably_ready_video(video_path):
            self.send_error(HTTPStatus.CONFLICT, "Video is still being recorded or is not ready yet.")
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Cannot open saved video for replay.")
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
        if fps <= 1.0 or fps > 60.0:
            fps = 15.0
        delay = min(max(1.0 / fps, 0.015), 0.15)
        jpeg_quality = int(np.clip(getattr(self.server, "jpeg_quality", 82), 40, 95))  # type: ignore[attr-defined]

        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if not ok:
                    continue
                payload = encoded.tobytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return
        finally:
            cap.release()


    def _serve_output_video(self, parsed: Any) -> None:
        query = parse_qs(parsed.query)
        values = query.get("path") or query.get("file")
        if not values:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing video path.")
            return

        try:
            video_path = self.server.prepare_output_video_for_web(values[-1])  # type: ignore[attr-defined]
        except ValueError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            return
        except RuntimeError as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return

        file_size = video_path.stat().st_size
        start = 0
        end = max(0, file_size - 1)
        status = HTTPStatus.OK
        range_header = self.headers.get("Range", "").strip()
        if range_header:
            try:
                start, end = self._parse_range(range_header, file_size)
                status = HTTPStatus.PARTIAL_CONTENT
            except ValueError:
                self._send_range_not_satisfiable(file_size)
                return

        content_length = max(0, end - start + 1) if file_size > 0 else 0
        mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"

        self.send_response(status)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Disposition", f'inline; filename="{video_path.name}"')
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        try:
            with video_path.open("rb") as file:
                file.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = file.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return

    def _parse_range(self, range_header: str, file_size: int) -> tuple[int, int]:
        if file_size <= 0 or not range_header.startswith("bytes="):
            raise ValueError("Invalid range.")
        range_value = range_header.split("=", 1)[1].split(",", 1)[0].strip()
        if "-" not in range_value:
            raise ValueError("Invalid range.")
        raw_start, raw_end = range_value.split("-", 1)
        if raw_start == "":
            suffix_length = int(raw_end)
            if suffix_length <= 0:
                raise ValueError("Invalid range.")
            start = max(0, file_size - suffix_length)
            end = file_size - 1
        else:
            start = int(raw_start)
            end = int(raw_end) if raw_end else file_size - 1
        end = min(end, file_size - 1)
        if start < 0 or end < start or start >= file_size:
            raise ValueError("Invalid range.")
        return start, end

    def _send_range_not_satisfiable(self, file_size: int) -> None:
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header("Content-Range", f"bytes */{file_size}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _stream_video(self, camera_id: int) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                worker = self.server.get_worker(camera_id)  # type: ignore[attr-defined]
                if worker is None:
                    frame = self.server.placeholder_frame(camera_id)  # type: ignore[attr-defined]
                    sleep_after_frame = 0.8
                elif worker.stop_event.is_set():
                    frame = self.server.placeholder_frame(camera_id)  # type: ignore[attr-defined]
                    sleep_after_frame = 0.2
                else:
                    frame = worker.wait_for_frame(timeout=2.0)
                    sleep_after_frame = 0.0
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                if sleep_after_frame > 0.0:
                    time.sleep(sleep_after_frame)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return


class FallThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        workers: Dict[int, InferenceStream],
        config: Dict[str, Any],
        checkpoint: Optional[str],
        jpeg_quality: int,
        logger: logging.Logger,
        camera_count: int,
        alert_manager: AlertManager,
        face_authenticator: FaceAuthenticator,
        auth_username: str,
        auth_password: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.workers = dict(workers)
        self.config = config
        self.checkpoint = checkpoint
        self.jpeg_quality = int(np.clip(jpeg_quality, 40, 95))
        self.logger = logger
        self.alert_manager = alert_manager
        self.face_authenticator = face_authenticator
        self.auth_username = str(auth_username or "admin")
        self.auth_password = str(auth_password or "123456")
        self.family_access_token = str(os.environ.get("FAMILY_ACCESS_TOKEN") or os.environ.get("ALERT_ACK_SECRET") or "").strip()
        self.auth_sessions: Dict[str, Dict[str, Any]] = {}
        self.auth_lock = threading.RLock()
        self.camera_count = max(1, int(camera_count))
        self.worker_lock = threading.RLock()
        self._idle_frames: Dict[int, bytes] = {}
        self.output_root = resolve_path(str(config["paths"]["outputs_dir"])).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.web_video_cache_root = self.output_root / WEB_VIDEO_CACHE_DIRNAME
        self.web_video_cache_root.mkdir(parents=True, exist_ok=True)
        self.video_cache_lock = threading.RLock()

    def family_care_phone(self) -> str:
        for name in ("FAMILY_CARE_PHONE", "CARE_PHONE", "STAFF_CARE_PHONE", "NURSING_CARE_PHONE", "HOTLINE_PHONE"):
            value = str(os.environ.get(name, "") or "").strip()
            if value:
                return value
        return "09xx xxx xxx"

    def camera_label(self, camera_id: int) -> str:
        index = int(camera_id)
        value = str(os.environ.get(f"CAMERA_{index}_LABEL", "") or "").strip()
        return value or f"Camera {index + 1}"

    def family_config(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "care_phone": self.family_care_phone(),
            "camera_labels": {str(camera_id): self.camera_label(camera_id) for camera_id in range(self.camera_count)},
        }

    def _prune_auth_sessions(self) -> None:
        now = time.time()
        expired = [
            token
            for token, session in self.auth_sessions.items()
            if now - float(session.get("updated_at", session.get("created_at", now))) > AUTH_SESSION_TTL_SECONDS
        ]
        for token in expired:
            self.auth_sessions.pop(token, None)

    def check_login(self, username: str, password: str) -> bool:
        return hmac.compare_digest(str(username), self.auth_username) and hmac.compare_digest(str(password), self.auth_password)

    def create_auth_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self.auth_lock:
            self._prune_auth_sessions()
            self.auth_sessions[token] = {
                "login_ok": True,
                "face_ok": False,
                "username": username,
                "created_at": now,
                "updated_at": now,
            }
        return token

    def get_auth_session(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        with self.auth_lock:
            self._prune_auth_sessions()
            session = self.auth_sessions.get(token)
            if session is not None:
                session["updated_at"] = time.time()
                return dict(session)
        return None

    def mark_face_verified(self, token: str, match: Dict[str, Any]) -> None:
        if not token:
            return
        with self.auth_lock:
            session = self.auth_sessions.get(token)
            if session is not None:
                session["face_ok"] = True
                session["face_match"] = match
                session["updated_at"] = time.time()

    def destroy_auth_session(self, token: str) -> None:
        with self.auth_lock:
            self.auth_sessions.pop(token, None)

    def verify_face_image(self, image_data: str) -> Dict[str, Any]:
        return self.face_authenticator.verify_image_data(image_data)

    def get_worker(self, camera_id: int = 0) -> Optional[InferenceStream]:
        with self.worker_lock:
            return self.workers.get(camera_id)

    def placeholder_frame(self, camera_id: int) -> bytes:
        with self.worker_lock:
            if camera_id not in self._idle_frames:
                self._idle_frames[camera_id] = placeholder_jpeg(f"{self.camera_label(camera_id)}: dang cho nguon", self.jpeg_quality)
            return self._idle_frames[camera_id]

    def list_output_videos(self) -> Dict[str, Any]:
        videos = []
        for path in self.output_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS or path.name.startswith("."):
                continue
            try:
                relative = path.resolve().relative_to(self.output_root)
            except ValueError:
                continue
            if relative.parts and relative.parts[0] == WEB_VIDEO_CACHE_DIRNAME:
                continue
            try:
                resolved = path.resolve()
                stat = resolved.stat()
            except (OSError, ValueError):
                continue
            if not is_probably_ready_video(resolved):
                # Do not ask FFmpeg/OpenCV to read unfinished MP4 files; this
                # prevents repeated "moov atom not found" messages in terminal.
                continue
            ok, metadata, _ = get_video_metadata(resolved)
            if not ok:
                continue
            videos.append(
                {
                    "path": relative.as_posix(),
                    "name": resolved.name,
                    "folder": "" if relative.parent == Path(".") else relative.parent.as_posix(),
                    "size_bytes": int(stat.st_size),
                    "modified_at": float(stat.st_mtime),
                    "duration_sec": round(float(metadata.get("duration_sec") or 0.0), 3),
                }
            )
        videos.sort(key=lambda item: item["modified_at"], reverse=True)
        return {"ok": True, "count": len(videos), "videos": videos}

    def resolve_output_video(self, relative_path: str) -> Path:
        text = str(relative_path or "").strip().replace("\\", "/")
        if not text or text.startswith("/") or "://" in text:
            raise ValueError("Invalid video path.")
        candidate = (self.output_root / text).resolve()
        try:
            candidate.relative_to(self.output_root)
        except ValueError as exc:
            raise ValueError("Invalid video path.") from exc
        if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("Video not found.")
        return candidate

    def prepare_output_video_for_web(self, relative_path: str) -> Path:
        source_path = self.resolve_output_video(relative_path)
        if not is_probably_ready_video(source_path):
            raise RuntimeError(f"Video is not ready for playback yet: {source_path.name}")
        ok, _, error = get_video_metadata(source_path)
        if not ok:
            raise RuntimeError(f"Video is not ready for playback yet: {error or source_path.name}")
        if is_browser_playable_video(source_path):
            return source_path

        relative = source_path.relative_to(self.output_root)
        with self.video_cache_lock:
            cached_mp4 = self.web_video_cache_root / relative.with_suffix(".mp4")
            if self._is_cache_fresh(source_path, cached_mp4) and is_browser_playable_video(cached_mp4):
                return cached_mp4

            if cached_mp4.exists():
                cached_mp4.unlink(missing_ok=True)
            try:
                transcode_video(source_path, cached_mp4, codec_candidates=WEB_MP4_CODECS)
                if is_browser_playable_video(cached_mp4):
                    return cached_mp4
            except Exception as exc:
                self.logger.warning("Failed to prepare MP4 web cache for %s: %s", source_path, exc)
                cached_mp4.unlink(missing_ok=True)

            raise RuntimeError(
                f"Cannot prepare {source_path.name} for browser playback. "
                "MP4/mp4v transcode failed; WebM fallback is disabled to avoid VP90/VP80 OpenCV errors."
            )

    def _is_cache_fresh(self, source_path: Path, cached_path: Path) -> bool:
        if not cached_path.exists():
            return False
        try:
            if cached_path.stat().st_mtime < source_path.stat().st_mtime or cached_path.stat().st_size <= 0:
                return False
        except OSError:
            return False
        if not is_probably_ready_video(cached_path):
            return False
        ok, _, _ = get_video_metadata(cached_path)
        return ok

    def idle_snapshot(self, camera_id: int) -> Dict[str, Any]:
        return {
            "camera_id": camera_id,
            "camera_label": self.camera_label(camera_id),
            "running": False,
            "source": "",
            "people": 0,
            "fall_alert": False,
            "faint_alert": False,
            "timestamp_sec": 0.0,
            "processing_fps": 0.0,
            "tracks": [],
            "error": "Chưa cấu hình nguồn camera.",
            "updated_at": time.time(),
        }

    def snapshot_camera(self, camera_id: int) -> Dict[str, Any]:
        with self.worker_lock:
            worker = self.workers.get(camera_id)
        if worker is None:
            return self.idle_snapshot(camera_id)
        payload = worker.snapshot()
        payload["camera_id"] = camera_id
        payload["camera_label"] = self.camera_label(camera_id)
        return payload

    def snapshot_all(self) -> Dict[str, Any]:
        cameras = [self.snapshot_camera(camera_id) for camera_id in range(self.camera_count)]
        active = [camera for camera in cameras if camera.get("running")]
        fps_values = [float(camera.get("processing_fps") or 0.0) for camera in active if float(camera.get("processing_fps") or 0.0) > 0.0]
        return {
            "camera_count": self.camera_count,
            "cameras": cameras,
            "running": bool(active),
            "active_cameras": len(active),
            "people": sum(int(camera.get("people") or 0) for camera in cameras),
            "fall_alert": any(bool(camera.get("fall_alert")) for camera in cameras),
            "faint_alert": any(bool(camera.get("faint_alert")) for camera in cameras),
            "processing_fps": round(sum(fps_values) / len(fps_values), 2) if fps_values else 0.0,
            "updated_at": time.time(),
        }

    def switch_source(self, camera_id: int, source: str) -> str:
        selected_source = normalize_stream_source(source)
        with self.worker_lock:
            old_worker = self.workers.get(camera_id)
            if (
                old_worker is not None
                and old_worker.source == selected_source
                and old_worker.thread is not None
                and old_worker.thread.is_alive()
            ):
                return selected_source

            old_source = old_worker.source if old_worker is not None else "idle"
            self.logger.info("Switching camera %d source from %s to %s", camera_id + 1, old_source, selected_source)
            if old_worker is not None:
                old_worker.stop()
            new_worker = InferenceStream(
                config=self.config,
                source=selected_source,
                checkpoint=self.checkpoint,
                jpeg_quality=self.jpeg_quality,
                logger=self.logger,
                camera_id=camera_id,
                alert_manager=self.alert_manager,
            )
            self.workers[camera_id] = new_worker
            new_worker.start()
        return selected_source

    def stop_all(self) -> None:
        with self.worker_lock:
            workers = list(self.workers.values())
        for worker in workers:
            worker.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fall detection web server.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--source", default=None, help="Webcam index or video path. Default comes from config.")
    parser.add_argument("--checkpoint", default=None, help="Path to trained checkpoint.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    parser.add_argument("--jpeg-quality", type=int, default=82, help="MJPEG JPEG quality, 40-95.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logging(config, "web_server")
    alert_manager = AlertManager.from_env(output_root=resolve_path(config["paths"]["outputs_dir"]), logger=logger)
    face_authenticator = FaceAuthenticator.from_env(logger=logger)
    auth_username = os.environ.get("AUTH_USERNAME", "admin")
    auth_password = os.environ.get("AUTH_PASSWORD", "123456")

    web_cfg = config.get("web", {})
    if not isinstance(web_cfg, dict):
        web_cfg = {}
    camera_count = max(1, int(web_cfg.get("camera_count", 4) or 4))
    configured_sources = web_cfg.get("sources", [])
    if not isinstance(configured_sources, list):
        configured_sources = []

    workers: Dict[int, InferenceStream] = {}
    for camera_id, configured_source in enumerate(configured_sources[:camera_count]):
        source_text = str(configured_source or "").strip()
        if not source_text:
            continue
        workers[camera_id] = InferenceStream(
            config=config,
            source=source_text,
            checkpoint=args.checkpoint,
            jpeg_quality=args.jpeg_quality,
            logger=logger,
            camera_id=camera_id,
            alert_manager=alert_manager,
        )

    source = str(args.source if args.source is not None else config["inference"]["source"]).strip()
    if source and 0 not in workers:
        workers[0] = InferenceStream(
            config=config,
            source=source,
            checkpoint=args.checkpoint,
            jpeg_quality=args.jpeg_quality,
            logger=logger,
            camera_id=0,
            alert_manager=alert_manager,
        )

    for worker in workers.values():
        worker.start()

    server = FallThreadingHTTPServer(
        (str(args.host), int(args.port)),
        FallWebHandler,
        workers=workers,
        config=config,
        checkpoint=args.checkpoint,
        jpeg_quality=args.jpeg_quality,
        logger=logger,
        camera_count=camera_count,
        alert_manager=alert_manager,
        face_authenticator=face_authenticator,
        auth_username=auth_username,
        auth_password=auth_password,
    )
    url = f"http://{args.host}:{args.port}"
    logger.info("Web server running at %s with %d camera slots", url, camera_count)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        logger.info("Stopping web server.")
    finally:
        server.stop_all()
        server.server_close()


if __name__ == "__main__":
    main()

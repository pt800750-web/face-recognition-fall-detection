from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Optional

import cv2
import numpy as np
import torch

from src.clip_writer import EventClipRecorder, EventLogger
from src.fall_logic import FallDecision, FallStateMachine
from src.features import (
    LABEL_TO_ID,
    YoloPersonPoseDetector,
    compute_feature_vector,
    compute_motion_inside_box,
)
from src.models import load_model_from_checkpoint
from src.postprocess import ProbabilitySmoother
from src.tracker import PersonTracker
from src.utils import (
    build_arg_parser,
    create_video_writer,
    ensure_project_dirs,
    load_config,
    make_unique_path,
    parse_source,
    resolve_path,
    safe_timestamp,
    select_device,
    setup_logging,
)
from src.visualization import draw_header, draw_track


def heuristic_probabilities(feature: np.ndarray, fall_logic_cfg: Dict[str, float]) -> np.ndarray:
    probs = np.zeros((len(LABEL_TO_ID),), dtype=np.float32)
    aspect = float(feature[4])
    height = float(feature[3])
    delta_center_y = float(feature[17])
    delta_height = float(feature[19])
    downward_speed = float(feature[24])
    motion = float(feature[25])
    torso_horizontal = float(feature[9])
    body_horizontal = float(feature[14])
    area_growth = float(feature[35])
    height_drop_rate = float(feature[36])

    lying_pose = (
        aspect >= float(fall_logic_cfg["lying_aspect_threshold"])
        or torso_horizontal >= 0.66
        or body_horizontal >= 0.72
    )
    fast_drop = (
        downward_speed >= float(fall_logic_cfg["downward_speed_threshold"])
        or delta_center_y >= float(fall_logic_cfg["rapid_center_drop_threshold"])
        or height_drop_rate >= float(fall_logic_cfg["rapid_height_drop_threshold"])
        or delta_height <= -float(fall_logic_cfg["rapid_height_drop_threshold"])
    )
    approach_like = area_growth > float(fall_logic_cfg["approach_area_growth_limit"]) and not lying_pose
    sitting_like = height < 0.58 and aspect < 1.05 and torso_horizontal < 0.62
    bending_like = torso_horizontal > 0.50 and body_horizontal < 0.68 and aspect < 1.18

    if fast_drop and lying_pose and not approach_like:
        probs[LABEL_TO_ID["FALLING"]] = 0.56
        probs[LABEL_TO_ID["FALLEN"]] = 0.22
        probs[LABEL_TO_ID["WALKING_STANDING"]] = 0.10
        probs[LABEL_TO_ID["LYING_INTENTIONAL"]] = 0.06
        probs[LABEL_TO_ID["NORMAL"]] = 0.06
    elif lying_pose:
        probs[LABEL_TO_ID["LYING_INTENTIONAL"]] = 0.68
        probs[LABEL_TO_ID["FALLEN"]] = 0.14
        probs[LABEL_TO_ID["NORMAL"]] = 0.10
        probs[LABEL_TO_ID["FALLING"]] = 0.08 if motion > 0.03 else 0.03
    elif sitting_like:
        probs[LABEL_TO_ID["SITTING"]] = 0.70
        probs[LABEL_TO_ID["NORMAL"]] = 0.18
        probs[LABEL_TO_ID["WALKING_STANDING"]] = 0.08
        probs[LABEL_TO_ID["BENDING"]] = 0.04
    elif bending_like:
        probs[LABEL_TO_ID["BENDING"]] = 0.66
        probs[LABEL_TO_ID["NORMAL"]] = 0.18
        probs[LABEL_TO_ID["SITTING"]] = 0.08
        probs[LABEL_TO_ID["WALKING_STANDING"]] = 0.08
    elif approach_like:
        probs[LABEL_TO_ID["WALKING_STANDING"]] = 0.64
        probs[LABEL_TO_ID["NORMAL"]] = 0.30
        probs[LABEL_TO_ID["FALLING"]] = 0.02
        probs[LABEL_TO_ID["BENDING"]] = 0.04
    else:
        probs[LABEL_TO_ID["WALKING_STANDING"]] = 0.54
        probs[LABEL_TO_ID["NORMAL"]] = 0.36
        probs[LABEL_TO_ID["BENDING"]] = 0.05
        probs[LABEL_TO_ID["SITTING"]] = 0.05
    return probs / max(float(np.sum(probs)), 1e-6)


@torch.no_grad()
def predict_probabilities(
    model: Optional[torch.nn.Module],
    history: Deque[np.ndarray],
    sequence_length: int,
    device: torch.device,
    feature: np.ndarray,
    fall_logic_cfg: Dict[str, float],
) -> np.ndarray:
    heuristic = heuristic_probabilities(feature, fall_logic_cfg)
    if model is None or len(history) == 0:
        return heuristic

    sequence = np.stack(list(history), axis=0).astype(np.float32)
    if len(sequence) < sequence_length:
        pad = np.repeat(sequence[0:1], sequence_length - len(sequence), axis=0)
        sequence = np.concatenate([pad, sequence], axis=0)
    else:
        sequence = sequence[-sequence_length:]

    tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
    logits = model(tensor)
    model_probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    if model_probs.shape[0] != heuristic.shape[0]:
        padded = np.zeros_like(heuristic)
        padded[: min(len(padded), len(model_probs))] = model_probs[: min(len(padded), len(model_probs))]
        model_probs = padded
    blended = 0.88 * model_probs + 0.12 * heuristic
    return blended / max(float(np.sum(blended)), 1e-6)


def source_name(source: str) -> str:
    if str(source).isdigit():
        return f"webcam_{source}"
    return Path(str(source)).stem or "source"

# Fix loi OpenH264 tren Windows: khong dung H264/avc1, uu tien mp4v va fallback sang AVI.
def create_compatible_video_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[Path, str]] = []
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        candidates = [
            (path, "mp4v"),
            (path.with_suffix(".avi"), "XVID"),
            (path.with_suffix(".avi"), "MJPG"),
        ]
    elif suffix == ".avi":
        candidates = [
            (path, "XVID"),
            (path, "MJPG"),
            (path.with_suffix(".mp4"), "mp4v"),
        ]
    else:
        candidates = [
            (path.with_suffix(".mp4"), "mp4v"),
            (path.with_suffix(".avi"), "XVID"),
            (path.with_suffix(".avi"), "MJPG"),
        ]

    for output_path, codec in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), frame_size)
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(
        "Cannot initialize cv2.VideoWriter. Tried codecs: "
        + ", ".join(f"{codec} -> {output_path}" for output_path, codec in candidates)
    )


def patch_video_writer_codec(logger=None) -> None:
    """Ep toan bo project dung codec tuong thich thay vi H264/OpenH264."""
    global create_video_writer
    create_video_writer = create_compatible_video_writer

    try:
        import src.utils as utils_module
        utils_module.create_video_writer = create_compatible_video_writer
    except Exception as exc:
        if logger is not None:
            logger.warning("Could not patch src.utils.create_video_writer: %s", exc)

    try:
        import src.clip_writer as clip_writer_module
        if hasattr(clip_writer_module, "create_video_writer"):
            clip_writer_module.create_video_writer = create_compatible_video_writer
    except Exception as exc:
        if logger is not None:
            logger.warning("Could not patch src.clip_writer.create_video_writer: %s", exc)


# Mau BGR cho tung trang thai khi ve khung tren OpenCV.
# BGR: xanh la = an toan, vang = nguy co, do = nguy hiem.
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

STATUS_COLOR_BGR = {
    "NORMAL": (80, 220, 80),              # xanh la
    "WALKING_STANDING": (80, 220, 80),    # xanh la
    "SITTING": (255, 190, 60),            # xanh duong nhat/cyan
    "BENDING": (0, 215, 255),             # vang/cam
    "LYING_INTENTIONAL": (255, 80, 200),  # tim/hong
    "FALLING": (0, 215, 255),             # vang/cam canh bao
    "FALLEN": (40, 40, 255),              # do
    "FAINT": (170, 40, 255),              # tim do
    "UNCONSCIOUS": (170, 40, 255),        # tim do
}


def status_to_frame_text(status: str) -> str:
    return STATUS_TEXT_FRAME.get(str(status), str(status).replace("_", " ").upper())


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
    """Ve bounding box co mau rieng cho tung trang thai.

    Ham nay thay cho draw_track mac dinh de tranh tat ca khung deu mau trang.
    """
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


def main() -> None:
    parser = build_arg_parser("Run multi-state fall detection inference on webcam or video.")
    parser.add_argument("--source", default=None, help="Webcam index or video path. Default comes from config.")
    parser.add_argument("--checkpoint", default=None, help="Path to trained checkpoint.")
    parser.add_argument("--no-display", action="store_true", help="Disable cv2.imshow.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logging(config, "inference")
    patch_video_writer_codec(logger)

    source = args.source if args.source is not None else str(config["inference"]["source"])
    src_name = source_name(str(source))
    checkpoint_path = resolve_path(args.checkpoint or config["inference"]["checkpoint"])
    output_root = resolve_path(config["paths"]["outputs_dir"])
    timestamp = safe_timestamp()

    detector_cfg = config["detector"]
    device = torch.device(select_device(str(detector_cfg["device"])))
    detector = YoloPersonPoseDetector(
        model_name=str(detector_cfg["model"]),
        confidence=float(detector_cfg["confidence"]),
        iou=float(detector_cfg["iou"]),
        image_size=int(detector_cfg["image_size"]),
        max_detections=int(detector_cfg["max_detections"]),
        device=str(detector_cfg["device"]),
        logger=logger,
    )

    model: Optional[torch.nn.Module] = None
    if checkpoint_path.exists():
        model, payload = load_model_from_checkpoint(str(checkpoint_path), device, fallback_config=config)
        logger.info("Loaded checkpoint %s at epoch %s", checkpoint_path, payload.get("epoch", "unknown"))
    elif bool(config["inference"]["allow_heuristic_without_checkpoint"]):
        logger.warning("Checkpoint not found: %s. Running heuristic-only temporal inference.", checkpoint_path)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tracker_cfg = config["tracker"]
    tracker = PersonTracker(
        iou_threshold=float(tracker_cfg["iou_threshold"]),
        max_center_distance=float(tracker_cfg["max_center_distance"]),
        max_missing_frames=int(tracker_cfg["max_missing_frames"]),
        min_hits=int(tracker_cfg["min_hits"]),
        smoothing_alpha=float(tracker_cfg["smoothing_alpha"]),
    )
    state_machine = FallStateMachine(config["fall_logic"])
    smoother = ProbabilitySmoother(window_size=5, alpha=0.70)
    sequence_length = int(config["features"]["sequence_length"])
    model_sample_fps = max(1.0, float(config["features"]["sample_fps"]))
    histories: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=sequence_length))
    last_decisions: Dict[int, FallDecision] = {}

    cap = cv2.VideoCapture(parse_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or config["inference"]["output_fps"])
    if fps <= 1.0 or fps > 120.0:
        fps = float(config["inference"]["output_fps"])

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Cannot read first frame from source: {source}")
    frame_h, frame_w = first_frame.shape[:2]
    frame_size = (frame_w, frame_h)

    annotated_writer: Optional[cv2.VideoWriter] = None
    if bool(config["inference"]["save_annotated"]):
        annotated_writer = create_video_writer(
            make_unique_path(output_root / f"annotated_{src_name}_{timestamp}.mp4"),
            fps,
            frame_size,
        )

    event_logger: Optional[EventLogger] = None
    if bool(config["inference"]["save_event_log"]):
        event_logger = EventLogger(output_root / "events" / f"events_{src_name}_{timestamp}.csv")

    pre_frames = int(round(float(config["inference"]["pre_event_seconds"]) * fps))
    post_frames = int(round(float(config["inference"]["post_event_seconds"]) * fps))
    no_person_post_frames = int(round(float(config["inference"]["clip_no_person_timeout_sec"]) * fps))
    person_recorder = EventClipRecorder(output_root / "person_clips", "person", fps, frame_size, 0, no_person_post_frames, src_name)
    fall_recorder = EventClipRecorder(output_root / "fall_clips", "fall", fps, frame_size, pre_frames, post_frames, src_name)
    faint_recorder = EventClipRecorder(
        output_root / "faint_clips", "faint", fps, frame_size, pre_frames, post_frames, src_name
    )

    display = bool(config["inference"]["display"]) and not args.no_display
    logger.info("Running inference on source=%s fps=%.2f output=%s", source, fps, output_root)
    temporal_stride = max(1, int(round(fps / model_sample_fps)))
    logger.info(
        "Temporal model update rate: %.2f FPS (source fps=%.2f, stride=%d)",
        fps / temporal_stride,
        fps,
        temporal_stride,
    )

    previous_sample_gray: Optional[np.ndarray] = None
    frame_index = 0
    pending_frame: Optional[np.ndarray] = first_frame
    try:
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            timestamp_sec = float(frame_index / fps)
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
            should_update_temporal = frame_index % temporal_stride == 0

            for track in tracks:
                if track.observation is None:
                    continue
                if should_update_temporal:
                    # Training feature extraction samples videos at features.sample_fps.
                    # Keep inference deltas at the same cadence; otherwise 30 FPS live
                    # input shrinks drop speed and compresses the GRU time window.
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
                        fall_logic_cfg=config["fall_logic"],
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
                any_fall = any_fall or decision.is_fall_alert
                any_faint = any_faint or decision.is_unconscious_alert
                draw_track_colored(
                    annotated,
                    track.bbox_xyxy,
                    track.track_id,
                    decision.status,
                    decision.model_confidence,
                    decision.fall_score,
                )

            header = f"people={len(tracks)} fall={int(any_fall)} faint={int(any_faint)} t={timestamp_sec:.1f}s"
            draw_header(annotated, header)

            if annotated_writer is not None:
                annotated_writer.write(annotated)
            if bool(config["inference"]["save_person_clips"]):
                person_recorder.update(annotated, len(tracks) > 0, frame_index)
            if bool(config["inference"]["save_fall_clips"]):
                fall_recorder.update(annotated, any_fall, frame_index)
            if bool(config["inference"].get("save_faint_clips", config["inference"].get("save_unconscious_clips", True))):
                faint_recorder.update(annotated, any_faint, frame_index)

            if display:
                cv2.imshow("Camera AI Multi-Dataset Fall Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    logger.info("Stopped by user.")
                    break

            if should_update_temporal:
                previous_sample_gray = gray
            frame_index += 1
    finally:
        cap.release()
        if annotated_writer is not None:
            annotated_writer.release()
        if event_logger is not None:
            event_logger.close()
        person_recorder.close()
        fall_recorder.close()
        faint_recorder.close()
        if display:
            cv2.destroyAllWindows()
    logger.info("Inference complete. Outputs saved to %s", output_root)


if __name__ == "__main__":
    main()

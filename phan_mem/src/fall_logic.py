from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np

from .label_mapping import ID_TO_LABEL, LABEL_TO_ID


@dataclass
class FallDecision:
    status: str
    raw_label: str
    model_confidence: float
    fall_score: float
    is_fall_alert: bool
    is_unconscious_alert: bool
    reason: str


@dataclass
class TrackFallState:
    track_id: int
    status: str = "NORMAL"
    frame_count: int = 0
    stable_upright_count: int = 0
    fall_event_seen: bool = False
    fall_event_time: Optional[float] = None
    fallen_start_time: Optional[float] = None
    fallen_status_start_time: Optional[float] = None
    still_start_time: Optional[float] = None
    recovery_start_time: Optional[float] = None
    intentional_lying_start_time: Optional[float] = None
    baseline_height: Optional[float] = None
    baseline_center_y: Optional[float] = None
    velocity_history: Deque[tuple[float, bool]] = field(default_factory=deque)
    evidence_history: Deque[tuple[float, bool]] = field(default_factory=deque)
    slow_lying_history: Deque[tuple[float, bool]] = field(default_factory=deque)
    last_decision: Optional[FallDecision] = None


class FallStateMachine:
    def __init__(self, config: Dict[str, float]) -> None:
        self.model_falling_threshold = float(config["model_falling_threshold"])
        self.model_fallen_threshold = float(config["model_fallen_threshold"])
        self.fall_score_threshold = float(config["fall_score_threshold"])
        self.lying_aspect_threshold = float(config["lying_aspect_threshold"])
        self.rapid_center_drop_threshold = float(config["rapid_center_drop_threshold"])
        self.rapid_height_drop_threshold = float(config["rapid_height_drop_threshold"])
        self.downward_speed_threshold = float(config["downward_speed_threshold"])
        self.strong_downward_speed_threshold = float(config.get("strong_downward_speed_threshold", self.downward_speed_threshold * 2.7))
        self.strong_center_drop_threshold = float(config.get("strong_center_drop_threshold", self.rapid_center_drop_threshold * 1.7))
        self.strong_height_drop_threshold = float(config.get("strong_height_drop_threshold", self.rapid_height_drop_threshold * 1.6))
        self.min_fall_evidence_frames = int(config["min_fall_evidence_frames"])
        self.fall_confirmation_window_sec = float(config["fall_confirmation_window_sec"])
        self.fall_velocity_window_sec = float(config.get("fall_velocity_window_sec", min(1.2, self.fall_confirmation_window_sec)))
        self.min_velocity_evidence_frames = max(1, int(config.get("min_velocity_evidence_frames", 1)))
        self.lying_no_velocity_score_penalty = float(config.get("lying_no_velocity_score_penalty", 0.30))
        self.fallen_hold_sec = float(config["fallen_hold_sec"])
        self.faint_seconds = float(config.get("faint_seconds", config.get("unconscious_seconds", 60.0)))
        self.intentional_lying_slow_speed = float(config["intentional_lying_slow_speed"])
        self.intentional_lying_window_sec = float(config["intentional_lying_window_sec"])
        self.still_motion_threshold = float(config["still_motion_threshold"])
        self.still_center_speed_threshold = float(config["still_center_speed_threshold"])
        self.unconscious_seconds = float(config["unconscious_seconds"])
        self.recovery_upright_seconds = float(config["recovery_upright_seconds"])
        self.sitting_probability_block = float(config["sitting_probability_block"])
        self.bending_probability_block = float(config["bending_probability_block"])
        self.intentional_lying_probability_block = float(config["intentional_lying_probability_block"])
        self.approach_area_growth_limit = float(config["approach_area_growth_limit"])
        self.approach_score_penalty = float(config["approach_score_penalty"])
        self.min_stable_upright_frames_for_fall = max(1, int(config.get("min_stable_upright_frames_for_fall", 4)))
        self.closeup_area_threshold = float(config.get("closeup_area_threshold", 0.34))
        self.closeup_width_threshold = float(config.get("closeup_width_threshold", 0.55))
        self.closeup_height_threshold = float(config.get("closeup_height_threshold", 0.68))
        self.closeup_top_threshold = float(config.get("closeup_top_threshold", 0.30))
        self.closeup_edge_margin = float(config.get("closeup_edge_margin", 0.04))
        self.closeup_visibility_threshold = float(config.get("closeup_visibility_threshold", 0.72))
        self.closeup_confidence_threshold = float(config.get("closeup_confidence_threshold", 0.80))
        self.closeup_score_penalty = float(config.get("closeup_score_penalty", 0.42))
        self.states: Dict[int, TrackFallState] = {}

    def _state(self, track_id: int) -> TrackFallState:
        if track_id not in self.states:
            self.states[track_id] = TrackFallState(track_id=track_id)
        return self.states[track_id]

    def remove_missing_tracks(self, active_track_ids: set[int]) -> None:
        stale_ids = [track_id for track_id in self.states if track_id not in active_track_ids]
        for track_id in stale_ids:
            del self.states[track_id]

    def update(
        self,
        track_id: int,
        feature: np.ndarray,
        probabilities: Optional[np.ndarray],
        timestamp: float,
    ) -> FallDecision:
        state = self._state(track_id)
        state.frame_count += 1
        previous_status = state.status

        if probabilities is None:
            probabilities = np.zeros((len(LABEL_TO_ID),), dtype=np.float32)
            probabilities[LABEL_TO_ID["NORMAL"]] = 1.0
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if probabilities.size < len(LABEL_TO_ID):
            padded = np.zeros((len(LABEL_TO_ID),), dtype=np.float32)
            padded[: probabilities.size] = probabilities
            probabilities = padded

        raw_id = int(np.argmax(probabilities))
        raw_label = ID_TO_LABEL.get(raw_id, "NORMAL")
        model_confidence = float(np.max(probabilities))

        p_normal = float(probabilities[LABEL_TO_ID["NORMAL"]])
        p_walk = float(probabilities[LABEL_TO_ID["WALKING_STANDING"]])
        p_sit = float(probabilities[LABEL_TO_ID["SITTING"]])
        p_bend = float(probabilities[LABEL_TO_ID["BENDING"]])
        p_lying_intentional = float(probabilities[LABEL_TO_ID["LYING_INTENTIONAL"]])
        p_falling = float(probabilities[LABEL_TO_ID["FALLING"]])
        p_fallen = float(probabilities[LABEL_TO_ID["FALLEN"]])

        center_y = float(feature[1])
        width = float(feature[2])
        area = float(feature[5])
        height = float(feature[3])
        aspect = float(feature[4])
        delta_center_y = float(feature[17])
        delta_height = float(feature[19])
        speed = float(feature[23])
        downward_speed = float(feature[24])
        motion_mean = float(feature[25])
        missing_flag = float(feature[26])
        torso_horizontal = float(feature[9])
        body_horizontal = float(feature[14])
        visible_ratio = float(feature[8])
        area_growth = float(feature[35])
        height_drop_rate = float(feature[36])
        top_y = float(feature[30])
        left_x = float(feature[31])
        right_x = float(feature[32])
        keypoint_confidence_mean = float(feature[33])

        upright_pose = aspect < 0.95 and torso_horizontal < 0.50 and body_horizontal < 0.58
        sitting_like = p_sit >= self.sitting_probability_block or (aspect < 1.05 and height < 0.58 and torso_horizontal < 0.62)
        bending_like = p_bend >= self.bending_probability_block or (torso_horizontal > 0.50 and body_horizontal < 0.68 and aspect < 1.18)
        edge_cropped = (
            top_y <= self.closeup_top_threshold
            or left_x <= self.closeup_edge_margin
            or right_x >= (1.0 - self.closeup_edge_margin)
        )
        closeup_partial_body = (
            edge_cropped
            and torso_horizontal < 0.45
            and body_horizontal < 0.55
            and (
                (
                    visible_ratio < self.closeup_visibility_threshold
                    and (area >= self.closeup_area_threshold or width >= self.closeup_width_threshold or height >= self.closeup_height_threshold)
                )
                or (
                    top_y <= min(0.14, self.closeup_top_threshold * 0.5)
                    and visible_ratio < 0.40
                    and height >= 0.55
                )
                or (
                    width >= 0.90
                    and top_y <= min(0.14, self.closeup_top_threshold * 0.5)
                    and visible_ratio < 0.60
                )
            )
        )
        lying_pose_raw = (
            aspect >= self.lying_aspect_threshold
            or torso_horizontal >= 0.66
            or body_horizontal >= 0.72
            or p_fallen >= self.model_fallen_threshold
            or p_lying_intentional >= self.intentional_lying_probability_block
        )
        lying_pose = lying_pose_raw and not closeup_partial_body
        fast_drop = (
            downward_speed >= self.downward_speed_threshold
            or delta_center_y >= self.rapid_center_drop_threshold
            or height_drop_rate >= self.rapid_height_drop_threshold
            or delta_height <= -self.rapid_height_drop_threshold
        )
        very_fast_drop = (
            downward_speed >= self.strong_downward_speed_threshold
            or delta_center_y >= self.strong_center_drop_threshold
            or height_drop_rate >= self.strong_height_drop_threshold
            or delta_height <= -self.strong_height_drop_threshold
        )
        drop_velocity_evidence = fast_drop or very_fast_drop
        state.velocity_history.append((float(timestamp), drop_velocity_evidence))
        while state.velocity_history and timestamp - state.velocity_history[0][0] > self.fall_velocity_window_sec:
            state.velocity_history.popleft()
        recent_velocity_count = sum(1 for _, value in state.velocity_history if value)
        has_recent_velocity = recent_velocity_count >= self.min_velocity_evidence_frames
        approach_like = area_growth > self.approach_area_growth_limit and not lying_pose

        stable_upright_frame = (
            missing_flag < 0.5
            and upright_pose
            and visible_ratio >= 0.65
            and not closeup_partial_body
        )
        if stable_upright_frame and not state.fall_event_seen:
            state.stable_upright_count = min(state.stable_upright_count + 1, self.min_stable_upright_frames_for_fall + 6)
            state.baseline_height = height if state.baseline_height is None else 0.94 * state.baseline_height + 0.06 * max(height, state.baseline_height)
            state.baseline_center_y = center_y if state.baseline_center_y is None else 0.94 * state.baseline_center_y + 0.06 * min(center_y, state.baseline_center_y)
        elif not state.fall_event_seen and closeup_partial_body:
            state.stable_upright_count = max(0, state.stable_upright_count - 1)

        baseline_height = state.baseline_height or max(height, 1e-6)
        height_drop_from_baseline = max(0.0, (baseline_height - height) / max(baseline_height, 1e-6))
        baseline_ready = state.stable_upright_count >= self.min_stable_upright_frames_for_fall
        direct_fall_pose_ready = lying_pose and visible_ratio >= 0.70 and keypoint_confidence_mean >= 0.45
        fall_gate_ready = baseline_ready or direct_fall_pose_ready

        slow_lying_evidence = (
            lying_pose
            and not has_recent_velocity
            and speed <= max(self.intentional_lying_slow_speed, self.still_center_speed_threshold * 2.0)
            and downward_speed <= self.intentional_lying_slow_speed
            and height_drop_rate < self.rapid_height_drop_threshold
        )
        state.slow_lying_history.append((float(timestamp), slow_lying_evidence))
        while state.slow_lying_history and timestamp - state.slow_lying_history[0][0] > self.intentional_lying_window_sec:
            state.slow_lying_history.popleft()
        slow_lying_votes = sum(1 for _, value in state.slow_lying_history if value)
        initial_lying_without_fall = state.frame_count <= self.min_fall_evidence_frames and lying_pose and not has_recent_velocity
        prior_intentional_lying = state.intentional_lying_start_time is not None and lying_pose
        intentional_lying = (
            not state.fall_event_seen
            and (
                (p_lying_intentional >= self.intentional_lying_probability_block and not has_recent_velocity)
                or initial_lying_without_fall
                or prior_intentional_lying
                or slow_lying_votes >= max(2, self.min_fall_evidence_frames // 2)
            )
        )
        if intentional_lying:
            if state.intentional_lying_start_time is None:
                state.intentional_lying_start_time = float(timestamp)
        elif state.fall_event_seen or not lying_pose or (upright_pose and (p_walk > 0.35 or p_normal > 0.35 or p_sit > 0.45)):
            state.intentional_lying_start_time = None

        fall_score = (
            0.40 * p_falling
            + 0.18 * p_fallen
            + 0.16 * float(fast_drop)
            + 0.12 * float(very_fast_drop)
            + 0.15 * float(lying_pose)
            + 0.11 * min(height_drop_from_baseline, 1.0)
            - self.approach_score_penalty * float(approach_like)
            - 0.24 * float(sitting_like and not fast_drop)
            - 0.20 * float(bending_like and not lying_pose)
            - 0.28 * float(intentional_lying)
            - self.closeup_score_penalty * float(closeup_partial_body)
            - self.lying_no_velocity_score_penalty * float(lying_pose and not has_recent_velocity and not state.fall_event_seen)
        )
        fall_score = float(np.clip(fall_score, 0.0, 1.0))

        fall_evidence = (
            missing_flag < 0.5
            and fall_gate_ready
            and fall_score >= self.fall_score_threshold
            and has_recent_velocity
            and (p_falling >= self.model_falling_threshold or fast_drop or very_fast_drop)
            and (lying_pose or height_drop_from_baseline > 0.18 or p_fallen >= self.model_fallen_threshold)
            and not approach_like
            and not intentional_lying
            and not closeup_partial_body
            and not (sitting_like and p_falling < self.model_falling_threshold)
            and not (bending_like and not lying_pose)
        )
        velocity_fall_evidence = (
            missing_flag < 0.5
            and fall_gate_ready
            and very_fast_drop
            and not approach_like
            and not intentional_lying
            and not closeup_partial_body
            and not (bending_like and not lying_pose)
            and (
                p_falling >= 0.30
                or p_fallen >= self.model_fallen_threshold
                or lying_pose
                or height_drop_from_baseline > 0.20
            )
            and not (sitting_like and p_falling < 0.45 and not lying_pose)
        )
        strong_single_fall_evidence = (
            missing_flag < 0.5
            and fall_gate_ready
            and fall_score >= max(self.fall_score_threshold + 0.10, 0.68)
            and p_falling >= max(self.model_falling_threshold + 0.25, 0.67)
            and has_recent_velocity
            and (fast_drop or very_fast_drop)
            and not approach_like
            and not intentional_lying
            and not closeup_partial_body
            and not (sitting_like and not lying_pose and not fast_drop)
            and not (bending_like and not lying_pose)
        )
        velocity_fall_pose_ready = (
            lying_pose
            or torso_horizontal >= 0.48
            or body_horizontal >= 0.52
            or aspect >= max(0.95, self.lying_aspect_threshold - 0.10)
            or downward_speed >= self.strong_downward_speed_threshold * 1.15
            or delta_center_y >= self.strong_center_drop_threshold * 1.15
        )
        state.evidence_history.append((float(timestamp), fall_evidence))
        while state.evidence_history and timestamp - state.evidence_history[0][0] > self.fall_confirmation_window_sec:
            state.evidence_history.popleft()
        evidence_count = sum(1 for _, value in state.evidence_history if value)

        reason = raw_label
        if not state.fall_event_seen and (
            (evidence_count >= self.min_fall_evidence_frames and has_recent_velocity)
            or strong_single_fall_evidence
            or (velocity_fall_evidence and velocity_fall_pose_ready)
        ):
            state.fall_event_seen = True
            state.fall_event_time = float(timestamp)
            state.fallen_start_time = None
            state.fallen_status_start_time = None
            state.still_start_time = None
            state.recovery_start_time = None
            state.status = "FALLING"
            if velocity_fall_evidence and velocity_fall_pose_ready:
                reason = "velocity_fall_evidence"
            elif strong_single_fall_evidence:
                reason = "strong_single_fall_evidence"
            else:
                reason = "confirmed_temporal_fall_evidence"

        if state.fall_event_seen:
            post_fall_lying = (lying_pose or p_fallen >= self.model_fallen_threshold) and not closeup_partial_body
            if post_fall_lying:
                if state.fallen_start_time is None:
                    state.fallen_start_time = float(timestamp)
                if timestamp - state.fallen_start_time >= self.fallen_hold_sec:
                    if state.status != "FAINT":
                        state.status = "FALLEN"
                        reason = "post_fall_lying_confirmed"
            elif state.status == "FALLING" and state.fall_event_time is not None:
                if timestamp - state.fall_event_time > self.fall_confirmation_window_sec:
                    state.status = "NORMAL"
                    state.fall_event_seen = False
                    state.fall_event_time = None
                    state.fallen_start_time = None
                    state.fallen_status_start_time = None
                    state.still_start_time = None
                    state.intentional_lying_start_time = None
                    state.velocity_history.clear()
                    state.evidence_history.clear()
                    reason = "fall_evidence_not_sustained"

            recovered = upright_pose and (p_walk > 0.35 or p_normal > 0.35 or p_sit > 0.45) and p_fallen < 0.35
            if recovered:
                if state.recovery_start_time is None:
                    state.recovery_start_time = float(timestamp)
                if timestamp - state.recovery_start_time >= self.recovery_upright_seconds:
                    state.status = "NORMAL" if p_sit < 0.45 else "SITTING"
                    state.fall_event_seen = False
                    state.fall_event_time = None
                    state.fallen_start_time = None
                    state.fallen_status_start_time = None
                    state.still_start_time = None
                    state.recovery_start_time = None
                    state.intentional_lying_start_time = None
                    state.velocity_history.clear()
                    state.evidence_history.clear()
                    reason = "recovered_before_faint"
            else:
                state.recovery_start_time = None

            still_after_fall = (
                state.status in {"FALLEN", "FAINT", "UNCONSCIOUS"}
                and post_fall_lying
                and motion_mean <= self.still_motion_threshold
                and speed <= self.still_center_speed_threshold
                and missing_flag < 0.5
            )
            if still_after_fall:
                if state.still_start_time is None:
                    state.still_start_time = float(timestamp)
            elif state.status not in {"FAINT", "UNCONSCIOUS"}:
                state.still_start_time = None
        else:
            if intentional_lying:
                state.status = "LYING_INTENTIONAL"
                reason = "slow_or_initial_lying_without_fall"
            elif closeup_partial_body:
                state.status = "WALKING_STANDING" if upright_pose else "NORMAL"
                reason = "closeup_partial_body_block"
            elif lying_pose and not has_recent_velocity:
                state.status = "LYING_INTENTIONAL"
                reason = "lying_without_velocity_evidence"
            elif sitting_like and not fast_drop:
                state.status = "SITTING"
                reason = "sitting_block"
            elif bending_like and not lying_pose:
                state.status = "BENDING"
                reason = "bending_block"
            elif p_walk >= 0.45 or upright_pose:
                state.status = "WALKING_STANDING"
                reason = "upright_activity"
            else:
                state.status = "NORMAL"
                reason = "normal_or_uncertain"

        if state.status == "FALLEN":
            if previous_status != "FALLEN" or state.fallen_status_start_time is None:
                state.fallen_status_start_time = float(timestamp)
            elif timestamp - state.fallen_status_start_time >= self.faint_seconds:
                state.status = "FAINT"
                reason = "fallen_unchanged_for_faint_timeout"
        elif state.status == "FAINT":
            if state.fallen_status_start_time is None:
                state.fallen_status_start_time = float(timestamp)
        elif state.status != "UNCONSCIOUS":
            state.fallen_status_start_time = None

        decision = FallDecision(
            status=state.status,
            raw_label=raw_label,
            model_confidence=model_confidence,
            fall_score=fall_score,
            is_fall_alert=state.status in {"FALLING", "FALLEN", "FAINT", "UNCONSCIOUS"},
            is_unconscious_alert=state.status in {"FAINT", "UNCONSCIOUS"},
            reason=reason,
        )
        state.last_decision = decision
        return decision

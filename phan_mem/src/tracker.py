from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import PersonObservation


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a.astype(float)
    bx1, by1, bx2, by2 = box_b.astype(float)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def bbox_intersection_over_min(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a.astype(float)
    bx1, by1, bx2, by2 = box_b.astype(float)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    min_area = min(area_a, area_b)
    if min_area <= 0.0:
        return 0.0
    return float(intersection / min_area)


def bbox_area_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a.astype(float)
    bx1, by1, bx2, by2 = box_b.astype(float)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    max_area = max(area_a, area_b)
    if max_area <= 0.0:
        return 0.0
    return float(min(area_a, area_b) / max_area)


def normalized_center_distance(box_a: np.ndarray, box_b: np.ndarray, frame_shape: Tuple[int, int]) -> float:
    frame_h, frame_w = frame_shape
    diag = max(1.0, float(np.hypot(frame_w, frame_h)))
    acx = (float(box_a[0]) + float(box_a[2])) * 0.5
    acy = (float(box_a[1]) + float(box_a[3])) * 0.5
    bcx = (float(box_b[0]) + float(box_b[2])) * 0.5
    bcy = (float(box_b[1]) + float(box_b[3])) * 0.5
    return float(np.hypot(acx - bcx, acy - bcy) / diag)


def horizontal_overlap_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, _, ax2, _ = box_a.astype(float)
    bx1, _, bx2, _ = box_b.astype(float)
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    min_width = min(max(0.0, ax2 - ax1), max(0.0, bx2 - bx1))
    if min_width <= 0.0:
        return 0.0
    return float(overlap / min_width)


@dataclass
class Track:
    track_id: int
    bbox_xyxy: np.ndarray
    confidence: float
    frame_shape: Tuple[int, int]
    first_frame: int
    last_frame: int
    observation: Optional[PersonObservation] = None
    age: int = 1
    hits: int = 1
    missing_frames: int = 0
    confirmed: bool = False
    last_feature: Optional[np.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def update(self, observation: PersonObservation, frame_index: int, smoothing_alpha: float) -> None:
        alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.bbox_xyxy = alpha * observation.bbox_xyxy.astype(np.float32) + (1.0 - alpha) * self.bbox_xyxy.astype(np.float32)
        self.confidence = float(observation.confidence)
        self.frame_shape = observation.frame_shape
        self.last_frame = frame_index
        self.observation = observation
        self.age += 1
        self.hits += 1
        self.missing_frames = 0

    def mark_missing(self) -> None:
        self.age += 1
        self.missing_frames += 1
        self.observation = None


class PersonTracker:
    def __init__(
        self,
        iou_threshold: float = 0.2,
        max_center_distance: float = 0.45,
        max_missing_frames: int = 30,
        min_hits: int = 2,
        smoothing_alpha: float = 0.65,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_center_distance = float(max_center_distance)
        self.max_missing_frames = int(max_missing_frames)
        self.min_hits = int(min_hits)
        self.smoothing_alpha = float(smoothing_alpha)
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    def _recycle_recent_track(self, observation: PersonObservation, frame_index: int) -> bool:
        candidates: List[Tuple[float, Track]] = []
        for track in self.tracks.values():
            if not track.confirmed or track.missing_frames <= 0:
                continue
            distance = normalized_center_distance(track.bbox_xyxy, observation.bbox_xyxy, observation.frame_shape)
            overlap_ratio = bbox_intersection_over_min(track.bbox_xyxy, observation.bbox_xyxy)
            area_ratio = bbox_area_ratio(track.bbox_xyxy, observation.bbox_xyxy)
            x_overlap = horizontal_overlap_ratio(track.bbox_xyxy, observation.bbox_xyxy)
            if distance > max(self.max_center_distance + 0.25, 0.72) and overlap_ratio < 0.12 and x_overlap < 0.30:
                continue
            if area_ratio < 0.08 and x_overlap < 0.55:
                continue
            score = (
                0.48 * overlap_ratio
                + 0.18 * area_ratio
                + 0.18 * x_overlap
                - 0.45 * distance
                + 0.02 * min(track.hits, 10)
                - 0.03 * float(track.missing_frames)
            )
            candidates.append((score, track))
        if not candidates:
            return False
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_track = candidates[0]
        if best_score < -0.04:
            return False
        best_track.update(observation, frame_index, self.smoothing_alpha)
        best_track.confirmed = True
        return True

    def _match(
        self,
        tracks: Sequence[Track],
        observations: Sequence[PersonObservation],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not observations:
            return [], list(range(len(tracks))), list(range(len(observations)))

        cost = np.full((len(tracks), len(observations)), fill_value=1e6, dtype=np.float32)
        for track_index, track in enumerate(tracks):
            for obs_index, observation in enumerate(observations):
                iou = bbox_iou(track.bbox_xyxy, observation.bbox_xyxy)
                overlap_ratio = bbox_intersection_over_min(track.bbox_xyxy, observation.bbox_xyxy)
                area_ratio = bbox_area_ratio(track.bbox_xyxy, observation.bbox_xyxy)
                x_overlap = horizontal_overlap_ratio(track.bbox_xyxy, observation.bbox_xyxy)
                distance = normalized_center_distance(track.bbox_xyxy, observation.bbox_xyxy, observation.frame_shape)
                relaxed_center_distance = self.max_center_distance + min(0.20, 0.04 * float(track.missing_frames))
                strong_overlap_hint = overlap_ratio >= 0.28 or (x_overlap >= 0.55 and area_ratio >= 0.18)
                if iou < self.iou_threshold and distance > relaxed_center_distance and not strong_overlap_hint:
                    continue
                score = (
                    1.20 * iou
                    + 0.45 * overlap_ratio
                    + 0.20 * area_ratio
                    + 0.12 * x_overlap
                    - 0.55 * distance
                    + 0.05 * float(observation.confidence)
                    - 0.02 * float(track.missing_frames)
                )
                cost[track_index, obs_index] = -score

        try:
            from scipy.optimize import linear_sum_assignment

            row_indices, col_indices = linear_sum_assignment(cost)
            candidate_pairs = list(zip(row_indices.tolist(), col_indices.tolist()))
        except Exception:
            candidate_pairs = []
            used_tracks = set()
            used_observations = set()
            flat = [
                (float(cost[t, o]), t, o)
                for t in range(cost.shape[0])
                for o in range(cost.shape[1])
                if cost[t, o] < 1e5
            ]
            for _, track_index, obs_index in sorted(flat, key=lambda item: item[0]):
                if track_index in used_tracks or obs_index in used_observations:
                    continue
                candidate_pairs.append((track_index, obs_index))
                used_tracks.add(track_index)
                used_observations.add(obs_index)

        matches: List[Tuple[int, int]] = []
        matched_tracks = set()
        matched_observations = set()
        for track_index, obs_index in candidate_pairs:
            if cost[track_index, obs_index] >= 1e5:
                continue
            matches.append((track_index, obs_index))
            matched_tracks.add(track_index)
            matched_observations.add(obs_index)

        unmatched_tracks = [index for index in range(len(tracks)) if index not in matched_tracks]
        unmatched_observations = [index for index in range(len(observations)) if index not in matched_observations]
        return matches, unmatched_tracks, unmatched_observations

    def update(self, observations: Sequence[PersonObservation], frame_index: int) -> List[Track]:
        active_tracks = list(self.tracks.values())
        matches, unmatched_tracks, unmatched_observations = self._match(active_tracks, observations)

        for track_index, obs_index in matches:
            active_tracks[track_index].update(observations[obs_index], frame_index, self.smoothing_alpha)
            if active_tracks[track_index].hits >= self.min_hits:
                active_tracks[track_index].confirmed = True

        for track_index in unmatched_tracks:
            active_tracks[track_index].mark_missing()

        for obs_index in unmatched_observations:
            observation = observations[obs_index]
            if self._recycle_recent_track(observation, frame_index):
                continue
            track = Track(
                track_id=self.next_track_id,
                bbox_xyxy=observation.bbox_xyxy.astype(np.float32),
                confidence=float(observation.confidence),
                frame_shape=observation.frame_shape,
                first_frame=frame_index,
                last_frame=frame_index,
                observation=observation,
                confirmed=self.min_hits <= 1,
            )
            self.tracks[track.track_id] = track
            self.next_track_id += 1

        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missing_frames > self.max_missing_frames
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

        visible_confirmed = [
            track
            for track in self.tracks.values()
            if track.confirmed and track.missing_frames == 0
        ]
        visible_confirmed.sort(key=lambda item: item.track_id)
        return visible_confirmed

    def all_tracks(self) -> List[Track]:
        return sorted(self.tracks.values(), key=lambda item: item.track_id)

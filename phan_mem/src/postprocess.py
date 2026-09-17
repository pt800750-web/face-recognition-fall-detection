from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict

import numpy as np


class ProbabilitySmoother:
    def __init__(self, window_size: int = 5, alpha: float = 0.65) -> None:
        self.window_size = max(1, int(window_size))
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.histories: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.window_size))
        self.ema: Dict[int, np.ndarray] = {}

    def update(self, track_id: int, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=np.float32)
        total = float(np.sum(probs))
        if total > 0:
            probs = probs / total
        if track_id not in self.ema:
            self.ema[track_id] = probs.copy()
        else:
            self.ema[track_id] = self.alpha * probs + (1.0 - self.alpha) * self.ema[track_id]
        self.histories[track_id].append(self.ema[track_id].copy())
        smoothed = np.mean(np.stack(list(self.histories[track_id]), axis=0), axis=0)
        return smoothed / max(float(np.sum(smoothed)), 1e-6)

    def keep_only(self, active_track_ids: set[int]) -> None:
        for track_id in list(self.histories.keys()):
            if track_id not in active_track_ids:
                del self.histories[track_id]
        for track_id in list(self.ema.keys()):
            if track_id not in active_track_ids:
                del self.ema[track_id]

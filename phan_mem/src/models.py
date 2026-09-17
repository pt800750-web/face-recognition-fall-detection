from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class TemporalGRUFallClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.25,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_classes = int(num_classes)
        self.bidirectional = bool(bidirectional)
        gru_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=self.bidirectional,
        )
        direction_multiplier = 2 if self.bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * direction_multiplier),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * direction_multiplier, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout) * 0.5),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected tensor shape [B, T, F], got {tuple(features.shape)}")
        outputs, _ = self.gru(features)
        last_output = outputs[:, -1, :]
        return self.head(last_output)


def build_model(config: Dict[str, object]) -> TemporalGRUFallClassifier:
    model_cfg = config["model"]  # type: ignore[index]
    return TemporalGRUFallClassifier(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_classes=int(model_cfg["num_classes"]),
        dropout=float(model_cfg["dropout"]),
        bidirectional=bool(model_cfg["bidirectional"]),
    )


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[object],
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, object],
    class_names: List[str],
) -> None:
    payload: Dict[str, object] = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "metrics": metrics,
        "config": config,
        "class_names": class_names,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    fallback_config: Optional[Dict[str, object]] = None,
) -> tuple[TemporalGRUFallClassifier, Dict[str, object]]:
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    config = payload.get("config", fallback_config)
    if config is None:
        raise ValueError("Checkpoint does not contain config and fallback_config was not provided.")
    model = build_model(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload

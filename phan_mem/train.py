from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FallFeatureSequenceDataset
from src.features import FEATURE_DIM, ID_TO_LABEL, LABEL_TO_ID, YoloPersonPoseDetector
from src.metrics import classification_metrics, format_confusion_matrix
from src.models import build_model, save_checkpoint
from src.utils import (
    build_arg_parser,
    ensure_project_dirs,
    load_config,
    resolve_path,
    select_device,
    set_seed,
    setup_logging,
    write_json,
)


def build_class_weights(counts: np.ndarray, smoothing: float) -> torch.Tensor:
    counts = counts.astype(np.float32)
    weights = np.zeros_like(counts, dtype=np.float32)
    present = counts > 0
    if not np.any(present):
        return torch.ones_like(torch.tensor(counts, dtype=torch.float32))

    median_count = float(np.median(counts[present]))
    # Square-root inverse frequency is less brittle than plain inverse
    # frequency on this dataset, where FALLEN windows naturally dominate.
    weights[present] = np.sqrt(median_count / np.maximum(counts[present], 1.0))
    weights[present] = (1.0 - float(smoothing)) * weights[present] + float(smoothing)

    weights[LABEL_TO_ID["FALLING"]] *= 1.55
    weights[LABEL_TO_ID["FALLEN"]] *= 1.35
    weights[LABEL_TO_ID["LYING_INTENTIONAL"]] *= 1.25

    # Classes with no samples in the current training set cannot be learned;
    # keep their loss weight at zero instead of letting them distort the mean.
    weights[~present] = 0.0
    nonzero_mean = float(np.mean(weights[present]))
    weights[present] = weights[present] / max(nonzero_mean, 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(loader, desc="train", unit="batch")
    for features, targets in progress:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        batch_size = int(features.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{float(loss.item()):.4f}")
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> Tuple[float, Dict[str, object]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets_all = []
    progress = tqdm(loader, desc="val", unit="batch")
    for features, targets in progress:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, targets)
        batch_size = int(features.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
        targets_all.append(targets.cpu().numpy())
        progress.set_postfix(loss=f"{float(loss.item()):.4f}")

    y_pred = np.concatenate(predictions, axis=0) if predictions else np.zeros((0,), dtype=np.int64)
    y_true = np.concatenate(targets_all, axis=0) if targets_all else np.zeros((0,), dtype=np.int64)
    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return float(metrics["loss"]), metrics


def main() -> None:
    parser = build_arg_parser("Train temporal fall detection model.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs from config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)
    set_seed(int(config["project"]["seed"]))
    logger = setup_logging(config, "train")

    if int(config["model"]["input_dim"]) != FEATURE_DIM:
        raise ValueError(f"Config model.input_dim must be {FEATURE_DIM}, got {config['model']['input_dim']}")

    device = torch.device(select_device(str(config["detector"]["device"])))
    logger.info("Training device: %s", device)

    detector_cfg = config["detector"]
    detector = YoloPersonPoseDetector(
        model_name=str(detector_cfg["model"]),
        confidence=float(detector_cfg["confidence"]),
        iou=float(detector_cfg["iou"]),
        image_size=int(detector_cfg["image_size"]),
        max_detections=int(detector_cfg["max_detections"]),
        device=str(detector_cfg["device"]),
        logger=logger,
    )

    manifest_dir = resolve_path(config["paths"]["manifest_dir"])
    train_manifest = manifest_dir / "train.jsonl"
    val_manifest = manifest_dir / "val.jsonl"
    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError("Missing train/val manifests. Run prepare_dataset.py first.")

    train_dataset = FallFeatureSequenceDataset(
        manifest_path=train_manifest,
        config=config,
        split="train",
        logger=logger,
        detector=detector,
        augment=True,
    )
    val_dataset = FallFeatureSequenceDataset(
        manifest_path=val_manifest,
        config=config,
        split="val",
        logger=logger,
        detector=detector,
        augment=False,
    )

    num_classes = int(config["model"]["num_classes"])
    class_names = [ID_TO_LABEL[index] for index in range(num_classes)]
    logger.info("Train label distribution: %s", train_dataset.describe_labels(num_classes))
    logger.info("Val label distribution: %s", val_dataset.describe_labels(num_classes))

    batch_size = int(args.batch_size or config["train"]["batch_size"])
    num_workers = int(config["train"]["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model(config).to(device)
    class_weights = build_class_weights(
        train_dataset.label_counts(num_classes),
        smoothing=float(config["train"]["class_weight_smoothing"]),
    ).to(device)
    logger.info("Class weights: %s", class_weights.detach().cpu().numpy().round(3).tolist())
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    checkpoints_dir = resolve_path(config["paths"]["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / "best_model.pt"
    last_path = checkpoints_dir / "last_model.pt"
    metrics_path = resolve_path(config["paths"]["logs_dir"]) / "train_metrics.json"

    epochs = int(args.epochs or config["train"]["epochs"])
    checkpoint_metric = str(config["train"]["checkpoint_metric"])
    patience = int(config["train"]["patience"])
    grad_clip_norm = float(config["train"]["grad_clip_norm"])
    best_score = -float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d/%d", epoch, epochs)
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, grad_clip_norm)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, class_names)
        metric_value = float(val_metrics.get(checkpoint_metric, val_metrics.get("f1_macro", 0.0)))
        if checkpoint_metric == "fall_safety_score":
            metric_value = (
                0.45 * float(val_metrics.get("recall_falling", 0.0))
                + 0.25 * float(val_metrics.get("f1_falling", 0.0))
                + 0.20 * float(val_metrics.get("f1_fallen", 0.0))
                + 0.10 * float(val_metrics.get("f1_macro", 0.0))
            )
            val_metrics["fall_safety_score"] = metric_value
        scheduler.step(metric_value)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(epoch_record)
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f f1_macro=%.4f recall_falling=%.4f precision_falling=%.4f",
            epoch,
            train_loss,
            val_loss,
            float(val_metrics["f1_macro"]),
            float(val_metrics.get("recall_falling", 0.0)),
            float(val_metrics.get("precision_falling", 0.0)),
        )
        logger.info("\n%s", format_confusion_matrix(val_metrics["confusion_matrix"], class_names))  # type: ignore[arg-type]

        save_checkpoint(
            path=str(last_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics={"train_loss": train_loss, **{k: v for k, v in val_metrics.items() if isinstance(v, float)}},
            config=config,
            class_names=class_names,
        )

        if metric_value > best_score:
            best_score = metric_value
            bad_epochs = 0
            save_checkpoint(
                path=str(best_path),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={"train_loss": train_loss, **{k: v for k, v in val_metrics.items() if isinstance(v, float)}},
                config=config,
                class_names=class_names,
            )
            logger.info("Saved new best checkpoint to %s (%s=%.4f)", best_path, checkpoint_metric, metric_value)
        else:
            bad_epochs += 1
            logger.info("No improvement for %d/%d epochs.", bad_epochs, patience)

        write_json(metrics_path, {"history": history, "best_score": best_score, "metric": checkpoint_metric})
        if bad_epochs >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training complete. Best checkpoint: %s", best_path)


if __name__ == "__main__":
    main()

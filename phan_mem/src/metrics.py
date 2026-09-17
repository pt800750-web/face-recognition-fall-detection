from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, float | List[List[int]]]:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    metrics: Dict[str, float | List[List[int]]] = {
        "accuracy": accuracy,
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "precision_weighted": float(weighted_precision),
        "recall_weighted": float(weighted_recall),
        "f1_weighted": float(weighted_f1),
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    for index, class_name in enumerate(class_names):
        safe_name = class_name.lower()
        metrics[f"precision_{safe_name}"] = float(precision[index])
        metrics[f"recall_{safe_name}"] = float(recall[index])
        metrics[f"f1_{safe_name}"] = float(f1[index])
        metrics[f"support_{safe_name}"] = float(support[index])
    return metrics


def format_confusion_matrix(matrix: List[List[int]], class_names: List[str]) -> str:
    header = "true\\pred".ljust(14) + " ".join(name[:10].rjust(10) for name in class_names)
    rows = [header]
    for class_name, row in zip(class_names, matrix):
        rows.append(class_name[:12].ljust(14) + " ".join(str(value).rjust(10) for value in row))
    return "\n".join(rows)

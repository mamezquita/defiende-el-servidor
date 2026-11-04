"""Métricas de clasificación binaria. Clase positiva = ataque (1)."""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

SWEEP_THRESHOLDS = tuple(percent / 100 for percent in range(5, 96))
METRIC_NAMES = ("recall_attack", "precision_attack", "fpr", "macro_f1", "accuracy")


def _ratio(numerator: int, denominator: int) -> tuple[float, bool]:
    """Devuelve la razón y si hubo denominador cero, en cuyo caso vale 0.0."""
    if denominator == 0:
        return 0.0, True
    return numerator / denominator, False


def predict_at_threshold(score: ArrayLike, threshold: float) -> np.ndarray:
    """Predice ataque (1) cuando score >= threshold."""
    return (np.asarray(score) >= threshold).astype(int)


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, Any]:
    """Calcula las métricas y la matriz [[TN, FP], [FN, TP]].

    Una métrica con denominador cero vale 0.0 y queda marcada en `zero_division`.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"tamaños distintos: {true.shape} y {pred.shape}")

    tp = int(np.sum((true == 1) & (pred == 1)))
    tn = int(np.sum((true == 0) & (pred == 0)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    if tp + tn + fp + fn != true.size:
        raise ValueError("y_true e y_pred solo pueden contener 0 y 1")

    recall, recall_zero = _ratio(tp, tp + fn)
    precision, precision_zero = _ratio(tp, tp + fp)
    fpr, fpr_zero = _ratio(fp, fp + tn)
    accuracy, accuracy_zero = _ratio(tp + tn, true.size)
    f1_attack, f1_attack_zero = _ratio(2 * tp, 2 * tp + fp + fn)
    f1_normal, f1_normal_zero = _ratio(2 * tn, 2 * tn + fp + fn)

    return {
        "recall_attack": recall,
        "precision_attack": precision,
        "fpr": fpr,
        "macro_f1": (f1_attack + f1_normal) / 2,
        "accuracy": accuracy,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "zero_division": {
            "recall_attack": recall_zero,
            "precision_attack": precision_zero,
            "fpr": fpr_zero,
            "macro_f1": f1_attack_zero or f1_normal_zero,
            "accuracy": accuracy_zero,
        },
    }


def threshold_sweep(y: ArrayLike, score: ArrayLike) -> pd.DataFrame:
    """Calcula las métricas para cada umbral de 0.05 a 0.95, en pasos de 0.01."""
    rows = []
    for threshold in SWEEP_THRESHOLDS:
        metrics = compute_metrics(y, predict_at_threshold(score, threshold))
        (tn, fp), (fn, tp) = metrics["confusion_matrix"]
        rows.append(
            {
                "threshold": threshold,
                **{name: metrics[name] for name in METRIC_NAMES},
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "zero_division": any(metrics["zero_division"].values()),
            }
        )
    return pd.DataFrame(rows)

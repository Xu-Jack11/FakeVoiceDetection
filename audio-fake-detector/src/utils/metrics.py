"""Metric helpers for F1 computation."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from sklearn.metrics import f1_score


def compute_f1_scores(
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> Tuple[float, np.ndarray]:
    """
    Compute macro-F1 and class-wise F1 (labels ordered as [0, 1]).
    """
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.int64)
    macro_f1 = f1_score(y_true_arr, y_pred_arr, average="macro")
    per_class = f1_score(y_true_arr, y_pred_arr, average=None, labels=[0, 1])
    return macro_f1, per_class


def format_f1_log(macro_f1: float, per_class: np.ndarray) -> str:
    """
    Format macro and per-class F1 scores for logging.
    """
    ai_f1 = float(per_class[0]) if per_class.size > 0 else float("nan")
    human_f1 = float(per_class[1]) if per_class.size > 1 else float("nan")
    return (
        f"macro-F1: {macro_f1:.4f} | "
        f"F1(AI=0): {ai_f1:.4f} | "
        f"F1(human=1): {human_f1:.4f}"
    )

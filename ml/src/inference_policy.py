"""Safety-critical inference policies: asymmetric thresholding."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

SAFE, WARNING, DANGER = 0, 1, 2


def softmax_probs(logits: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(logits, torch.Tensor):
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    else:
        x = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(x)
        probs = e / e.sum(axis=-1, keepdims=True)
    return probs.astype(np.float64)


def argmax_predict(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(axis=1).astype(np.int64)


def asymmetric_predict(
    probs: np.ndarray,
    danger_thresh: float = 0.15,
    warning_thresh: float | None = None,
) -> np.ndarray:
    """
    Fail-safe priority: Danger if P(Danger) >= danger_thresh,
    else Warning if warning_thresh set and P(Warning) >= warning_thresh,
    else Safe.
    """
    n = probs.shape[0]
    preds = np.zeros(n, dtype=np.int64)
    p_safe = probs[:, SAFE]
    p_warn = probs[:, WARNING]
    p_danger = probs[:, DANGER]

    danger_mask = p_danger >= danger_thresh
    preds[danger_mask] = DANGER

    remaining = ~danger_mask
    if warning_thresh is not None:
        warn_mask = remaining & (p_warn >= warning_thresh)
        preds[warn_mask] = WARNING
        remaining = remaining & ~warn_mask

    preds[remaining] = SAFE
    return preds


def class_recall_precision(
    y_true: np.ndarray, y_pred: np.ndarray, cls: int
) -> tuple[float, float]:
    tp = int(((y_true == cls) & (y_pred == cls)).sum())
    fn = int(((y_true == cls) & (y_pred != cls)).sum())
    fp = int(((y_true != cls) & (y_pred == cls)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return recall, precision


def false_positive_rate_safe_to_danger(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    safe_mask = y_true == SAFE
    if safe_mask.sum() == 0:
        return 0.0
    return float((y_pred[safe_mask] == DANGER).sum() / safe_mask.sum())


def evaluate_policy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    policy_name: str = "policy",
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    danger_recall, danger_precision = class_recall_precision(y_true, y_pred, DANGER)
    warn_recall, warn_precision = class_recall_precision(y_true, y_pred, WARNING)
    return {
        "policy": policy_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "danger_recall": danger_recall,
        "danger_precision": danger_precision,
        "warning_recall": warn_recall,
        "warning_precision": warn_precision,
        "safe_to_danger_fpr": false_positive_rate_safe_to_danger(y_true, y_pred),
    }


def sweep_danger_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray | None = None,
    min_precision: float = 0.0,
) -> dict:
    """Find danger_thresh on validation set maximizing Danger recall."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.55, 0.01)

    best = {"danger_thresh": 0.15, "danger_recall": 0.0, "danger_precision": 0.0}
    sweep_results: list[dict] = []

    for t in thresholds:
        preds = asymmetric_predict(probs, danger_thresh=float(t))
        dr, dp = class_recall_precision(y_true, preds, DANGER)
        entry = {
            "danger_thresh": float(t),
            "danger_recall": dr,
            "danger_precision": dp,
            "safe_to_danger_fpr": false_positive_rate_safe_to_danger(y_true, preds),
        }
        sweep_results.append(entry)
        if dp >= min_precision and dr >= best["danger_recall"]:
            best = {
                "danger_thresh": float(t),
                "danger_recall": dr,
                "danger_precision": dp,
            }

    return {"best": best, "sweep": sweep_results}


def is_danger_alarm(y_pred: np.ndarray) -> np.ndarray:
    """Binary danger alarm from class predictions."""
    return (y_pred == DANGER).astype(bool)

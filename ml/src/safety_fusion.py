"""Multi-layer safety fusion: NPU soft-AI OR hard threshold logic."""

from __future__ import annotations

from typing import Any

import numpy as np

from inference_policy import DANGER, SAFE, WARNING, class_recall_precision, evaluate_policy

# Channel indices in 9-channel input (8 sensors + transient at index 8)
CH_PT08_S1 = 0
CH_TRANSIENT = 8


def denormalize_channel(
    x_norm: np.ndarray, mean: float, std: float
) -> np.ndarray:
    std_safe = std if std > 1e-8 else 1.0
    return x_norm * std_safe + mean


def compute_hard_thresholds(
    x_train: np.ndarray,
    y_train: np.ndarray,
    norm_stats: dict[str, Any],
    percentile: float = 90.0,
) -> dict[str, float]:
    """
    Calibrate static thresholds on train Danger windows only.
    Uses denormalized PT08.S1 and transient at last timestep.
    """
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    danger_mask = y_train == DANGER
    if danger_mask.sum() == 0:
        return {"s1_thresh": 0.0, "transient_thresh": 0.0}

    danger_windows = x_train[danger_mask]
    s1_last = denormalize_channel(
        danger_windows[:, CH_PT08_S1, -1], mean[CH_PT08_S1], std[CH_PT08_S1]
    )
    tr_last = denormalize_channel(
        danger_windows[:, CH_TRANSIENT, -1], mean[CH_TRANSIENT], std[CH_TRANSIENT]
    )

    return {
        "s1_thresh": float(np.percentile(s1_last, 100 - percentile)),
        "transient_thresh": float(np.percentile(tr_last, 100 - percentile)),
        "calibration_percentile": percentile,
    }


def hard_logic_trigger(
    x_windows: np.ndarray,
    thresholds: dict[str, float],
    norm_stats: dict[str, Any],
) -> np.ndarray:
    """
    Deployable hard-logic: OR of raw sensor rules (no CO(GT)).
    x_windows: (N, C, W) normalized
    Returns: (N,) bool hard alarm
    """
    mean = norm_stats["mean"]
    std = norm_stats["std"]

    s1_last = denormalize_channel(
        x_windows[:, CH_PT08_S1, -1], mean[CH_PT08_S1], std[CH_PT08_S1]
    )
    tr_last = denormalize_channel(
        x_windows[:, CH_TRANSIENT, -1], mean[CH_TRANSIENT], std[CH_TRANSIENT]
    )

    s1_alarm = s1_last >= thresholds["s1_thresh"]
    tr_alarm = tr_last >= thresholds["transient_thresh"]
    return s1_alarm | tr_alarm


def fuse_alarms(
    npu_pred: np.ndarray,
    hard_alarm: np.ndarray,
) -> np.ndarray:
    """
    Final alarm: OR(soft NPU danger, hard logic).
    Output class: DANGER if fused alarm else keep npu_pred (or SAFE if only binary needed).
    """
    npu_danger = npu_pred == DANGER
    fused_danger = npu_danger | hard_alarm
    out = npu_pred.copy()
    out[fused_danger] = DANGER
    return out


def system_danger_recall(y_true: np.ndarray, final_pred: np.ndarray) -> float:
    recall, _ = class_recall_precision(y_true, final_pred, DANGER)
    return recall


def evaluate_fusion_paths(
    y_true: np.ndarray,
    npu_pred: np.ndarray,
    hard_alarm: np.ndarray,
    fused_pred: np.ndarray,
) -> dict[str, Any]:
    soft_only = npu_pred.copy()
    hard_only = np.full_like(npu_pred, SAFE)
    hard_only[hard_alarm] = DANGER

    return {
        "argmax_or_soft": evaluate_policy(y_true, soft_only, "npu_soft"),
        "hard_only": evaluate_policy(y_true, hard_only, "hard_only"),
        "fused_or": evaluate_policy(y_true, fused_pred, "fused_or"),
        "system_danger_recall_fused": system_danger_recall(y_true, fused_pred),
        "hard_trigger_rate": float(hard_alarm.mean()),
    }

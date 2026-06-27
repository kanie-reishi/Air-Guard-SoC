"""Classification losses with class imbalance handling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def class_balanced_alpha(class_counts: list[int], num_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency weights normalized to sum to num_classes."""
    counts = torch.tensor(class_counts, dtype=torch.float32)
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        log_pt = (log_probs * targets_one_hot).sum(dim=1)
        focal_weight = (1.0 - pt).pow(self.gamma)
        loss = -focal_weight * log_pt
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.class_weights)


SAFETY_FOCAL_ALPHA = [0.05, 0.15, 0.80]
SAFETY_FOCAL_GAMMA = 3.0


def build_classification_loss(
    loss_name: str,
    class_counts: list[int],
    device: torch.device,
    gamma: float = 2.0,
    focal_alpha: list[float] | None = None,
    focal_preset: str | None = None,
) -> tuple[nn.Module, dict]:
    meta: dict = {"gamma": gamma}
    if focal_preset == "safety":
        alpha = torch.tensor(SAFETY_FOCAL_ALPHA, dtype=torch.float32).to(device)
        gamma = SAFETY_FOCAL_GAMMA
        meta = {"gamma": gamma, "focal_alpha": SAFETY_FOCAL_ALPHA, "focal_preset": "safety"}
    elif focal_alpha is not None:
        alpha = torch.tensor(focal_alpha, dtype=torch.float32).to(device)
        meta["focal_alpha"] = focal_alpha
    else:
        alpha = class_balanced_alpha(class_counts).to(device)
        meta["focal_alpha"] = alpha.cpu().tolist()

    if loss_name == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha).to(device), meta
    if loss_name == "weighted_ce":
        return WeightedCrossEntropyLoss(alpha).to(device), meta
    if loss_name == "ce":
        return nn.CrossEntropyLoss().to(device), meta
    raise ValueError(f"Unknown loss: {loss_name}")

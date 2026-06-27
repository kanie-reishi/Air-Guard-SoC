"""MINA Mini InceptionNet 1D-CNN with multi-task heads and SE attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinaInceptionBlock1D(nn.Module):
    """MINA Inception Block: five parallel branches (1x1, 1x1, 3x1, 5x1, pool+1x1)."""

    def __init__(self, in_channels: int, filters_per_branch: int = 4):
        super().__init__()
        b = filters_per_branch
        self.out_channels = 5 * b

        self.branch_1x1_a = nn.Sequential(
            nn.Conv1d(in_channels, b, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        self.branch_1x1_b = nn.Sequential(
            nn.Conv1d(in_channels, b, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        self.branch_3 = nn.Sequential(
            nn.Conv1d(in_channels, b, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.branch_5 = nn.Sequential(
            nn.Conv1d(in_channels, b, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
        )
        self.branch_7 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, b, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.branch_1x1_a(x),
                self.branch_1x1_b(x),
                self.branch_3(x),
                self.branch_5(x),
                self.branch_7(x),
            ],
            dim=1,
        )


class MinaBackbone(nn.Module):
    """MINA stem + inception blocks + residual; outputs feature map (B, C, L)."""

    def __init__(
        self,
        in_channels: int = 9,
        stem_filters: int = 8,
        branch_filters: tuple[int, int] = (4, 8),
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_filters, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
        )
        self.block1 = MinaInceptionBlock1D(stem_filters, filters_per_branch=branch_filters[0])
        self.block2 = MinaInceptionBlock1D(
            self.block1.out_channels, filters_per_branch=branch_filters[1]
        )
        self.residual_proj = nn.Conv1d(
            stem_filters, self.block2.out_channels, kernel_size=1, bias=False
        )
        self.out_channels = self.block2.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem_out = self.stem(x)
        x = self.block1(stem_out)
        x = self.block2(x)
        res = self.residual_proj(stem_out)
        if res.shape[-1] != x.shape[-1]:
            min_len = min(res.shape[-1], x.shape[-1])
            res = res[..., :min_len]
            x = x[..., :min_len]
        return F.relu(x + res)


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation channel attention for 1D feature maps."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class MinaMultiTask1D(nn.Module):
    """
    MINA backbone + SE attention + classification and CO regression heads.

    forward(x) -> (logits_cls, co_pred)
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 3,
        stem_filters: int = 8,
        branch_filters: tuple[int, int] = (4, 8),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = MinaBackbone(in_channels, stem_filters, branch_filters)
        ch = self.backbone.out_channels
        self.se = SEBlock1D(ch)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.cls_head = nn.Linear(ch, num_classes)
        self.reg_head = nn.Linear(ch, 1)
        self._out_channels = ch

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.se(x)
        x = self.pool(x).flatten(1)
        return self.dropout(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.forward_features(x)
        logits = self.cls_head(feats)
        co_pred = self.reg_head(feats).squeeze(-1)
        return logits, co_pred

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        """Classification-only forward (for Q1.15 eval)."""
        return self.cls_head(self.forward_features(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MinaMiniInception1D(MinaMultiTask1D):
    """Backward-compatible alias: classification-only via forward_cls."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_cls(x)


def _conv_output_length(length: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    return (length + 2 * padding - kernel) // stride + 1


def estimate_macs(
    model: MinaMultiTask1D | MinaMiniInception1D,
    in_channels: int,
    window_size: int,
) -> int:
    macs = 0
    length = window_size
    b1 = model.backbone.block1.out_channels // 5
    b2 = model.backbone.block2.out_channels // 5
    ch = model.backbone.out_channels

    def add_conv(c_in: int, c_out: int, k: int, y_in: int, stride: int = 1) -> int:
        nonlocal macs
        pad = k // 2
        y_out = _conv_output_length(y_in, k, stride, pad)
        macs += c_in * k * y_out * c_out
        return y_out

    length = add_conv(in_channels, 8, 7, length, stride=2)

    def add_inception_block(c_in: int, b: int, y: int) -> int:
        nonlocal macs
        macs += c_in * 1 * y * b * 2
        macs += c_in * 3 * y * b
        macs += c_in * 5 * y * b
        macs += c_in * 1 * y * b
        return 5 * b

    c = add_inception_block(8, b1, length)
    c = add_inception_block(c, b2, length)
    macs += 8 * 1 * length * c
    macs += ch * (ch // 4) * 2  # SE fc approx
    macs += ch * 3 + ch * 1  # cls + reg heads
    return macs


def build_model(
    in_channels: int,
    num_classes: int = 3,
    model_type: str = "mina",
    multi_task: bool = True,
    stem_filters: int = 8,
    branch_filters: tuple[int, int] = (4, 8),
    dropout: float = 0.2,
    **kwargs,
) -> MinaMultiTask1D:
    if model_type != "mina":
        raise ValueError(f"Unsupported model_type: {model_type!r}")
    return MinaMultiTask1D(
        in_channels=in_channels,
        num_classes=num_classes,
        stem_filters=stem_filters,
        branch_filters=branch_filters,
        dropout=dropout,
    )

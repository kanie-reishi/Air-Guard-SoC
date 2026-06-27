"""Training loop for MINA multi-task 1D-CNN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import DatasetConfig, denormalize_co, make_loaders, prepare_datasets, save_meta
from losses import build_classification_loss
from model import build_model, estimate_macs

ML_DIR = SRC_DIR.parent
RESULTS_DIR = ML_DIR / "results"
DEFAULT_CKPT = RESULTS_DIR / "best_model.pt"
DANGER_CLASS = 2


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _moving_average(values: list[float], window: int = 3) -> float:
    if not values:
        return 0.0
    w = min(window, len(values))
    return float(np.mean(values[-w:]))


def _unpack_batch(batch: tuple, device: torch.device):
    if len(batch) == 3:
        xb, y_cls, y_reg = batch
        return xb.to(device), y_cls.to(device), y_reg.to(device).float()
    xb, y_cls = batch
    return xb.to(device), y_cls.to(device), None


def _model_outputs(model: nn.Module, xb: torch.Tensor, multi_task: bool):
    out = model(xb)
    if multi_task and isinstance(out, tuple):
        return out[0], out[1]
    if isinstance(out, tuple):
        return out[0], None
    return out, None


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    cls_criterion: nn.Module,
    reg_criterion: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    multi_task: bool,
    alpha: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        xb, y_cls, y_reg = _unpack_batch(batch, device)
        optimizer.zero_grad()
        logits, co_pred = _model_outputs(model, xb, multi_task)
        loss = cls_criterion(logits, y_cls)
        if multi_task and y_reg is not None and co_pred is not None:
            loss = loss + alpha * reg_criterion(co_pred, y_reg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == y_cls).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    cls_criterion: nn.Module,
    reg_criterion: nn.Module | None,
    device: torch.device,
    multi_task: bool,
    alpha: float,
    co_stats: dict[str, float] | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    reg_errs: list[np.ndarray] = []

    for batch in loader:
        xb, y_cls, y_reg = _unpack_batch(batch, device)
        logits, co_pred = _model_outputs(model, xb, multi_task)
        loss = cls_criterion(logits, y_cls)
        if multi_task and y_reg is not None and co_pred is not None:
            loss = loss + alpha * reg_criterion(co_pred, y_reg)
        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y_cls).sum().item()
        total += xb.size(0)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(y_cls.cpu().numpy())
        if multi_task and y_reg is not None and co_pred is not None and co_stats:
            pred_ppm = denormalize_co(co_pred.cpu().numpy(), co_stats["mean"], co_stats["std"])
            true_ppm = denormalize_co(y_reg.cpu().numpy(), co_stats["mean"], co_stats["std"])
            reg_errs.append(np.abs(pred_ppm - true_ppm))

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    danger_mask = y_true == DANGER_CLASS
    danger_recall = float((y_pred[danger_mask] == DANGER_CLASS).mean()) if danger_mask.any() else 0.0
    metrics = {
        "loss": total_loss / total,
        "acc": correct / total,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "danger_recall": danger_recall,
    }
    if reg_errs:
        errs = np.concatenate(reg_errs)
        metrics["reg_mae_ppm"] = float(errs.mean())
    return metrics


@torch.no_grad()
def evaluate_checkpoint_on_test(ckpt: dict[str, Any], device: str = "cpu") -> dict[str, float]:
    """Evaluate a training checkpoint on the held-out test split."""
    device_t = torch.device(device)
    cfg_dict = ckpt.get("config", ckpt["meta"].get("config", {}))
    config = DatasetConfig(**{k: v for k, v in cfg_dict.items() if k in DatasetConfig.__dataclass_fields__})
    data = prepare_datasets(config=config)
    loaders = make_loaders(data, batch_size=128, multi_task=ckpt.get("multi_task", True))

    model = build_model(
        in_channels=ckpt["num_channels"],
        num_classes=ckpt["num_classes"],
        multi_task=ckpt.get("multi_task", True),
    ).to(device_t)
    model.load_state_dict(ckpt["model_state_dict"])

    class_counts = data["meta"]["class_distribution"]["train"]
    cls_criterion, _ = build_classification_loss(
        ckpt.get("loss_name", "focal"), class_counts, device_t,
        gamma=ckpt.get("focal_gamma", 2.0),
        focal_alpha=ckpt.get("focal_alpha"),
        focal_preset=ckpt.get("focal_preset"),
    )
    reg_criterion = nn.SmoothL1Loss() if config.multi_task else None
    co_stats = data["meta"].get("co_reg_stats")
    return evaluate_epoch(
        model, loaders["test"], cls_criterion, reg_criterion,
        device_t, config.multi_task, ckpt.get("alpha", 0.3), co_stats,
    )


def run_training(
    config: DatasetConfig,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler: str = "cosine",
    patience: int = 8,
    smooth_window: int = 3,
    loss_name: str = "focal",
    alpha: float = 0.3,
    focal_gamma: float = 2.0,
    focal_alpha: list[float] | None = None,
    focal_preset: str | None = None,
    device: str = "cpu",
    out: Path | None = None,
    verbose: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    if seed is not None:
        set_seed(seed)

    device_t = torch.device(device)
    data = prepare_datasets(config=config)
    loaders = make_loaders(data, batch_size=batch_size)

    model = build_model(
        in_channels=data["num_channels"],
        num_classes=data["num_classes"],
        multi_task=config.multi_task,
    ).to(device_t)

    n_params = model.count_parameters()
    n_macs = estimate_macs(model, data["num_channels"], data["window_size"])
    if verbose:
        print(f"Parameters: {n_params:,}")
        print(f"Estimated MACs/inference: {n_macs:,}")

    class_counts = data["meta"]["class_distribution"]["train"]
    cls_criterion, loss_meta = build_classification_loss(
        loss_name, class_counts, device_t,
        gamma=focal_gamma, focal_alpha=focal_alpha, focal_preset=focal_preset,
    )
    reg_criterion = nn.SmoothL1Loss() if config.multi_task else None
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    co_stats = data["meta"].get("co_reg_stats")

    lr_scheduler: CosineAnnealingLR | ReduceLROnPlateau | None
    if scheduler == "plateau":
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    elif scheduler == "none":
        lr_scheduler = None
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_macro_f1_smooth": [],
        "val_reg_mae": [],
        "lr": [],
    }
    best_smooth_f1 = -1.0
    best_raw_f1 = -1.0
    best_state: dict | None = None
    best_epoch = 0
    epochs_no_improve = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], cls_criterion, reg_criterion,
            optimizer, device_t, config.multi_task, alpha,
        )
        val_m = evaluate_epoch(
            model, loaders["val"], cls_criterion, reg_criterion,
            device_t, config.multi_task, alpha, co_stats,
        )
        smooth_f1 = _moving_average(history["val_macro_f1"] + [val_m["macro_f1"]], smooth_window)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_m["loss"])
        history["val_acc"].append(val_m["acc"])
        history["val_macro_f1"].append(val_m["macro_f1"])
        history["val_macro_f1_smooth"].append(smooth_f1)
        history["val_reg_mae"].append(val_m.get("reg_mae_ppm", 0.0))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if lr_scheduler is not None:
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(val_m["macro_f1"])
            else:
                lr_scheduler.step()

        if verbose:
            msg = (
                f"Epoch {epoch:02d}/{epochs} | "
                f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                f"val loss={val_m['loss']:.4f} acc={val_m['acc']:.4f} "
                f"macro-F1={val_m['macro_f1']:.4f} smooth={smooth_f1:.4f}"
            )
            if "reg_mae_ppm" in val_m:
                msg += f" CO-MAE={val_m['reg_mae_ppm']:.3f}ppm"
            print(msg)

        if smooth_f1 > best_smooth_f1:
            best_smooth_f1 = smooth_f1
            best_raw_f1 = val_m["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
            stopped_early = True
            break

    assert best_state is not None
    checkpoint = {
        "model_state_dict": best_state,
        "model_type": "mina",
        "multi_task": config.multi_task,
        "config": config.__dict__,
        "meta": data["meta"],
        "num_channels": data["num_channels"],
        "window_size": data["window_size"],
        "num_classes": data["num_classes"],
        "n_params": n_params,
        "n_macs": n_macs,
        "best_val_macro_f1": best_raw_f1,
        "best_val_macro_f1_smooth": best_smooth_f1,
        "best_val_acc": max(history["val_acc"]),
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "history": history,
        "loss_name": loss_name,
        "alpha": alpha,
        "weight_decay": weight_decay,
        "scheduler": scheduler,
        "patience": patience,
        "smooth_window": smooth_window,
        "focal_gamma": loss_meta.get("gamma", focal_gamma),
        "focal_alpha": loss_meta.get("focal_alpha"),
        "focal_preset": loss_meta.get("focal_preset"),
        "seed": seed,
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, out)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MINA multi-task 1D")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["cosine", "plateau", "none"], default="cosine")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (0=off)")
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-env", action="store_true")
    parser.add_argument("--no-transient", action="store_true")
    parser.add_argument("--no-multi-task", action="store_true")
    parser.add_argument("--loss", choices=["focal", "weighted_ce", "ce"], default="focal")
    parser.add_argument("--focal-preset", choices=["safety", "balanced"], default="balanced")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, nargs=3, default=None)
    parser.add_argument("--alpha", type=float, default=0.3, help="Regression loss weight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=DEFAULT_CKPT)
    args = parser.parse_args()

    set_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = DatasetConfig(
        window_size=args.window_size,
        stride=args.stride,
        use_env_channels=not args.no_env,
        use_transient=not args.no_transient,
        multi_task=not args.no_multi_task,
        random_seed=args.seed,
    )
    save_meta(prepare_datasets(config=config)["meta"], RESULTS_DIR / "dataset_meta.json")

    focal_preset = args.focal_preset if args.focal_preset == "safety" else None
    ckpt = run_training(
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        patience=args.patience,
        smooth_window=args.smooth_window,
        loss_name=args.loss,
        alpha=args.alpha,
        focal_gamma=args.focal_gamma if focal_preset is None else 3.0,
        focal_alpha=list(args.focal_alpha) if args.focal_alpha else None,
        focal_preset=focal_preset,
        device=args.device,
        out=args.out,
        seed=args.seed,
    )
    with open(RESULTS_DIR / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(ckpt["history"], f, indent=2)
    print(f"Best val macro-F1: {ckpt['best_val_macro_f1']:.4f} (epoch {ckpt['best_epoch']})")
    print(f"Checkpoint saved: {args.out}")


if __name__ == "__main__":
    main()

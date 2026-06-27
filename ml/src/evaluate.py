"""Evaluation: classification, safety policies, and fusion metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import CLASS_NAMES, DatasetConfig, denormalize_co, majority_baseline_accuracy, make_loaders, prepare_datasets
from inference_policy import (
    argmax_predict,
    asymmetric_predict,
    evaluate_policy,
    softmax_probs,
    sweep_danger_threshold,
)
from model import build_model
from safety_fusion import (
    compute_hard_thresholds,
    evaluate_fusion_paths,
    fuse_alarms,
    hard_logic_trigger,
)

ML_DIR = SRC_DIR.parent
RESULTS_DIR = ML_DIR / "results"
DEFAULT_CKPT = RESULTS_DIR / "best_model.pt"


@torch.no_grad()
def collect_logits_and_data(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    multi_task: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    co_preds: list[np.ndarray] = []
    co_true: list[np.ndarray] = []

    for batch in loader:
        if len(batch) == 3:
            xb, y_cls, y_reg = batch
            co_true.append(y_reg.numpy())
        else:
            xb, y_cls = batch
        xb = xb.to(device)
        out = model(xb)
        if multi_task and isinstance(out, tuple):
            logits, co_pred = out
            co_preds.append(co_pred.cpu().numpy())
        else:
            logits = out[0] if isinstance(out, tuple) else out
        all_logits.append(logits.cpu().numpy())
        all_x.append(xb.cpu().numpy())
        all_y.append(y_cls.numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    x_np = np.concatenate(all_x, axis=0)
    y_np = np.concatenate(all_y, axis=0)
    co_pred_np = np.concatenate(co_preds) if co_preds else None
    co_true_np = np.concatenate(co_true) if co_true else None
    return logits_np, x_np, y_np, co_pred_np, co_true_np


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_sweep(sweep: list[dict], out_path: Path) -> None:
    recalls = [s["danger_recall"] for s in sweep]
    precisions = [s["danger_precision"] for s in sweep]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recalls, precisions, "b-o", markersize=3)
    ax.set_xlabel("Danger Recall")
    ax.set_ylabel("Danger Precision")
    ax.set_title("Asymmetric Threshold Sweep (Val)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_safety_evaluation(
    ckpt_path: Path,
    device: str = "cpu",
    danger_thresh: float | None = None,
    min_precision: float = 0.0,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    class_names = meta.get("class_names", CLASS_NAMES)
    co_stats = meta.get("co_reg_stats")
    multi_task = ckpt.get("multi_task", True)
    norm_stats = meta.get("norm_stats", {})

    cfg = ckpt.get("config", meta.get("config", {}))
    config = DatasetConfig(**{k: v for k, v in cfg.items() if k in DatasetConfig.__dataclass_fields__})
    data = prepare_datasets(config=config)

    model = build_model(
        in_channels=ckpt["num_channels"],
        num_classes=ckpt["num_classes"],
        model_type=ckpt.get("model_type", "mina"),
        multi_task=multi_task,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    device_t = torch.device(device)
    model = model.to(device_t)

    loaders = make_loaders(data, batch_size=128, multi_task=multi_task)

    val_logits, val_x, val_y, _, _ = collect_logits_and_data(
        model, loaders["val"], device_t, multi_task
    )
    test_logits, test_x, test_y, co_pred_norm, co_true_norm = collect_logits_and_data(
        model, loaders["test"], device_t, multi_task
    )

    val_probs = softmax_probs(val_logits)
    test_probs = softmax_probs(test_logits)

    sweep = sweep_danger_threshold(val_probs, val_y, min_precision=min_precision)
    chosen_thresh = danger_thresh if danger_thresh is not None else sweep["best"]["danger_thresh"]

    test_argmax = argmax_predict(test_probs)
    test_asym = asymmetric_predict(test_probs, danger_thresh=chosen_thresh)

    hard_thresh = compute_hard_thresholds(
        data["x_train"], data["y_train"], norm_stats
    )
    test_hard = hard_logic_trigger(test_x, hard_thresh, norm_stats)
    test_fused = fuse_alarms(test_asym, test_hard)

    results = {
        "danger_thresh": chosen_thresh,
        "hard_thresholds": hard_thresh,
        "val_sweep_best": sweep["best"],
        "test_argmax": evaluate_policy(test_y, test_argmax, "argmax"),
        "test_asymmetric": evaluate_policy(test_y, test_asym, "asymmetric"),
        "test_fusion": evaluate_fusion_paths(test_y, test_asym, test_hard, test_fused),
        "confusion_matrix_argmax": confusion_matrix(test_y, test_argmax, labels=[0, 1, 2]).tolist(),
        "confusion_matrix_asymmetric": confusion_matrix(test_y, test_asym, labels=[0, 1, 2]).tolist(),
        "confusion_matrix_fused": confusion_matrix(test_y, test_fused, labels=[0, 1, 2]).tolist(),
        "classification_report_fused": classification_report(
            test_y, test_fused, target_names=class_names, zero_division=0
        ),
        "model_type": ckpt.get("model_type"),
        "multi_task": multi_task,
        "window_size": ckpt.get("window_size"),
        "n_params": ckpt.get("n_params"),
        "focal_preset": ckpt.get("focal_preset"),
    }

    if co_pred_norm is not None and co_true_norm is not None and co_stats:
        co_pred_ppm = denormalize_co(co_pred_norm, co_stats["mean"], co_stats["std"])
        co_true_ppm = denormalize_co(co_true_norm, co_stats["mean"], co_stats["std"])
        results["co_mae_ppm"] = float(mean_absolute_error(co_true_ppm, co_pred_ppm))
        results["co_rmse_ppm"] = float(np.sqrt(mean_squared_error(co_true_ppm, co_pred_ppm)))

    return results, sweep["sweep"], class_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MINA with safety policies")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--danger-thresh", type=float, default=None)
    parser.add_argument("--min-precision", type=float, default=0.0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results, sweep, class_names = run_safety_evaluation(
        args.ckpt, args.device, args.danger_thresh, args.min_precision
    )

    print("=== Safety-Critical Evaluation (Test) ===")
    print(f"Danger threshold (from val sweep): {results['danger_thresh']:.3f}")
    print()
    for key in ("test_argmax", "test_asymmetric"):
        m = results[key]
        print(f"[{m['policy']}] acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} "
              f"Danger Recall={m['danger_recall']:.4f} Precision={m['danger_precision']:.4f} "
              f"Safe->Danger FPR={m['safe_to_danger_fpr']:.4f}")
    print()
    fused = results["test_fusion"]["fused_or"]
    print(f"[fused OR] acc={fused['accuracy']:.4f} macro-F1={fused['macro_f1']:.4f} "
          f"Danger Recall={fused['danger_recall']:.4f} Precision={fused['danger_precision']:.4f} "
          f"Safe->Danger FPR={fused['safe_to_danger_fpr']:.4f}")
    print(f"System Danger Recall (fused): {results['test_fusion']['system_danger_recall_fused']:.4f}")
    if "co_mae_ppm" in results:
        print(f"CO MAE (ppm): {results['co_mae_ppm']:.4f}")
    print()
    print(results["classification_report_fused"])

    with open(RESULTS_DIR / "safety_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS_DIR / "threshold_sweep.json", "w", encoding="utf-8") as f:
        json.dump({"sweep": sweep, "chosen": results["danger_thresh"]}, f, indent=2)

    plot_confusion_matrix(
        np.array(results["confusion_matrix_fused"]),
        class_names,
        RESULTS_DIR / "confusion_matrix_fused.png",
        "Fused OR (Test)",
    )
    plot_pr_sweep(sweep, RESULTS_DIR / "danger_pr_sweep.png")

    # Keep legacy eval_metrics.json for compatibility
    legacy = {
        "test_accuracy": results["test_asymmetric"]["accuracy"],
        "test_macro_f1": results["test_asymmetric"]["macro_f1"],
        "test_f1_per_class": {
            class_names[i]: results["test_asymmetric"].get(f"{class_names[i].lower()}_f1", 0)
            for i in range(3)
        },
        "danger_recall_argmax": results["test_argmax"]["danger_recall"],
        "danger_recall_asymmetric": results["test_asymmetric"]["danger_recall"],
        "danger_recall_fused": results["test_fusion"]["fused_or"]["danger_recall"],
        "confusion_matrix": results["confusion_matrix_fused"],
        "classification_report": results["classification_report_fused"],
        **{k: results[k] for k in ("co_mae_ppm", "co_rmse_ppm", "n_params", "window_size") if k in results},
    }
    with open(RESULTS_DIR / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(legacy, f, indent=2)

    print(f"\nSaved safety metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

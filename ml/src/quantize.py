"""Post-training quantization to Q1.15 (16-bit signed fixed-point)."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import DatasetConfig, make_loaders, prepare_datasets
from model import MinaMultiTask1D, build_model

ML_DIR = SRC_DIR.parent
RESULTS_DIR = ML_DIR / "results"
DEFAULT_CKPT = RESULTS_DIR / "best_model.pt"

Q15_SCALE = 32768.0
Q15_MIN = -32768
Q15_MAX = 32767


def float_to_q15(x: np.ndarray | float) -> np.ndarray:
    q = np.round(np.asarray(x, dtype=np.float64) * Q15_SCALE)
    return np.clip(q, Q15_MIN, Q15_MAX).astype(np.int16)


def q15_to_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) / Q15_SCALE


def quantize_model_weights(model: nn.Module) -> nn.Module:
    qmodel = copy.deepcopy(model)
    qmodel.eval()
    with torch.no_grad():
        for module in qmodel.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                w_np = module.weight.detach().cpu().numpy()
                module.weight.data = torch.from_numpy(
                    q15_to_float(float_to_q15(w_np))
                ).to(module.weight.device)
                if module.bias is not None:
                    b_np = module.bias.detach().cpu().numpy()
                    module.bias.data = torch.from_numpy(
                        q15_to_float(float_to_q15(b_np))
                    ).to(module.bias.device)
    return qmodel


@torch.no_grad()
def evaluate_cls_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    multi_task: bool,
) -> tuple[float, float]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        xb = batch[0].to(device)
        y_cls = batch[1]
        if multi_task and hasattr(model, "forward_cls"):
            logits = model.forward_cls(xb)
        else:
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(y_cls.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    return accuracy_score(y_true, y_pred), f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )


def export_q15_weights(model: nn.Module, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    idx = 0
    for name, param in model.named_parameters():
        q = float_to_q15(param.detach().cpu().numpy())
        flat = q.flatten()
        fname = f"weight_{idx:02d}_{name.replace('.', '_')}.npy"
        np.save(out_dir / fname, flat)
        manifest[name] = {
            "file": fname,
            "shape": list(param.shape),
            "q15_min": int(flat.min()),
            "q15_max": int(flat.max()),
        }
        idx += 1
    with open(out_dir / "q15_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q1.15 PTQ for MINA multi-task")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    multi_task = ckpt.get("multi_task", True)

    cfg = ckpt.get("config", ckpt["meta"].get("config", {}))
    config = DatasetConfig(**{k: v for k, v in cfg.items() if k in DatasetConfig.__dataclass_fields__})
    data = prepare_datasets(config=config)
    loaders = make_loaders(data, batch_size=128, multi_task=multi_task)
    device = torch.device(args.device)

    float_model = build_model(
        in_channels=ckpt["num_channels"],
        num_classes=ckpt["num_classes"],
        model_type=ckpt.get("model_type", "mina"),
        multi_task=multi_task,
    )
    float_model.load_state_dict(ckpt["model_state_dict"])
    float_model = float_model.to(device)

    q_model = quantize_model_weights(float_model)
    q_model = q_model.to(device)

    float_acc, float_f1 = evaluate_cls_model(float_model, loaders["test"], device, multi_task)
    q_acc, q_f1 = evaluate_cls_model(q_model, loaders["test"], device, multi_task)

    results = {
        "format": "Q1.15 (16-bit signed)",
        "model_type": ckpt.get("model_type", "mina"),
        "multi_task": multi_task,
        "float_test_accuracy": float(float_acc),
        "float_test_macro_f1": float(float_f1),
        "q15_test_accuracy": float(q_acc),
        "q15_test_macro_f1": float(q_f1),
        "accuracy_drop": float(float_acc - q_acc),
        "f1_drop": float(float_f1 - q_f1),
    }

    print("=== Q1.15 Post-Training Quantization (classification) ===")
    print(f"Float  — acc={float_acc:.4f}  macro-F1={float_f1:.4f}")
    print(f"Q1.15  — acc={q_acc:.4f}  macro-F1={q_f1:.4f}")
    print(f"Drop   — acc={results['accuracy_drop']:.4f}  F1={results['f1_drop']:.4f}")

    with open(RESULTS_DIR / "quantize_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    export_q15_weights(q_model, RESULTS_DIR / "q15_weights")
    torch.save(q_model.state_dict(), RESULTS_DIR / "q15_model_state.pt")
    print(f"Saved Q1.15 artifacts to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

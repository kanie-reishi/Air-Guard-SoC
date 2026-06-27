"""Classical ML baselines for comparison with MINA 1D-CNN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import DatasetConfig, majority_baseline_accuracy, prepare_datasets
from train import DANGER_CLASS, evaluate_checkpoint_on_test, set_seed

ML_DIR = SRC_DIR.parent
RESULTS_DIR = ML_DIR / "results"


def flatten_windows(x: np.ndarray) -> np.ndarray:
    """(N, C, W) -> (N, C*W)."""
    return x.reshape(x.shape[0], -1)


def _danger_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true == DANGER_CLASS
    if mask.sum() == 0:
        return 0.0
    return float((y_pred[mask] == DANGER_CLASS).mean())


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "danger_recall": _danger_recall(y_true, y_pred),
    }


class MLPBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, float]:
    set_seed(seed)
    device_t = torch.device(device)
    in_dim = x_train.shape[1]
    model = MLPBaseline(in_dim).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device_t), yb.to(device_t)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test.astype(np.float32)).to(device_t))
        pred = logits.argmax(dim=1).cpu().numpy()
    return _metrics(y_test, pred)


def run_baselines(
    config: DatasetConfig | None = None,
    seed: int = 42,
    mlp_epochs: int = 30,
    device: str = "cpu",
) -> dict[str, Any]:
    config = config or DatasetConfig(window_size=3, use_transient=True, multi_task=True)
    data = prepare_datasets(config=config)

    x_train = flatten_windows(data["x_train"])
    x_val = flatten_windows(data["x_val"])
    x_test = flatten_windows(data["x_test"])
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    # Fit sklearn on train, report on test
    results: dict[str, Any] = {
        "config": config.__dict__,
        "seed": seed,
        "models": {},
    }

    maj_acc = majority_baseline_accuracy(y_test)
    maj_pred = np.full_like(y_test, fill_value=int(np.bincount(y_train).argmax()))
    results["models"]["majority"] = {
        "description": "Predict most frequent train class",
        "majority_accuracy": maj_acc,
        **_metrics(y_test, maj_pred),
    }

    lr_model = LogisticRegression(max_iter=2000, random_state=seed)
    lr_model.fit(x_train, y_train)
    results["models"]["logistic_regression"] = {
        "description": "Flattened window + LogisticRegression",
        **_metrics(y_test, lr_model.predict(x_test)),
    }

    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1,
    )
    rf_model.fit(x_train, y_train)
    results["models"]["random_forest"] = {
        "description": "Flattened window + RandomForest",
        **_metrics(y_test, rf_model.predict(x_test)),
    }

    results["models"]["mlp"] = {
        "description": "Flattened window + 1-hidden-layer MLP",
        **train_mlp(x_train, y_train, x_test, y_test, epochs=mlp_epochs, device=device, seed=seed),
    }

    # Reference MINA from checkpoint (argmax on test)
    ckpt_path = RESULTS_DIR / "best_model.pt"
    if ckpt_path.exists():
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        test_m = evaluate_checkpoint_on_test(ckpt, device=device)
        results["models"]["mina_cnn"] = {
            "description": "MINA MultiTask1D (best_model.pt, argmax)",
            "accuracy": test_m["acc"],
            "macro_f1": test_m["macro_f1"],
            "danger_recall": test_m["danger_recall"],
            "n_params": ckpt.get("n_params"),
            "n_macs": ckpt.get("n_macs"),
            "source": str(ckpt_path),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical baselines")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--no-transient", action="store_true")
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "baselines.json")
    args = parser.parse_args()

    config = DatasetConfig(
        window_size=args.window_size,
        use_transient=not args.no_transient,
        multi_task=True,
        random_seed=args.seed,
    )
    results = run_baselines(config=config, seed=args.seed, mlp_epochs=args.mlp_epochs, device=args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== Baselines (test set) ===")
    for name, m in results["models"].items():
        acc = m.get("accuracy")
        f1 = m.get("macro_f1")
        dr = m.get("danger_recall")
        if acc is not None:
            print(f"[{name}] acc={acc:.4f} macro-F1={f1:.4f} Danger Recall={dr:.4f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

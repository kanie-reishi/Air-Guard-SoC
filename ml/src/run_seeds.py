"""Multi-seed training for statistical robustness (mean +/- std)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import DatasetConfig
from train import evaluate_checkpoint_on_test, run_training

ML_DIR = SRC_DIR.parent
RESULTS_DIR = ML_DIR / "results"

DEFAULT_SEEDS = [42, 1, 7, 123, 2024]


def _summarize(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run_seed_study(
    seeds: list[int] | None = None,
    epochs: int = 30,
    device: str = "cpu",
    patience: int = 8,
    window_size: int = 3,
) -> dict[str, Any]:
    seeds = seeds or DEFAULT_SEEDS
    config = DatasetConfig(window_size=window_size, use_transient=True, multi_task=True)
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        ckpt_path = RESULTS_DIR / "seeds" / f"seed_{seed}.pt"
        ckpt = run_training(
            config=config,
            epochs=epochs,
            device=device,
            out=ckpt_path,
            verbose=True,
            seed=seed,
            patience=patience,
        )
        test_m = evaluate_checkpoint_on_test(ckpt, device=device)
        per_seed.append({
            "seed": seed,
            "val_macro_f1": ckpt["best_val_macro_f1"],
            "val_macro_f1_smooth": ckpt["best_val_macro_f1_smooth"],
            "best_epoch": ckpt["best_epoch"],
            "stopped_early": ckpt["stopped_early"],
            "test_accuracy": test_m["acc"],
            "test_macro_f1": test_m["macro_f1"],
            "test_danger_recall": test_m["danger_recall"],
            "checkpoint": str(ckpt_path),
        })
        print(
            f"  test acc={test_m['acc']:.4f} macro-F1={test_m['macro_f1']:.4f} "
            f"Danger Recall={test_m['danger_recall']:.4f}"
        )

    summary = {
        "test_accuracy": _summarize([r["test_accuracy"] for r in per_seed]),
        "test_macro_f1": _summarize([r["test_macro_f1"] for r in per_seed]),
        "test_danger_recall": _summarize([r["test_danger_recall"] for r in per_seed]),
        "val_macro_f1": _summarize([r["val_macro_f1"] for r in per_seed]),
    }

    return {
        "seeds": seeds,
        "epochs": epochs,
        "patience": patience,
        "window_size": window_size,
        "per_seed": per_seed,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed robustness study")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "seed_robustness.json")
    args = parser.parse_args()

    results = run_seed_study(
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device,
        patience=args.patience,
        window_size=args.window_size,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print("\n=== Seed robustness summary (test) ===")
    print(
        f"Accuracy:     {s['test_accuracy']['mean']:.4f} +/- {s['test_accuracy']['std']:.4f}"
    )
    print(
        f"Macro-F1:     {s['test_macro_f1']['mean']:.4f} +/- {s['test_macro_f1']['std']:.4f}"
    )
    print(
        f"Danger Recall:{s['test_danger_recall']['mean']:.4f} +/- "
        f"{s['test_danger_recall']['std']:.4f}"
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

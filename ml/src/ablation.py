"""Ablation study: quantify contribution of each pipeline improvement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import DatasetConfig
from train import evaluate_checkpoint_on_test, run_training, set_seed

ML_DIR = SRC_DIR.parent
RESULTS_DIR = ML_DIR / "results"


ABLATION_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "baseline_8ch_w24_ce",
        "description": "Original MINA: 8ch, W=24, CE loss, no transient/multi-task",
        "window_size": 24,
        "use_transient": False,
        "multi_task": False,
        "loss_name": "ce",
    },
    {
        "name": "plus_transient",
        "description": "+ transient channel (9ch)",
        "window_size": 24,
        "use_transient": True,
        "multi_task": False,
        "loss_name": "ce",
    },
    {
        "name": "plus_focal",
        "description": "+ Focal Loss",
        "window_size": 24,
        "use_transient": True,
        "multi_task": False,
        "loss_name": "focal",
    },
    {
        "name": "plus_window3",
        "description": "+ window W=3h",
        "window_size": 3,
        "use_transient": True,
        "multi_task": False,
        "loss_name": "focal",
    },
    {
        "name": "plus_multitask",
        "description": "+ multi-task CO regression",
        "window_size": 3,
        "use_transient": True,
        "multi_task": True,
        "loss_name": "focal",
    },
    {
        "name": "full_improved",
        "description": "Full improved MINA (current best config)",
        "window_size": 3,
        "use_transient": True,
        "multi_task": True,
        "loss_name": "focal",
    },
    {
        "name": "window6_compare",
        "description": "W=6h instead of W=3 (window sweep control)",
        "window_size": 6,
        "use_transient": True,
        "multi_task": True,
        "loss_name": "focal",
    },
]


def run_ablation(
    epochs: int = 15,
    seed: int = 42,
    device: str = "cpu",
    patience: int = 5,
    configs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    configs = configs or ABLATION_CONFIGS
    results: list[dict[str, Any]] = []

    for i, spec in enumerate(configs):
        name = spec["name"]
        print(f"\n--- Ablation [{i + 1}/{len(configs)}]: {name} ---")
        config = DatasetConfig(
            window_size=spec["window_size"],
            use_transient=spec["use_transient"],
            multi_task=spec["multi_task"],
            random_seed=seed,
        )
        ckpt_path = RESULTS_DIR / "ablation" / f"{name}.pt"
        ckpt = run_training(
            config=config,
            epochs=epochs,
            loss_name=spec["loss_name"],
            device=device,
            out=ckpt_path,
            verbose=True,
            seed=seed,
            patience=patience,
        )
        test_m = evaluate_checkpoint_on_test(ckpt, device=device)
        entry = {
            "name": name,
            "description": spec["description"],
            "config": {
                "window_size": spec["window_size"],
                "use_transient": spec["use_transient"],
                "multi_task": spec["multi_task"],
                "loss_name": spec["loss_name"],
            },
            "val_macro_f1": ckpt["best_val_macro_f1"],
            "val_macro_f1_smooth": ckpt["best_val_macro_f1_smooth"],
            "best_epoch": ckpt["best_epoch"],
            "test_accuracy": test_m["acc"],
            "test_macro_f1": test_m["macro_f1"],
            "test_danger_recall": test_m["danger_recall"],
            "n_params": ckpt["n_params"],
            "n_macs": ckpt["n_macs"],
            "checkpoint": str(ckpt_path),
        }
        results.append(entry)
        print(
            f"  -> test acc={test_m['acc']:.4f} macro-F1={test_m['macro_f1']:.4f} "
            f"Danger Recall={test_m['danger_recall']:.4f}"
        )

    return {
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        "runs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study for MINA improvements")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per ablation run")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action="store_true", help="Run only last 3 configs")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "ablation.json")
    args = parser.parse_args()

    configs = ABLATION_CONFIGS
    if args.quick:
        configs = ABLATION_CONFIGS[-3:]

    results = run_ablation(
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
        configs=configs,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Ablation summary (test) ===")
    for r in results["runs"]:
        print(
            f"{r['name']:20s} acc={r['test_accuracy']:.4f} "
            f"F1={r['test_macro_f1']:.4f} DR={r['test_danger_recall']:.4f}"
        )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

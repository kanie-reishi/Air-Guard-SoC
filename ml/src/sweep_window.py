"""Sweep sliding window size (6h vs 3h) by val macro-F1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import DatasetConfig
from train import run_training

ML_DIR = SRC_DIR.parent
RESULTS_DIR = ML_DIR / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep window sizes for MINA")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per window candidate")
    parser.add_argument("--windows", type=int, nargs="+", default=[6, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    best_window = args.windows[0]
    best_f1 = -1.0

    for w in args.windows:
        print(f"\n=== Window size: {w}h ===")
        config = DatasetConfig(window_size=w, random_seed=args.seed)
        ckpt_path = RESULTS_DIR / f"sweep_w{w}.pt"
        ckpt = run_training(
            config=config,
            epochs=args.epochs,
            device=args.device,
            out=ckpt_path,
            verbose=True,
        )
        entry = {
            "window_size": w,
            "best_val_macro_f1": ckpt["best_val_macro_f1"],
            "best_val_acc": ckpt["best_val_acc"],
            "n_params": ckpt["n_params"],
            "n_macs": ckpt["n_macs"],
            "checkpoint": str(ckpt_path),
        }
        results.append(entry)
        if ckpt["best_val_macro_f1"] > best_f1:
            best_f1 = ckpt["best_val_macro_f1"]
            best_window = w

    sweep_out = {
        "candidates": results,
        "best_window_size": best_window,
        "best_val_macro_f1": best_f1,
    }
    with open(RESULTS_DIR / "window_sweep.json", "w", encoding="utf-8") as f:
        json.dump(sweep_out, f, indent=2)

    print(f"\nBest window: {best_window}h (val macro-F1={best_f1:.4f})")
    print(f"Saved sweep results to {RESULTS_DIR / 'window_sweep.json'}")
    print(f"Run full training: python src/train.py --epochs 30 --window-size {best_window}")


if __name__ == "__main__":
    main()

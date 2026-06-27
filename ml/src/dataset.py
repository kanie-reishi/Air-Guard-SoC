"""UCI Air Quality dataset: cleaning, windowing, labeling, PyTorch loaders."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from download_data import DEFAULT_DATA_DIR, download_air_quality

SENSOR_COLS = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(COx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]
LABEL_COL = "CO(GT)"
MISSING_VALUE = -200.0
CLASS_NAMES = ["Safe", "Warning", "Danger"]
TRANSIENT_COL = "mean_abs_delta"

COL_ALIASES = {
    "PT08.S3(COx)": ["PT08.S3(COx)", "PT08.S3(NOx)"],
}


@dataclass
class DatasetConfig:
    window_size: int = 6
    stride: int = 1
    use_env_channels: bool = True
    use_transient: bool = True
    multi_task: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    label_percentiles: tuple[float, float] = (33.33, 66.67)
    random_seed: int = 42


@dataclass
class NormStats:
    mean: list[float]
    std: list[float]
    feature_cols: list[str]


def _resolve_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for canonical in SENSOR_COLS:
        aliases = COL_ALIASES.get(canonical, [canonical])
        found = next((a for a in aliases if a in df.columns), None)
        if found is None:
            if canonical in ("T", "RH", "AH"):
                continue
            raise KeyError(f"Missing required column. Tried: {aliases}")
        cols.append(found)
    if LABEL_COL not in df.columns:
        raise KeyError(f"Missing label column {LABEL_COL}")
    return cols


def load_raw_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        encoding="utf-8",
        low_memory=False,
    )
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    df.columns = df.columns.str.strip()
    return df


def clean_sensor_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    work = df[feature_cols + [LABEL_COL]].copy()
    for col in feature_cols + [LABEL_COL]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work.loc[work[col] == MISSING_VALUE, col] = np.nan

    work[feature_cols] = (
        work[feature_cols]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    work[LABEL_COL] = work[LABEL_COL].interpolate(limit_direction="both").ffill().bfill()
    work = work.dropna(subset=feature_cols + [LABEL_COL])
    return work.reset_index(drop=True)


def append_transient_channel(features: np.ndarray) -> np.ndarray:
    """Add channel 9: mean(|delta|) across sensors at each timestep."""
    delta = np.zeros_like(features)
    delta[1:] = features[1:] - features[:-1]
    transient = np.mean(np.abs(delta), axis=1, keepdims=True)
    return np.concatenate([features, transient], axis=1)


def compute_label_thresholds(
    co_values: np.ndarray, percentiles: tuple[float, float]
) -> tuple[float, float]:
    p1, p2 = percentiles
    t1, t2 = np.percentile(co_values, [p1, p2])
    return float(t1), float(t2)


def co_to_class(co: float, t1: float, t2: float) -> int:
    if co < t1:
        return 0
    if co < t2:
        return 1
    return 2


def build_windows(
    features: np.ndarray,
    labels_co: np.ndarray,
    window_size: int,
    stride: int,
    t1: float,
    t2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows: X (N, C, W), y_cls (N,), y_reg (N,) CO at window end."""
    n = len(features)
    xs: list[np.ndarray] = []
    y_cls: list[int] = []
    y_reg: list[float] = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        xs.append(features[start:end].T)
        co_end = float(labels_co[end - 1])
        y_cls.append(co_to_class(co_end, t1, t2))
        y_reg.append(co_end)
    return (
        np.stack(xs, axis=0).astype(np.float32),
        np.array(y_cls, dtype=np.int64),
        np.array(y_reg, dtype=np.float32),
    )


def normalize_features(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    std_safe = np.where(std < 1e-8, 1.0, std)
    mean_b = mean.reshape(1, -1, 1)
    std_b = std_safe.reshape(1, -1, 1)
    return ((x - mean_b) / std_b).astype(np.float32)


def normalize_co_targets(y_reg: np.ndarray, mean: float, std: float) -> np.ndarray:
    std_safe = std if std > 1e-8 else 1.0
    return ((y_reg - mean) / std_safe).astype(np.float32)


def denormalize_co(y_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
    std_safe = std if std > 1e-8 else 1.0
    return y_norm * std_safe + mean


def prepare_datasets(
    csv_path: Path | None = None,
    config: DatasetConfig | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    config = config or DatasetConfig()
    data_dir = data_dir or DEFAULT_DATA_DIR

    if csv_path is None:
        csv_path = download_air_quality(data_dir)

    df = load_raw_dataframe(csv_path)
    all_feature_cols = _resolve_columns(df)
    if not config.use_env_channels:
        all_feature_cols = [c for c in all_feature_cols if c not in ("T", "RH", "AH")]

    clean = clean_sensor_frame(df, all_feature_cols)
    features = clean[all_feature_cols].to_numpy(dtype=np.float64)
    if config.use_transient:
        features = append_transient_channel(features)
        all_feature_cols = all_feature_cols + [TRANSIENT_COL]

    labels_co = clean[LABEL_COL].to_numpy(dtype=np.float64)

    n = len(clean)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)
    n_test = n - n_train - n_val

    train_co = labels_co[:n_train]
    t1, t2 = compute_label_thresholds(train_co, config.label_percentiles)

    mean = features[:n_train].mean(axis=0)
    std = features[:n_train].std(axis=0)
    co_mean = float(train_co.mean())
    co_std = float(train_co.std())

    def _split_and_window(feat_slice: np.ndarray, co_slice: np.ndarray):
        x, y_cls, y_reg = build_windows(
            feat_slice, co_slice, config.window_size, config.stride, t1, t2
        )
        x = normalize_features(x, mean, std)
        if config.multi_task:
            y_reg = normalize_co_targets(y_reg, co_mean, co_std)
        return x, y_cls, y_reg

    x_train, y_train, yr_train = _split_and_window(features[:n_train], labels_co[:n_train])
    x_val, y_val, yr_val = _split_and_window(
        features[n_train : n_train + n_val], labels_co[n_train : n_train + n_val]
    )
    x_test, y_test, yr_test = _split_and_window(
        features[n_train + n_val :], labels_co[n_train + n_val :]
    )

    norm_stats = NormStats(
        mean=mean.tolist(),
        std=std.tolist(),
        feature_cols=all_feature_cols,
    )

    meta = {
        "config": asdict(config),
        "label_thresholds": {"t1_safe_warning": t1, "t2_warning_danger": t2},
        "class_names": CLASS_NAMES,
        "norm_stats": asdict(norm_stats),
        "co_reg_stats": {"mean": co_mean, "std": co_std},
        "transient_channel": TRANSIENT_COL if config.use_transient else None,
        "shapes": {
            "train": list(x_train.shape),
            "val": list(x_val.shape),
            "test": list(x_test.shape),
        },
        "class_distribution": {
            "train": np.bincount(y_train, minlength=3).tolist(),
            "val": np.bincount(y_val, minlength=3).tolist(),
            "test": np.bincount(y_test, minlength=3).tolist(),
        },
    }

    result: dict[str, Any] = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "meta": meta,
        "num_channels": x_train.shape[1],
        "window_size": config.window_size,
        "num_classes": 3,
        "multi_task": config.multi_task,
    }
    if config.multi_task:
        result.update({
            "yr_train": yr_train,
            "yr_val": yr_val,
            "yr_test": yr_test,
        })
    return result


def make_loaders(
    data: dict[str, Any],
    batch_size: int = 64,
    num_workers: int = 0,
    multi_task: bool | None = None,
) -> dict[str, DataLoader]:
    multi_task = data.get("multi_task", True) if multi_task is None else multi_task

    def _loader(
        x: np.ndarray,
        y_cls: np.ndarray,
        y_reg: np.ndarray | None,
        shuffle: bool,
    ) -> DataLoader:
        if multi_task and y_reg is not None:
            ds: Dataset = TensorDataset(
                torch.from_numpy(x),
                torch.from_numpy(y_cls),
                torch.from_numpy(y_reg),
            )
        else:
            ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y_cls))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    yr_train = data.get("yr_train")
    yr_val = data.get("yr_val")
    yr_test = data.get("yr_test")

    return {
        "train": _loader(data["x_train"], data["y_train"], yr_train, shuffle=True),
        "val": _loader(data["x_val"], data["y_val"], yr_val, shuffle=False),
        "test": _loader(data["x_test"], data["y_test"], yr_test, shuffle=False),
    }


def save_meta(meta: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def majority_baseline_accuracy(y: np.ndarray) -> float:
    counts = np.bincount(y, minlength=3)
    return float(counts.max() / counts.sum())


if __name__ == "__main__":
    out = prepare_datasets()
    print("Train:", out["x_train"].shape, "classes:", np.bincount(out["y_train"]))
    print("Val:  ", out["x_val"].shape)
    print("Test: ", out["x_test"].shape)
    print("Thresholds:", out["meta"]["label_thresholds"])
    print("Majority baseline (test):", majority_baseline_accuracy(out["y_test"]))

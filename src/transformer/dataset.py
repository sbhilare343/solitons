"""Dataset utilities for EXP-005: train-only scaling and lookback windows.

Scaling contract (the research protocol): the z-score statistics are computed on
the **training partition only** and then applied unchanged to validation inputs
and targets. Windows never cross partition boundaries in their *targets*; inputs
may span them because the rolling protocol legitimately uses revealed history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Scaler:
    """Train-fitted z-score scaler. mean/std are frozen after fit."""

    mean: float
    std: float

    @classmethod
    def fit(cls, train_values: np.ndarray) -> "Scaler":
        train_values = np.asarray(train_values, dtype=float)
        if train_values.size == 0:
            raise ValueError("Cannot fit scaler on empty array.")
        std = float(train_values.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid training std: {std}")
        return cls(mean=float(train_values.mean()), std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "fit_on": "train partition only"}


def make_windows(
    series: np.ndarray,
    lookback: int,
    target_slice: slice,
) -> tuple[np.ndarray, np.ndarray]:
    """Inputs ending at t-1 for every target index t in ``target_slice``.

    Returns (X, y) where X[i] = series[t_i - lookback : t_i] and y[i] = series[t_i].
    Only indices present in ``series`` are produced; targets outside
    ``target_slice`` are never emitted (no training on validation targets).
    """
    series = np.asarray(series, dtype=float)
    lo, hi, step = target_slice.indices(len(series))
    xs, ys = [], []
    for t in range(max(lo, lookback), hi):
        xs.append(series[t - lookback:t])
        ys.append(series[t])
    if not xs:
        raise ValueError("Empty window set — check lookback vs target_slice.")
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Deterministic DataLoader: seeded generator, fixed worker init."""
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _init_worker(worker_id: int) -> None:
        np.random.seed(seed + worker_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        worker_init_fn=_init_worker if shuffle else None,
        drop_last=False,
    )


def validation_input_matrix(
    scaled_series: np.ndarray,
    first_validation_index: int,
    lookback: int,
    n_origins: int,
) -> np.ndarray:
    """Windows for origins t = first_validation_index ... + n_origins (ends at t-1).

    ``n_origins`` is mandatory: without it the construction would silently walk
    to the end of the series and build origins inside the *test* partition
    (caught in review before the EXP-005 run; see change log). Each row uses
    only values strictly before its origin — protocol-equivalent to a reveal
    loop, since later windows legitimately include revealed actuals.
    """
    scaled_series = np.asarray(scaled_series, dtype=float)
    stop = first_validation_index + n_origins
    if stop > len(scaled_series):
        raise ValueError(
            f"Origins [{first_validation_index}, {stop}) exceed series length "
            f"{len(scaled_series)}."
        )
    rows = [
        scaled_series[t - lookback:t]
        for t in range(first_validation_index, stop)
    ]
    if not rows:
        raise ValueError("No validation origins available.")
    return np.stack(rows).astype(np.float32)

"""EXP-005 runner: two-phase training of the registered small PatchTST.

Phase 1 — early stopping on an internal split *inside the training partition*
         (last 15% of train as targets) to select best_epoch.
Phase 2 — fresh seeded retrain on ALL train windows for exactly best_epoch
         epochs, so the model uses the full 2,815 training observations like the
         classical fits, with zero contact with the registered validation window.

Then rolling one-step validation forecasts (frozen weights, forecast-before-
reveal, batched windows ending at t-1) on the identical 603 dates.

Registered in docs/experiment_log.md (EXP-005); reproduce with:

    uv run python -m src.transformer.train
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import random
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.classical.garch_mean_spec import EXPECTED_BOUNDARIES
from src.classical.sarima import load_returns
from src.data.splits import chronological_split
from src.transformer.dataset import (
    Scaler,
    make_loader,
    make_windows,
    validation_input_matrix,
)
from src.transformer.patchtst import PatchTST, build_model, count_parameters, default_config

EXPERIMENT_ID = "EXP-005"
SEED = 42
DEVICE = "cpu"          # determinism over speed (registered)
TRAIN_CONFIG: dict[str, Any] = {
    "optimizer": "Adam",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 50,
    "early_stopping_patience": 8,
    "loss": "MSE (on scaled targets)",
    "internal_val_fraction": 0.15,   # of train, used ONLY for epoch selection
}
TABLES_DIR = Path("results/tables")
FORECASTS_DIR = Path("results/forecasts")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover
        return None


def _epoch_losses(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """Mean per-sample loss over loader; trains if optimizer is given."""
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    total, n = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for xb, yb in loader:
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
    return total / max(n, 1)


def fit_model(
    X: np.ndarray,
    y: np.ndarray,
    arch_config: dict,
    train_cfg: dict,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    seed: int = SEED,
) -> tuple[nn.Module, dict[str, list[float]], int]:
    """Train one model; returns (best_model, history, epochs_run).

    With X_val given: early stopping on val MSE, best weights restored, and
    ``epochs_run`` = best epoch (for the Phase-2 retrain). Without X_val: runs
    exactly ``train_cfg['best_epoch']`` epochs (Phase 2).
    """
    seed_everything(seed)
    model = build_model(arch_config).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    criterion = nn.MSELoss()
    # Preserve chronological window order: the project protocol forbids shuffling
    # time-series observations, including their supervised windows.
    train_loader = make_loader(X, y, train_cfg["batch_size"], shuffle=False, seed=seed)

    history: dict[str, list[float]] = {"train": [], "val": []}
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    patience_left = train_cfg["early_stopping_patience"]
    max_epochs = train_cfg.get("best_epoch", train_cfg["max_epochs"])

    for epoch in range(1, max_epochs + 1):
        train_loss = _epoch_losses(model, train_loader, criterion, optimizer)
        history["train"].append(train_loss)

        if X_val is not None:
            val_loader = make_loader(X_val, y_val, 256, shuffle=False, seed=seed)
            val_loss = _epoch_losses(model, val_loader, criterion, None)
            history["val"].append(val_loss)
            if val_loss < best_val - 1e-7:
                best_val, best_epoch, patience_left = val_loss, epoch, (
                    train_cfg["early_stopping_patience"]
                )
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

    if X_val is not None:
        if best_state is None:  # pragma: no cover - no improvement ever recorded
            raise RuntimeError("Early stopping never recorded an improvement.")
        model.load_state_dict(best_state)
        return model, history, best_epoch

    # Phase 2 (no validation): fixed budget, model as-is at last epoch
    return model, history, max_epochs


def run(
    tables_dir: Path = TABLES_DIR,
    forecasts_dir: Path = FORECASTS_DIR,
) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:4]
    arch_config = default_config()
    train_cfg = dict(TRAIN_CONFIG)

    # ---- Data and registered boundaries -------------------------------- #
    returns = load_returns()
    train, validation, _test = chronological_split(returns, 0.70, 0.15)
    if len(train) != EXPECTED_BOUNDARIES["train_n"]:
        raise ValueError(f"Unexpected train length {len(train)}")
    if len(validation) != EXPECTED_BOUNDARIES["validation_n"]:
        raise ValueError(f"Unexpected validation length {len(validation)}")

    # ---- Scaler: TRAIN ONLY (asserted) --------------------------------- #
    scaler = Scaler.fit(train.to_numpy())
    train_scaled = scaler.transform(train.to_numpy())
    # Scale only through the validation end. Test values stay outside the model
    # input path until the registered sealed-test experiment.
    train_validation = pd.concat([train, validation])
    train_validation_scaled = scaler.transform(train_validation.to_numpy())
    train_end = len(train)                      # first validation index
    internal_end = int(len(train) * (1 - train_cfg["internal_val_fraction"]))

    # ---- Phase 1: internal early stopping inside TRAIN ------------------ #
    X_tr, y_tr = make_windows(train_scaled, arch_config["lookback"],
                              slice(arch_config["lookback"], internal_end))
    X_iv, y_iv = make_windows(train_scaled, arch_config["lookback"],
                              slice(internal_end, train_end))
    model_p1, history, best_epoch = fit_model(
        X_tr, y_tr, arch_config, train_cfg, X_iv, y_iv, seed=SEED
    )
    train_cfg = {**train_cfg, "best_epoch": best_epoch}

    # ---- Phase 2: fresh retrain on ALL train windows -------------------- #
    X_all, y_all = make_windows(train_scaled, arch_config["lookback"],
                                slice(arch_config["lookback"], train_end))
    model, phase2_history, _ = fit_model(X_all, y_all, arch_config, train_cfg, seed=SEED)

    # ---- Rolling one-step validation forecasts (frozen weights) --------- #
    # Only the 603 validation origins are built — never test origins.
    model.eval()
    X_val = validation_input_matrix(
        train_validation_scaled, train_end, arch_config["lookback"],
        n_origins=len(validation)
    )
    if len(X_val) != len(validation):
        raise ValueError(f"Origin count {len(X_val)} != validation {len(validation)}")
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_val)).squeeze(-1).numpy()
    forecast = scaler.inverse(pred_scaled)      # back to return_pct units

    actual = validation.to_numpy(dtype=float)
    if len(forecast) != len(actual):
        raise ValueError(f"Forecast/actual length mismatch: {len(forecast)} vs {len(actual)}")
    rmse = float(np.sqrt(np.mean((actual - forecast) ** 2)))
    mae = float(np.mean(np.abs(actual - forecast)))
    zero_rmse = float(np.sqrt(np.mean(actual ** 2)))

    # ---- Artifacts ------------------------------------------------------- #
    payload: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "seed": SEED,
        "device": DEVICE,
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "architecture": arch_config,
        "architecture_config_hash": PatchTST.config_hash(arch_config),
        "train_config": {k: v for k, v in train_cfg.items() if k != "best_epoch"} |
                        {"best_epoch": best_epoch},
        "parameter_count": count_parameters(model),
        "scaler": scaler.to_dict(),
        "split": {
            "train_n": int(len(train)),
            "validation_n": int(len(validation)),
            "internal_val_n": int(train_end - internal_end),
            "internal_val_target_range": [
                str(train.index[internal_end].date()),
                str(train.index[-1].date()),
            ],
            "test_partition_read": False,
        },
        "phase1": {
            "purpose": "epoch selection only (early stopping on internal split)",
            "train_windows": int(len(X_tr)),
            "internal_val_windows": int(len(X_iv)),
            "best_epoch": best_epoch,
            "history": history,
        },
        "phase2": {
            "purpose": "final weights on ALL train windows for best_epoch epochs",
            "train_windows": int(len(X_all)),
            "history": phase2_history,
        },
        "validation_metrics": {
            "rmse": rmse,
            "mae": mae,
            "zero_baseline_rmse": zero_rmse,
            "beats_zero_baseline": bool(rmse < zero_rmse),
        },
        "forecast_units": "return_pct (percentage-point log return)",
    }

    tables_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    json_path = tables_dir / f"exp005_patchtst_{run_id}.json"
    csv_path = forecasts_dir / f"exp005_patchtst_validation_{run_id}.csv"
    model_path = tables_dir / f"exp005_patchtst_weights_{run_id}.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, model_path)
    payload["model_weights_path"] = str(model_path)
    payload["model_weights_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    out = pd.DataFrame({
        "Date": validation.index,
        "actual_return_pct": actual,
        "patchtst_forecast": forecast,
    })
    out.to_csv(csv_path, index=False)

    print(f"\n=== {EXPERIMENT_ID}: PatchTST (small, univariate, one-step) ===")
    print(f"config: {arch_config} | params: {payload['parameter_count']} | seed {SEED} on {DEVICE}")
    print(f"scaler (train only): mean={scaler.mean:.6f} std={scaler.std:.6f}")
    print(f"phase 1: {len(X_tr)} train / {len(X_iv)} internal-val windows -> "
          f"best_epoch={best_epoch} (val MSE {min(history['val']):.6f})")
    print(f"phase 2: {len(X_all)} train windows, {best_epoch} epochs -> "
          f"final train MSE {phase2_history['train'][-1]:.6f}")
    print(f"validation: rmse={rmse:.4f} mae={mae:.4f} | zero-baseline rmse={zero_rmse:.4f} "
          f"| beats zero: {payload['validation_metrics']['beats_zero_baseline']}")
    print(f"\nRun ID       : {run_id}")
    print(f"JSON artifact: {json_path}")
    print(f"CSV artifact : {csv_path}")
    print(f"Weights      : {model_path}")
    return payload


def main() -> None:
    run()


if __name__ == "__main__":
    main()

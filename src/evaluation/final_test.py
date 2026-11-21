"""EXP-009: one sealed evaluation of the validation-frozen forecast system."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.classical.garch_mean_spec import SPECS, fit_spec, rolling_one_step
from src.classical.sarima import load_returns
from src.data.splits import chronological_split
from src.evaluation.combination import fit_convex_weight
from src.evaluation.dm_test import dm_test
from src.transformer.dataset import Scaler, validation_input_matrix
from src.transformer.patchtst import build_model

MANIFEST = Path("docs/freeze_manifest.json")
PATCH_RUN = "20261116T164839Z_17b6"
PATCH_JSON = Path(f"results/tables/exp005_patchtst_{PATCH_RUN}.json")
PATCH_CSV = Path(f"results/forecasts/exp005_patchtst_validation_{PATCH_RUN}.csv")
OUT = Path("results/tables")
FORECASTS = Path("results/forecasts")


def run() -> dict:
    if not MANIFEST.exists():
        raise FileNotFoundError("EXP-009 is sealed: create and commit docs/freeze_manifest.json first.")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("status") != "frozen_before_test" or not manifest.get("test_gate"):
        raise ValueError("Freeze manifest is incomplete; refusing to open test partition.")

    # This is the first EXP-009 access to test outcomes. Every setting and
    # validation weight is read from the already-frozen manifest above.
    returns = load_returns()
    train, validation, test = chronological_split(returns, 0.70, 0.15)
    if len(test) != manifest["partition"]["test_n"]:
        raise ValueError("Test length differs from frozen manifest.")
    actual = test.to_numpy(float)
    predictions: dict[str, np.ndarray] = {"zero": np.zeros(len(test))}
    predictions["training_mean"] = np.full(len(test), float(train.mean()))

    # Classical model parameters are estimated on train only, then frozen while
    # one-step states update through the already observed validation and test.
    for name in SPECS:
        fitted, warnings_seen = fit_spec(train, name)
        if warnings_seen:
            print(f"EXP-009 {name} fit warnings: {warnings_seen}")
        full_eval = pd.concat([validation, test])
        forecast = rolling_one_step(fitted, train, full_eval)
        predictions[name] = forecast.loc[test.index, "mean_forecast"].to_numpy(float)

    # Reload the exact frozen Phase-2 weights and scaler; no retraining occurs.
    patch_meta = json.loads(PATCH_JSON.read_text())
    weights_path = Path(patch_meta["model_weights_path"])
    weights_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if weights_hash != manifest["patchtst"]["weights_sha256"]:
        raise ValueError("PatchTST checkpoint hash differs from freeze manifest.")
    config = patch_meta["architecture"]
    model = build_model(config)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    scaler = Scaler(**{k: patch_meta["scaler"][k] for k in ("mean", "std")})
    scaled = scaler.transform(returns.to_numpy(float))
    test_start = len(train) + len(validation)
    X = validation_input_matrix(scaled, test_start, config["lookback"], len(test))
    with torch.no_grad():
        predictions["patchtst"] = scaler.inverse(model(torch.from_numpy(X)).squeeze(-1).numpy())

    # Fixed validation-fitted convex combination, never re-estimated on test.
    combo_cfg = manifest["combination"]
    w = float(combo_cfg["weight_a"])
    predictions["frozen_combination"] = (
        w * predictions[combo_cfg["a"]] + (1-w) * predictions[combo_cfg["b"]]
    )
    scores = {}
    for name, forecast in predictions.items():
        e = actual - forecast
        scores[name] = {"rmse": float(np.mean(e**2)**0.5),
                        "mae": float(np.mean(np.abs(e))), "mean_error": float(np.mean(e))}
    primary = manifest["primary_forecast"]
    primary_dm = {name: dm_test(actual, predictions[primary], pred, loss="squared")
                  for name, pred in predictions.items() if name != primary}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:4]
    FORECASTS.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    forecast_path = FORECASTS / f"exp009_test_forecasts_{run_id}.csv"
    json_path = OUT / f"exp009_test_results_{run_id}.json"
    frame = pd.DataFrame({"Date": test.index, "actual_return_pct": actual, **predictions})
    frame.to_csv(forecast_path, index=False)
    payload = {"experiment_id": "EXP-009", "run_id": run_id,
               "freeze_manifest": str(MANIFEST), "n_test": len(test),
               "start": str(test.index[0].date()), "end": str(test.index[-1].date()),
               "primary_forecast": primary, "metrics": scores, "primary_dm_vs_each": primary_dm,
               "dm_hac_lag": next(iter(primary_dm.values()))["hac_lag"],
               "test_read_once": True, "no_reselection_or_refit_on_test": True,
               "units": "return_pct percentage points", "warnings": []}
    json_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"test_window": payload["start"] + " to " + payload["end"],
                      "primary_forecast": primary, "metrics": scores,
                      "results": str(json_path), "forecasts": str(forecast_path)}, indent=2))
    return payload


if __name__ == "__main__":
    run()

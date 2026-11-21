"""EXP-007: validation-only convex combinations and forecast encompassing."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.compare import assemble
from src.evaluation.encompassing import encompassing_regression

OUT = Path("results/tables")


def _forecasts(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return unique named mean forecasts, retaining one zero series."""
    return {
        "zero": frame["baseline_mean_zero"].to_numpy(float),
        "training_mean": frame["baseline_mean_training"].to_numpy(float),
        "constant": frame["mean_forecast_constant"].to_numpy(float),
        "ar1": frame["mean_forecast_ar1"].to_numpy(float),
        "ar2": frame["mean_forecast_ar2"].to_numpy(float),
        "patchtst": frame["patchtst"].to_numpy(float),
    }


def fit_convex_weight(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Minimize SSE of w*a+(1-w)*b subject to 0<=w<=1."""
    delta = a - b
    denom = float(delta @ delta)
    if denom <= np.finfo(float).eps:
        return 0.5
    return float(np.clip((delta @ (y - b)) / denom, 0.0, 1.0))


def run() -> dict:
    frame, _ = assemble()
    y = frame.actual_return_pct.to_numpy(float)
    preds = _forecasts(frame)
    pairs = []
    for a_name, b_name in combinations(preds, 2):
        a, b = preds[a_name], preds[b_name]
        w = fit_convex_weight(y, a, b)
        combined = w * a + (1 - w) * b
        pairs.append({"a": a_name, "b": b_name, "weight_a": w,
                      "weight_b": 1 - w,
                      "rmse_a": float(np.mean((y-a)**2)**0.5),
                      "rmse_b": float(np.mean((y-b)**2)**0.5),
                      "rmse_combined": float(np.mean((y-combined)**2)**0.5),
                      "mae_combined": float(np.mean(np.abs(y-combined))),
                      "forecast": combined.tolist()})
    best = min(pairs, key=lambda p: p["rmse_combined"])

    n = len(y)
    hac_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    encomp = [encompassing_regression(y, preds[a], preds[b], a, b, hac_lag)
              for a, b in combinations(preds, 2)]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:4]
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / f"exp007_combinations_{run_id}.csv"
    json_path = OUT / f"exp007_combinations_{run_id}.json"
    pd.DataFrame([{k: v for k, v in p.items() if k != "forecast"} for p in pairs]).to_csv(csv_path, index=False)
    payload = {"experiment_id": "EXP-007", "run_id": run_id,
               "n_validation": n, "hac_lag": hac_lag,
               "combination_objective": "validation SSE; weights clipped to [0,1]",
               "pairs": [{k: v for k, v in p.items() if k != "forecast"} for p in pairs],
               "best_validation_pair": {k: v for k, v in best.items() if k != "forecast"},
               "best_validation_forecast": best["forecast"],
               "encompassing_regressions": encomp,
               "test_partition_read": False,
               "note": "Validation-fitted combinations are optimistically biased; no test selection."}
    json_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"best_validation_pair": payload["best_validation_pair"],
                      "json": str(json_path), "csv": str(csv_path)}, indent=2))
    return payload


if __name__ == "__main__":
    run()

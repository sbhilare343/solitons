"""EXP-006: assemble fair validation comparisons on identical one-step origins."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CLASSICAL = Path("results/forecasts/exp003_validation_forecasts_20261116T113919Z_6cb9.csv")
PATCHTST = Path("results/forecasts/exp005_patchtst_validation_20261116T164839Z_17b6.csv")
OUT = Path("results/tables")


def assemble(classical_path: Path = CLASSICAL, patch_path: Path = PATCHTST) -> tuple[pd.DataFrame, dict]:
    """Join frozen validation forecasts and score them on the exact same rows."""
    a = pd.read_csv(classical_path, parse_dates=["Date"])
    b = pd.read_csv(patch_path, parse_dates=["Date"])
    if not a["Date"].equals(b["Date"]):
        raise ValueError("Forecast dates/order differ; fair comparison aborted.")
    if not np.allclose(a["actual_return_pct"], b["actual_return_pct"], rtol=0, atol=1e-12):
        raise ValueError("Actual return values differ between forecast artifacts.")
    frame = a.copy()
    frame["patchtst"] = b["patchtst_forecast"].to_numpy(dtype=float)
    cols = [c for c in frame if c.startswith("mean_forecast_") or c.startswith("baseline_mean_")]
    forecasts = {c.removeprefix("mean_forecast_").removeprefix("baseline_mean_"): frame[c].to_numpy(float)
                 for c in cols}
    forecasts["patchtst"] = frame["patchtst"].to_numpy(float)
    y = frame["actual_return_pct"].to_numpy(float)
    metrics = {}
    for name, forecast in forecasts.items():
        error = y - forecast
        metrics[name] = {"rmse": float(np.mean(error**2)**0.5), "mae": float(np.mean(np.abs(error)))}
    frame.attrs["forecast_columns"] = list(forecasts)
    frame.attrs["metrics"] = metrics
    return frame, metrics


def run() -> dict:
    frame, metrics = assemble()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:4]
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / f"exp006_fair_comparison_{run_id}.csv"
    json_path = OUT / f"exp006_fair_comparison_{run_id}.json"
    frame.to_csv(csv_path, index=False)
    payload = {"experiment_id": "EXP-006", "run_id": run_id,
               "window": {"n": len(frame), "start": str(frame.Date.iloc[0].date()),
                          "end": str(frame.Date.iloc[-1].date())},
               "forecast_columns": frame.attrs["forecast_columns"], "metrics": metrics,
               "sources": [str(CLASSICAL), str(PATCHTST)],
               "test_partition_read": False, "forecast_units": "return_pct percentage points"}
    json_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"CSV: {csv_path}\nJSON: {json_path}")
    return payload


if __name__ == "__main__":
    run()

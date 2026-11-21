"""Persist the missing EXP-001 and EXP-002 summaries without touching test data."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller

from src.classical.sarima import load_returns
from src.data.splits import chronological_split

OUT = Path("results/tables")
ARIMA_FORECASTS = Path("results/forecasts/validation_arima_candidates.csv")


def run() -> dict:
    returns = load_returns()
    train, validation, _test = chronological_split(returns)
    r = train.to_numpy(float)
    adf = adfuller(r, autolag="AIC")
    lb = {}
    for series_name, values in (("returns", r), ("squared_returns", r**2)):
        table = acorr_ljungbox(values, lags=[5, 10, 20], return_df=True)
        lb[series_name] = {str(k): {"stat": float(table.loc[k, "lb_stat"]),
                                    "pvalue": float(table.loc[k, "lb_pvalue"])} for k in (5,10,20)}
    arch_stat, arch_p, _, _ = het_arch(r, nlags=10)
    metrics = {}
    forecasts = pd.read_csv(ARIMA_FORECASTS)
    if len(forecasts) != len(validation):
        raise ValueError("EXP-002 artifact does not match registered validation length.")
    y = forecasts["actual_return_pct"].to_numpy(float)
    for col in forecasts.columns:
        if col in ("Date", "actual_return_pct"):
            continue
        e = y - forecasts[col].to_numpy(float)
        metrics[col] = {"rmse": float(np.mean(e**2)**0.5), "mae": float(np.mean(abs(e)))}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:4]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"exp001_exp002_metrics_{run_id}.json"
    payload = {"run_id": run_id, "EXP-001": {
        "status": "train-only metrics persisted; prior plots/console run are retrospective",
        "partition": {"n": len(train), "start": str(train.index[0].date()), "end": str(train.index[-1].date())},
        "adf": {"statistic": float(adf[0]), "pvalue": float(adf[1]), "lags": int(adf[2]),
                "nobs": int(adf[3]), "critical_values": {k: float(v) for k,v in adf[4].items()}},
        "ljung_box": lb, "arch_lm_10": {"stat": float(arch_stat), "pvalue": float(arch_p)},
        "skew": float(pd.Series(r).skew()), "excess_kurtosis": float(pd.Series(r).kurt())},
        "EXP-002": {"status": "validation metrics reconstructed from frozen forecast artifact",
                    "n": len(forecasts), "start": str(forecasts.Date.iloc[0]),
                    "end": str(forecasts.Date.iloc[-1]), "metrics": metrics,
                    "source": str(ARIMA_FORECASTS)}, "test_partition_read": False}
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"artifact": str(path), "exp001_adf_p": payload["EXP-001"]["adf"]["pvalue"],
                      "exp001_arch_p": payload["EXP-001"]["arch_lm_10"]["pvalue"],
                      "exp002_metrics": metrics}, indent=2))
    return payload


if __name__ == "__main__":
    run()

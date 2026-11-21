"""EXP-008: paired Diebold-Mariano tests with Bartlett HAC variance."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src.evaluation.compare import assemble
from src.evaluation.combination import _forecasts

OUT = Path("results/tables")


def automatic_hac_lag(n: int) -> int:
    """Predeclared bandwidth floor(4*(T/100)^(2/9)), bounded by T-1."""
    if n < 3:
        raise ValueError("At least three paired observations are required.")
    return min(n - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def dm_test(actual: np.ndarray, forecast_a: np.ndarray, forecast_b: np.ndarray,
            loss: str = "squared", horizon: int = 1, hac_lag: int | None = None) -> dict:
    """Test equal expected loss; positive d means A has larger loss than B."""
    y, a, b = (np.asarray(v, dtype=float) for v in (actual, forecast_a, forecast_b))
    if not (y.ndim == a.ndim == b.ndim == 1 and len(y) == len(a) == len(b)):
        raise ValueError("Actual and forecast arrays must be aligned one-dimensional arrays.")
    if not np.isfinite(np.column_stack([y, a, b])).all():
        raise ValueError("DM inputs must be finite.")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if loss == "squared":
        d = (y-a)**2 - (y-b)**2
    elif loss == "absolute":
        d = np.abs(y-a) - np.abs(y-b)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")
    n = len(d)
    lag = automatic_hac_lag(n) if hac_lag is None else int(hac_lag)
    if not 0 <= lag < n:
        raise ValueError("hac_lag must be in [0, n-1]")
    centered = d - d.mean()
    gamma0 = float(centered @ centered / n)
    long_run = gamma0
    for k in range(1, lag + 1):
        gamma = float(centered[k:] @ centered[:-k] / n)
        long_run += 2 * (1 - k / (lag + 1)) * gamma
    if long_run <= 0:
        raise ValueError(f"Nonpositive HAC long-run variance: {long_run}")
    raw = float(d.mean() / np.sqrt(long_run / n))
    hln = float(np.sqrt((n + 1 - 2*horizon + horizon*(horizon-1)/n) / n))
    statistic = raw * hln
    p = float(2 * student_t.sf(abs(statistic), df=n-1))
    se = float(np.sqrt(long_run / n))
    tcrit = float(student_t.ppf(0.975, df=n - 1))
    return {"n": n, "mean_loss_difference_a_minus_b": float(d.mean()),
            "hac_long_run_variance": long_run, "hac_standard_error": se,
            "raw_dm_statistic": raw, "hln_factor": hln, "dm_statistic": statistic,
            "pvalue_two_sided_t": p, "ci95_loss_difference":
                [float(d.mean() - tcrit*se), float(d.mean() + tcrit*se)],
            "loss": loss, "horizon": horizon, "hac_method": "Newey-West Bartlett",
            "hac_lag": lag, "reference_df": n-1}


def run() -> dict:
    frame, _ = assemble()
    y = frame.actual_return_pct.to_numpy(float)
    preds = _forecasts(frame)
    tests = []
    for a_name, b_name in combinations(preds, 2):
        for loss in ("squared", "absolute"):
            tests.append({"model_a": a_name, "model_b": b_name,
                          **dm_test(y, preds[a_name], preds[b_name], loss=loss)})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:4]
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / f"exp008_validation_dm_{run_id}.csv"
    json_path = OUT / f"exp008_validation_dm_{run_id}.json"
    pd.DataFrame(tests).to_csv(csv_path, index=False)
    payload = {"experiment_id": "EXP-008", "run_id": run_id, "partition": "validation",
               "n_dates": len(frame), "hac_lag": automatic_hac_lag(len(frame)),
               "tests": tests, "multiplicity_adjustment": "none; exploratory pairwise family",
               "test_partition_read": False}
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Validation DM results: {json_path}\nPair table: {csv_path}")
    return payload


if __name__ == "__main__":
    run()

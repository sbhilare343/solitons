"""EXP-004: validation variance evaluation across justified volatility proxies.

Reads the **frozen** EXP-003 forecast artifact (no refit — identical forecasts,
identical 603 validation dates), joins raw OHLC for the Parkinson range proxy,
and scores every candidate variance forecast plus two baselines with QLIKE under
three pre-declared proxies:

  P1 squared return   q_t = r_t^2
  P2 Parkinson range  q_t = ln(H/L)^2 / (4 ln 2) x 1e4   (decimal^2 -> return_pct^2)
  P3 EWMA(0.94)       q_t = 0.94*q_{t-1} + 0.06*r_t^2     (smoothed measurement)

Baselines (genuine one-step forecasts, no future information):

  B1 constant variance = training sample variance
  B2 EWMA(0.94) forecast s_t = 0.94*s_{t-1} + 0.06*r_{t-1}^2

Mean metrics (RMSE/MAE) are consolidated from the same artifact. The test
partition is never read. Registered pre-run in docs/experiment_log.md (EXP-004);
reproduce with:

    uv run python -m src.evaluation.variance_proxy_eval
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.classical.garch_mean_spec import EXPECTED_BOUNDARIES, SPECS
from src.classical.sarima import load_returns
from src.data.splits import chronological_split
from src.evaluation.metrics import (
    DEFAULT_EWMA_LAM,
    ewma_forecast,
    ewma_proxy,
    mae,
    parkinson_proxy,
    qlike_proxy,
    rmse,
    squared_return_proxy,
)

EXPERIMENT_ID = "EXP-004"
FORECASTS_DIR = Path("results/forecasts")
TABLES_DIR = Path("results/tables")
RAW_DATA_PATH = Path("data/raw/sp500.csv")

PROXY_NAMES = ("p1_squared_return", "p2_parkinson", "p3_ewma_094")
BASELINE_NAMES = ("b1_constant_variance", "b2_ewma_forecast")

PROXY_DEFINITIONS: dict[str, str] = {
    "p1_squared_return": "q_t = r_t^2 (return_pct^2 units)",
    "p2_parkinson": "q_t = ln(H_t/L_t)^2 / (4 ln 2) x 1e4 (raw OHLC, decimal^2 -> return_pct^2)",
    "p3_ewma_094": "q_t = 0.94*q_{t-1} + 0.06*r_t^2, state initialized with training variance",
}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover
        return None


def load_forecast_artifact(path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Load the EXP-003 validation forecast CSV (most recent if not given)."""
    if path is None:
        candidates = sorted(FORECASTS_DIR.glob("exp003_validation_forecasts_*.csv"))
        if not candidates:
            raise FileNotFoundError(
                "No EXP-003 forecast artifact found in results/forecasts/. "
                "Run `uv run python -m src.classical.garch_mean_spec` first."
            )
        path = candidates[-1]

    if not path.exists():
        raise FileNotFoundError(f"Forecast artifact not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "actual_return_pct", "baseline_mean_training"}
    for spec_name in SPECS:
        required.add(f"mean_forecast_{spec_name}")
        required.add(f"variance_forecast_{spec_name}")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Forecast artifact {path} is missing columns: {sorted(missing)}")

    df = df.sort_values("Date").reset_index(drop=True)
    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates in forecast artifact.")
    if not np.isfinite(df.drop(columns=["Date"]).to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values in forecast artifact.")
    return df, path


def load_raw_ohlc(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Raw OHLC restricted to ``dates``; raises listing any missing dates."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_DATA_PATH}")
    raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["Date"])
    missing = [str(d.date()) for d in dates if d not in set(raw["Date"])]
    if missing:
        raise ValueError(
            f"{len(missing)} validation date(s) absent from raw OHLC: {missing[:10]}"
            + ("..." if len(missing) > 10 else "")
        )
    joined = raw.set_index("Date").loc[dates, ["High", "Low"]]
    if joined.isna().any().any():
        raise ValueError("NaN in joined OHLC data.")
    return joined


def check_validation_dates(df: pd.DataFrame) -> None:
    """Assert the artifact matches the registered EXP-003/EXP-004 validation window."""
    checks = [
        (len(df) == EXPECTED_BOUNDARIES["validation_n"], f"n={len(df)}"),
        (str(df["Date"].iloc[0].date()) == "2021-03-12",
         f"start={df['Date'].iloc[0].date()}"),
        (str(df["Date"].iloc[-1].date()) == EXPECTED_BOUNDARIES["validation_end"],
         f"end={df['Date'].iloc[-1].date()}"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        raise ValueError(
            "Forecast artifact dates do not match the EXP-004 registration: "
            + "; ".join(failed)
        )


def build_proxies(
    df: pd.DataFrame,
    ohlc: pd.DataFrame,
    train_variance: float,
) -> dict[str, pd.Series]:
    dates = pd.DatetimeIndex(df["Date"])
    returns = pd.Series(df["actual_return_pct"].to_numpy(dtype=float), index=dates)

    proxies: dict[str, pd.Series] = {
        "p1_squared_return": squared_return_proxy(returns),
        "p2_parkinson": parkinson_proxy(ohlc["High"], ohlc["Low"]),
        "p3_ewma_094": ewma_proxy(returns, init=train_variance),
    }

    for name, series in proxies.items():
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Proxy {name} contains non-finite values.")
        if np.any(values < 0):
            raise ValueError(f"Proxy {name} contains negative values.")
    return proxies


def build_variance_baselines(
    df: pd.DataFrame,
    train_variance: float,
    last_train_r2: float,
) -> dict[str, pd.Series]:
    dates = pd.DatetimeIndex(df["Date"])
    returns = pd.Series(df["actual_return_pct"].to_numpy(dtype=float), index=dates)

    return {
        "b1_constant_variance": pd.Series(
            train_variance, index=dates, name="b1_constant_variance"
        ),
        "b2_ewma_forecast": ewma_forecast(
            returns, init=train_variance, last_history_r2=last_train_r2
        ),
    }


def evaluate(
    df: pd.DataFrame,
    proxies: dict[str, pd.Series],
    baselines: dict[str, pd.Series],
) -> dict[str, Any]:
    """Mean metrics + QLIKE matrix + rankings (candidates only for rankings)."""
    actual_mean = df["actual_return_pct"].to_numpy(dtype=float)

    mean_metrics: dict[str, dict[str, float]] = {}
    for spec_name in SPECS:
        fc = df[f"mean_forecast_{spec_name}"].to_numpy(dtype=float)
        mean_metrics[spec_name] = {"rmse": rmse(actual_mean, fc), "mae": mae(actual_mean, fc)}
    mean_metrics["b0_zero_forecast"] = {
        "rmse": rmse(actual_mean, np.zeros_like(actual_mean)),
        "mae": mae(actual_mean, np.zeros_like(actual_mean)),
    }
    train_mean = float(df["baseline_mean_training"].iloc[0])
    mean_metrics["b0_training_mean_forecast"] = {
        "rmse": rmse(actual_mean, np.full_like(actual_mean, train_mean)),
        "mae": mae(actual_mean, np.full_like(actual_mean, train_mean)),
    }

    qlike_matrix: dict[str, dict[str, float]] = {}
    for proxy_name, proxy in proxies.items():
        row: dict[str, float] = {}
        for spec_name in SPECS:
            variance = df[f"variance_forecast_{spec_name}"].to_numpy(dtype=float)
            row[spec_name] = qlike_proxy(proxy.to_numpy(dtype=float), variance)
        for baseline_name, forecast in baselines.items():
            row[baseline_name] = qlike_proxy(
                proxy.to_numpy(dtype=float), forecast.to_numpy(dtype=float)
            )
        qlike_matrix[proxy_name] = row

    # Candidate rankings per proxy (best = lowest QLIKE); baselines excluded.
    rankings: dict[str, list[str]] = {}
    for proxy_name, row in qlike_matrix.items():
        rankings[proxy_name] = sorted(SPECS, key=lambda s: row[s])

    ranking_stable = len({tuple(r) for r in rankings.values()}) == 1

    # Relative QLIKE improvement of each candidate vs the EWMA forecast baseline.
    improvement: dict[str, dict[str, float]] = {}
    for proxy_name, row in qlike_matrix.items():
        base = row["b2_ewma_forecast"]
        improvement[proxy_name] = {
            spec_name: float((base - row[spec_name]) / base) for spec_name in SPECS
        }

    return {
        "mean_metrics": mean_metrics,
        "qlike": qlike_matrix,
        "candidate_rankings_by_proxy": rankings,
        "candidate_ranking_stable_across_proxies": ranking_stable,
        "qlike_improvement_vs_b2_ewma_forecast": improvement,
    }


def run(
    tables_dir: Path = TABLES_DIR,
    forecast_csv: Path | None = None,
) -> dict[str, Any]:
    """Execute EXP-004 end to end and write uniquely-named artifacts."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:4]

    # Frozen EXP-003 forecasts (no refit) + train statistics for initialization.
    df, source_path = load_forecast_artifact(forecast_csv)
    check_validation_dates(df)

    returns = load_returns()
    train, _validation, _ = chronological_split(returns, 0.70, 0.15)
    if len(train) != EXPECTED_BOUNDARIES["train_n"]:
        raise ValueError(f"Unexpected train length {len(train)}")
    train_variance = float(train.var(ddof=1))
    last_train_r2 = float(train.iloc[-1]) ** 2

    # Raw OHLC for the Parkinson proxy; identical dates for every proxy.
    ohlc = load_raw_ohlc(pd.DatetimeIndex(df["Date"]))

    proxies = build_proxies(df, ohlc, train_variance)
    baselines = build_variance_baselines(df, train_variance, last_train_r2)
    results = evaluate(df, proxies, baselines)

    source_run_id = source_path.stem.removeprefix("exp003_validation_forecasts_")
    payload: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "source": {
            "forecast_artifact": str(source_path),
            "exp003_run_id": source_run_id,
            "raw_ohlc": str(RAW_DATA_PATH),
        },
        "partition": {
            "validation_dates": {
                "n": int(len(df)),
                "start": str(df["Date"].iloc[0].date()),
                "end": str(df["Date"].iloc[-1].date()),
            },
            "train_used_for_initialization_only": {
                "n": int(len(train)),
                "end": str(train.index[-1].date()),
                "variance": train_variance,
            },
            "test_partition_read": False,
        },
        "ewma_lambda": DEFAULT_EWMA_LAM,
        "proxies": PROXY_DEFINITIONS,
        "baselines": {
            "b1_constant_variance": "training sample variance (ddof=1), frozen",
            "b2_ewma_forecast": (
                f"s_t = {DEFAULT_EWMA_LAM}*s_{{t-1}} + "
                f"{1 - DEFAULT_EWMA_LAM}*r_{{t-1}}^2; "
                f"first forecast = {DEFAULT_EWMA_LAM}*train_var + "
                f"{1 - DEFAULT_EWMA_LAM}*r_{{train_last}}^2"
            ),
        },
        **results,
        "robustness_sensitivity_registered": "EXP-004-S1 (persistence>1 constrained refit)",
    }

    tables_dir.mkdir(parents=True, exist_ok=True)
    json_path = tables_dir / f"exp004_variance_proxy_eval_{run_id}.json"
    csv_path = tables_dir / f"exp004_qlike_by_proxy_{run_id}.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    rows = []
    for proxy_name, row in results["qlike"].items():
        for model_name, value in row.items():
            kind = "candidate" if model_name in SPECS else "baseline"
            rows.append(
                {"proxy": proxy_name, "model": model_name, "kind": kind, "qlike": value}
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    _print_summary(payload)
    print(f"\nRun ID          : {run_id}")
    print(f"JSON artifact   : {json_path}")
    print(f"CSV artifact    : {csv_path}")
    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    print(f"\n=== {payload['experiment_id']} : variance evaluation across proxies ===")
    src = payload["source"]
    print(f"source forecasts: {src['forecast_artifact']} (EXP-003 run {src['exp003_run_id']})")
    part = payload["partition"]["validation_dates"]
    print(f"validation dates: {part['n']} ({part['start']} -> {part['end']}), test unread")

    proxies = list(payload["qlike"].keys())
    models = list(payload["qlike"][proxies[0]].keys())
    width = max(len(m) for m in models)
    header = f"{'model':{width}s}" + "".join(f" {p:>18s}" for p in proxies)
    print(header)
    print("-" * len(header))
    for model in models:
        cells = "".join(f" {payload['qlike'][p][model]:18.4f}" for p in proxies)
        marker = " *" if model in SPECS else ""
        print(f"{model:{width}s}{cells}{marker}")
    print("  (* = GARCH candidate; lower QLIKE is better)")

    print("\ncandidate rankings (best -> worst):")
    for proxy_name, ranking in payload["candidate_rankings_by_proxy"].items():
        print(f"  {proxy_name}: {' > '.join(ranking)}")
    print(
        "ranking stable across proxies: "
        f"{payload['candidate_ranking_stable_across_proxies']}"
    )

    print("\nmean forecasts (RMSE / MAE):")
    for name, m in payload["mean_metrics"].items():
        print(f"  {name:30s} rmse={m['rmse']:.4f}  mae={m['mae']:.4f}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()

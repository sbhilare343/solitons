"""Tests for EXP-004 (src/evaluation/metrics.py + variance_proxy_eval.py).

Covers: proxy arithmetic and unit conversion (Parkinson decimal^2 -> return_pct^2),
EWMA proxy vs EWMA forecast semantics (measurement may use the current return;
forecast must not), no-future-leakage of the EWMA forecast baseline, QLIKE
arithmetic and domain errors, raw-OHLC date-join integrity, the registered
validation window, and the end-to-end artifact schema + determinism.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classical.garch_mean_spec import EXPECTED_BOUNDARIES, SPECS
from src.evaluation.metrics import (
    DEFAULT_EWMA_LAM,
    PCT2_SCALE,
    ewma_forecast,
    ewma_proxy,
    mae,
    parkinson_proxy,
    qlike_proxy,
    rmse,
    squared_return_proxy,
)
from src.evaluation.variance_proxy_eval import (
    BASELINE_NAMES,
    PROXY_NAMES,
    check_validation_dates,
    load_forecast_artifact,
    load_raw_ohlc,
    run,
)

FORECAST_ARTIFACT = next(
    iter(sorted(Path("results/forecasts").glob("exp003_validation_forecasts_*.csv"))), None
)


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #
def test_rmse_mae_values_and_shape_guard():
    actual = np.array([0.0, 0.0])
    forecast = np.array([3.0, 4.0])
    assert rmse(actual, forecast) == pytest.approx(np.sqrt(12.5))
    assert mae(actual, forecast) == pytest.approx(3.5)
    with pytest.raises(ValueError):
        rmse(actual, forecast[:1])
    with pytest.raises(ValueError):
        mae(actual, forecast[:1])


# --------------------------------------------------------------------------- #
# QLIKE
# --------------------------------------------------------------------------- #
def test_qlike_proxy_arithmetic_and_domain():
    assert qlike_proxy(np.array([4.0]), np.array([2.0])) == pytest.approx(2.0 + np.log(2.0))
    with pytest.raises(ValueError):
        qlike_proxy(np.array([1.0]), np.array([0.0]))       # non-positive forecast
    with pytest.raises(ValueError):
        qlike_proxy(np.array([-1.0]), np.array([1.0]))      # negative proxy
    with pytest.raises(ValueError):
        qlike_proxy(np.array([1.0, 2.0]), np.array([1.0]))  # shape mismatch
    with pytest.raises(ValueError):
        qlike_proxy(np.array([np.nan]), np.array([1.0]))    # non-finite


# --------------------------------------------------------------------------- #
# Proxies
# --------------------------------------------------------------------------- #
def test_squared_return_proxy():
    r = pd.Series([1.0, -2.0, 0.5], index=pd.date_range("2021-01-04", periods=3))
    out = squared_return_proxy(r)
    assert out.to_list() == pytest.approx([1.0, 4.0, 0.25])


def test_parkinson_arithmetic_and_unit_conversion():
    # ln(H/L) = 0.01  =>  q_decimal = 0.01^2 / (4 ln 2)
    high = pd.Series([np.exp(0.01)])
    low = pd.Series([1.0])
    expected_decimal = 0.01 ** 2 / (4.0 * np.log(2.0))

    q_dec = parkinson_proxy(high, low, scale_to_pct2=False)
    q_pct = parkinson_proxy(high, low, scale_to_pct2=True)
    assert float(q_dec.iloc[0]) == pytest.approx(expected_decimal)
    assert float(q_pct.iloc[0]) == pytest.approx(expected_decimal * PCT2_SCALE)
    assert PCT2_SCALE == 10_000.0

    with pytest.raises(ValueError):
        parkinson_proxy(pd.Series([0.0]), pd.Series([1.0]))   # non-positive price
    with pytest.raises(ValueError):
        parkinson_proxy(pd.Series([0.9]), pd.Series([1.0]))   # H < L


def test_ewma_proxy_recursion_and_current_return_use():
    r = pd.Series([1.0, 4.0], index=pd.date_range("2021-01-04", periods=2))
    init, lam = 2.0, 0.94
    q = ewma_proxy(r, init=init, lam=lam)
    expected = [
        lam * init + (1 - lam) * 1.0 ** 2,
        lam * (lam * init + (1 - lam) * 1.0 ** 2) + (1 - lam) * 4.0 ** 2,
    ]
    assert q.to_list() == pytest.approx(expected)

    # A measurement may use the current return: perturbing r_0 changes q_0.
    r2 = r.copy()
    r2.iloc[0] = 9.0
    assert float(ewma_proxy(r2, init=init, lam=lam).iloc[0]) != pytest.approx(expected[0])

    with pytest.raises(ValueError):
        ewma_proxy(r, init=0.0)
    with pytest.raises(ValueError):
        ewma_proxy(r, init=1.0, lam=1.0)


def test_ewma_forecast_first_value_and_no_current_return_use():
    r = pd.Series([1.0, 4.0, 9.0], index=pd.date_range("2021-01-04", periods=3))
    init, lam, last_hist_r2 = 2.0, 0.94, 0.25
    fc = ewma_forecast(r, init=init, last_history_r2=last_hist_r2, lam=lam)

    first = lam * init + (1 - lam) * last_hist_r2
    assert float(fc.iloc[0]) == pytest.approx(first)

    # Forecast must NOT use the current return: perturbing r_0 leaves fc_0 intact.
    r_perturbed = r.copy()
    r_perturbed.iloc[0] = 100.0
    fc_pert = ewma_forecast(
        r_perturbed, init=init, last_history_r2=last_hist_r2, lam=lam
    )
    assert float(fc_pert.iloc[0]) == pytest.approx(first)
    # ... but the *next* forecast must change once r_0 is revealed.
    assert float(fc_pert.iloc[1]) != pytest.approx(float(fc.iloc[1]))

    # Proxy and forecast differ on the same date (measurement vs forecast).
    proxy = ewma_proxy(r, init=init, lam=lam)
    assert float(proxy.iloc[0]) != pytest.approx(first)  # uses r_0, not last_hist_r2

    with pytest.raises(ValueError):
        ewma_forecast(r, init=-1.0, last_history_r2=0.25)


# --------------------------------------------------------------------------- #
# Artifact loading / joins / registered window
# --------------------------------------------------------------------------- #
def test_forecast_artifact_loads_and_matches_registered_window():
    assert FORECAST_ARTIFACT is not None, "EXP-003 forecast artifact missing from repo"
    df, path = load_forecast_artifact()
    assert path == sorted(Path("results/forecasts").glob("exp003_validation_forecasts_*.csv"))[-1]
    check_validation_dates(df)
    assert len(df) == EXPECTED_BOUNDARIES["validation_n"] == 603
    assert str(df["Date"].iloc[0].date()) == "2021-03-12"
    assert str(df["Date"].iloc[-1].date()) == EXPECTED_BOUNDARIES["validation_end"]


def test_raw_ohlc_join_covers_every_validation_date():
    df, _ = load_forecast_artifact()
    ohlc = load_raw_ohlc(pd.DatetimeIndex(df["Date"]))
    assert len(ohlc) == 603
    assert ohlc.notna().all().all()


def test_raw_ohlc_missing_date_is_reported():
    df, _ = load_forecast_artifact()
    bad_dates = pd.DatetimeIndex(df["Date"]).append(
        pd.DatetimeIndex([pd.Timestamp("2026-01-01")])  # New Year, not a trading day
    )
    with pytest.raises(ValueError, match="absent from raw OHLC"):
        load_raw_ohlc(bad_dates)


def test_wrong_dates_rejected():
    df, _ = load_forecast_artifact()
    shifted = df.copy()
    shifted.loc[0, "Date"] = pd.Timestamp("2021-03-11")  # train boundary, not validation
    with pytest.raises(ValueError, match="do not match the EXP-004 registration"):
        check_validation_dates(shifted)


# --------------------------------------------------------------------------- #
# End-to-end: schema, coverage, determinism
# --------------------------------------------------------------------------- #
def test_run_writes_valid_artifacts(tmp_path: Path):
    payload = run(tables_dir=tmp_path)

    json_files = list(tmp_path.glob("exp004_variance_proxy_eval_*.json"))
    csv_files = list(tmp_path.glob("exp004_qlike_by_proxy_*.csv"))
    assert len(json_files) == 1 and len(csv_files) == 1

    with json_files[0].open(encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["run_id"] == payload["run_id"]
    assert on_disk["experiment_id"] == "EXP-004"
    assert on_disk["partition"]["test_partition_read"] is False
    assert on_disk["source"]["exp003_run_id"]
    assert set(on_disk["qlike"]) == set(PROXY_NAMES)
    assert on_disk["robustness_sensitivity_registered"].startswith("EXP-004-S1")

    # Full matrix: every proxy x (4 candidates + 2 baselines), finite and positive
    for proxy_name, row in on_disk["qlike"].items():
        assert set(row) == set(SPECS) | set(BASELINE_NAMES), proxy_name
        for model_name, value in row.items():
            assert np.isfinite(value) and value > 0, (proxy_name, model_name)

    # Mean metrics consolidated for candidates + both mean baselines
    for name in list(SPECS) + ["b0_zero_forecast", "b0_training_mean_forecast"]:
        assert name in on_disk["mean_metrics"]
        assert set(on_disk["mean_metrics"][name]) == {"rmse", "mae"}

    # Rankings: a permutation of the four candidates, stability flag boolean
    for ranking in on_disk["candidate_rankings_by_proxy"].values():
        assert sorted(ranking) == sorted(SPECS)
    assert isinstance(on_disk["candidate_ranking_stable_across_proxies"], bool)

    # CSV: long format, 3 proxies x 6 models
    csv_df = pd.read_csv(csv_files[0])
    assert len(csv_df) == len(PROXY_NAMES) * (len(SPECS) + len(BASELINE_NAMES))
    assert set(csv_df["kind"]) == {"candidate", "baseline"}
    assert set(csv_df["proxy"]) == set(PROXY_NAMES)


def test_run_is_deterministic(tmp_path: Path):
    a = run(tables_dir=tmp_path / "a")
    b = run(tables_dir=tmp_path / "b")
    # Pure arithmetic over the same frozen inputs: results are bitwise identical
    # (pytest.approx does not support nested dicts, and exactness is the point).
    assert a["qlike"] == b["qlike"]
    assert a["mean_metrics"] == b["mean_metrics"]
    assert a["candidate_rankings_by_proxy"] == b["candidate_rankings_by_proxy"]


def test_ewma_lambda_is_premeclared_convention():
    assert DEFAULT_EWMA_LAM == 0.94

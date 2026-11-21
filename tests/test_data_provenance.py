"""EXP-000 gap fill: automated data provenance and integrity assertions.

Retrospectively registered in docs/experiment_log.md (EXP-000); these tests close
the gap recorded there ("raw-vs-processed Close equality, return_pct identity,
and exchange holiday integrity have not been asserted in an automated test").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

RAW_PATH = "data/raw/sp500.csv"
PROC_PATH = "data/processed/sp500_returns.csv"

# Registered dataset boundaries (docs/freeze_manifest.json).
RAW_N = 4023
PROC_N = 4022


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, parse_dates=["Date"])


@pytest.fixture(scope="module")
def proc() -> pd.DataFrame:
    return pd.read_csv(PROC_PATH, parse_dates=["Date"])


def test_registered_row_counts(raw, proc):
    assert len(raw) == RAW_N
    assert len(proc) == PROC_N


def test_dates_are_real_trading_dates_and_ordered(raw, proc):
    for df in (raw, proc):
        assert df["Date"].is_monotonic_increasing
        assert not df["Date"].duplicated().any()
        # weekdays only (exchange holidays are non-trading days, not fabricated)
        assert (df["Date"].dt.weekday < 5).all()
    # processed dates are exactly raw dates minus the first (differencing drops it)
    assert list(proc["Date"]) == list(raw["Date"].iloc[1:])


def test_close_matches_raw_exactly(raw, proc):
    merged = proc[["Date", "Close"]].merge(
        raw[["Date", "Close"]], on="Date", suffixes=("", "_raw")
    )
    assert len(merged) == PROC_N
    assert np.array_equal(merged["Close"].to_numpy(), merged["Close_raw"].to_numpy())


def test_log_price_is_log_of_close(raw, proc):
    np.testing.assert_allclose(
        proc["log_price"].to_numpy(), np.log(proc["Close"].to_numpy()), atol=1e-12
    )


def test_return_pct_identity(raw, proc):
    """return_pct = 100 * diff(log(Close)); log_return = diff(log(Close)) decimal."""
    log_close = np.log(proc["Close"].to_numpy())
    expected_decimal = np.diff(log_close)
    np.testing.assert_allclose(proc["log_return"].to_numpy()[1:], expected_decimal, atol=1e-12)
    np.testing.assert_allclose(
        proc["return_pct"].to_numpy()[1:], 100.0 * expected_decimal, atol=1e-12
    )
    # units: return_pct is exactly 100x log_return (no silent mixing)
    np.testing.assert_allclose(
        proc["return_pct"].to_numpy()[1:],
        100.0 * proc["log_return"].to_numpy()[1:],
        atol=1e-12,
    )


def test_ohlc_sanity(raw):
    assert (raw["High"] >= raw[["Open", "Close"]].max(axis=1)).all()
    assert (raw["Low"] <= raw[["Open", "Close"]].min(axis=1)).all()
    assert (raw["High"] >= raw["Low"]).all()
    assert (raw[["Open", "High", "Low", "Close"]] > 0).all().all()
    assert (raw["Volume"] >= 0).all()


def test_no_missing_or_nonfinite_values(proc):
    assert proc.isna().sum().sum() == 0
    numeric = proc.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(numeric).all()

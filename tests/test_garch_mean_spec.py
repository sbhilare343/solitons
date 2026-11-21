"""Tests for EXP-003 (src/classical/garch_mean_spec.py).

Covers: registered split boundaries, common effective sample across mean
specifications, mean-function formulas, equivalence of the manual one-step
recursion with arch's own one-step forecast at the train-end origin,
no-future-leakage of the rolling protocol, metric arithmetic, parameter
positivity for converged fits, and the artifact schema written by run().
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classical.garch_mean_spec import (
    EXPECTED_BOUNDARIES,
    SPECS,
    fit_spec,
    mae,
    mean_function,
    qlike,
    rmse,
    rolling_one_step,
    run,
)
from src.classical.sarima import load_returns
from src.data.splits import chronological_split

# Subsample sizes keep fitting fast while exercising the same code paths.
FIT_N = 600
EVAL_N = 20


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    return load_returns()


@pytest.fixture(scope="module")
def split(returns: pd.Series):
    return chronological_split(returns, 0.70, 0.15)


@pytest.fixture(scope="module")
def train(split) -> pd.Series:
    train, _, _ = split
    return train


@pytest.fixture(scope="module")
def fitted(train: pd.Series) -> dict[str, tuple]:
    """Fit all four specifications on a train subsample (common hold_back=2)."""
    history = train.iloc[:FIT_N]
    return {name: fit_spec(history, name) for name in SPECS}


# --------------------------------------------------------------------------- #
# Data / split integrity (registered boundaries)
# --------------------------------------------------------------------------- #
def test_split_boundaries_match_registration(split):
    train, validation, test = split
    assert len(train) == EXPECTED_BOUNDARIES["train_n"] == 2815
    assert str(train.index[-1].date()) == EXPECTED_BOUNDARIES["train_end"] == "2021-03-11"
    assert len(validation) == EXPECTED_BOUNDARIES["validation_n"] == 603
    assert str(validation.index[-1].date()) == EXPECTED_BOUNDARIES["validation_end"] == "2023-08-03"
    assert len(test) == EXPECTED_BOUNDARIES["test_n"] == 604
    # Chronological, strictly increasing, no duplicates, no shuffle.
    assert train.index.is_monotonic_increasing
    assert validation.index[0] > train.index[-1]
    assert test.index[0] > validation.index[-1]
    assert returns_dup_check(split) is True


def returns_dup_check(split) -> bool:
    train, validation, test = split
    all_idx = train.index.append(validation.index).append(test.index)
    return not all_idx.duplicated().any() and all_idx.is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Common effective sample (precondition for AIC/BIC comparison)
# --------------------------------------------------------------------------- #
def test_common_effective_sample(fitted):
    effective = {name: int(res.nobs) for name, (res, _) in fitted.items()}
    assert set(effective.values()) == {FIT_N - 2}, effective
    for name, (res, _) in fitted.items():
        # hold_back is applied on the model object, not fit()
        assert res.model.hold_back == 2, name


# --------------------------------------------------------------------------- #
# Mean function formulas
# --------------------------------------------------------------------------- #
def test_mean_function_zero(fitted):
    res, _ = fitted["zero"]
    fn = mean_function(res)
    hist = np.array([1.0, -2.0, 3.0])
    assert fn(hist) == 0.0


def test_mean_function_constant(fitted):
    res, _ = fitted["constant"]
    fn = mean_function(res)
    mu = float(res.params["mu"])
    assert fn(np.array([100.0, -100.0])) == pytest.approx(mu)


def _ar_lag_params(res) -> dict[int, float]:
    """Parse lag coefficients '<series>[i]' from mean parameters (before 'omega')."""
    idx = list(res.params.index)
    out: dict[int, float] = {}
    for name in idx[: idx.index("omega")]:
        if "[" in name and name.endswith("]"):
            out[int(name.rsplit("[", 1)[1][:-1])] = float(res.params[name])
    return out


def test_mean_function_ar1(fitted):
    res, _ = fitted["ar1"]
    fn = mean_function(res)
    const = float(res.params["Const"])
    phi1 = _ar_lag_params(res)[1]
    hist = np.array([0.5, -1.25])
    assert fn(hist) == pytest.approx(const + phi1 * hist[-1])


def test_mean_function_ar2(fitted):
    res, _ = fitted["ar2"]
    fn = mean_function(res)
    const = float(res.params["Const"])
    lags = _ar_lag_params(res)
    phi1, phi2 = lags[1], lags[2]
    hist = np.array([0.5, -1.25, 2.0])
    assert fn(hist) == pytest.approx(const + phi1 * hist[-1] + phi2 * hist[-2])


# --------------------------------------------------------------------------- #
# Manual recursion ≡ arch one-step forecast at the train-end origin
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec_name", list(SPECS))
def test_manual_recursion_matches_arch_forecast(fitted, train, spec_name):
    res, _ = fitted[spec_name]
    history = train.iloc[:FIT_N]
    evaluation = train.iloc[FIT_N:FIT_N + EVAL_N]

    manual = rolling_one_step(res, history, evaluation)
    arch_fc = res.forecast(horizon=1, reindex=False)

    manual_mean = float(manual["mean_forecast"].iloc[0])
    manual_var = float(manual["variance_forecast"].iloc[0])
    arch_mean = float(np.asarray(arch_fc.mean).ravel()[-1])
    arch_var = float(np.asarray(arch_fc.variance).ravel()[-1])

    assert manual_mean == pytest.approx(arch_mean, abs=1e-8), spec_name
    assert manual_var == pytest.approx(arch_var, abs=1e-8), spec_name


# --------------------------------------------------------------------------- #
# Forecast-before-reveal: no future leakage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec_name", list(SPECS))
def test_last_actual_cannot_affect_any_forecast(fitted, train, spec_name):
    """Perturbing the final evaluation target changes nothing: it is only
    revealed *after* its own forecast, and there are no later origins."""
    res, _ = fitted[spec_name]
    history = train.iloc[:FIT_N]
    evaluation = train.iloc[FIT_N:FIT_N + EVAL_N].copy()

    base = rolling_one_step(res, history, evaluation)

    perturbed = evaluation.copy()
    perturbed.iloc[-1] = float(perturbed.iloc[-1]) + 5.0
    alt = rolling_one_step(res, history, perturbed)

    pd.testing.assert_frame_equal(base, alt)


@pytest.mark.parametrize("spec_name", list(SPECS))
def test_first_actual_affects_only_later_origins(fitted, train, spec_name):
    """Perturbing the first evaluation target leaves its own origin's forecast
    untouched (computed before reveal) but must change subsequent origins."""
    res, _ = fitted[spec_name]
    history = train.iloc[:FIT_N]
    evaluation = train.iloc[FIT_N:FIT_N + EVAL_N].copy()

    base = rolling_one_step(res, history, evaluation)

    perturbed = evaluation.copy()
    perturbed.iloc[0] = float(perturbed.iloc[0]) + 5.0
    alt = rolling_one_step(res, history, perturbed)

    pd.testing.assert_series_equal(
        base.iloc[0], alt.iloc[0], check_names=False
    )
    # Later origins must differ (mean via AR terms and/or variance via eps_0).
    assert not base.iloc[1:].equals(alt.iloc[1:])


# --------------------------------------------------------------------------- #
# Metric arithmetic
# --------------------------------------------------------------------------- #
def test_rmse_mae_values():
    actual = np.array([0.0, 0.0])
    forecast = np.array([3.0, 4.0])
    assert rmse(actual, forecast) == pytest.approx(np.sqrt(12.5))
    assert mae(actual, forecast) == pytest.approx(3.5)


def test_qlike_value_and_domain():
    assert qlike(np.array([1.0]), np.array([4.0])) == pytest.approx(0.25 + np.log(4.0))
    with pytest.raises(ValueError):
        qlike(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        qlike(np.array([1.0]), np.array([-1.0]))


# --------------------------------------------------------------------------- #
# Parameter positivity for converged fits (arch constraints)
# --------------------------------------------------------------------------- #
def test_positivity_and_persistence_recorded(fitted):
    for name, (res, _) in fitted.items():
        if int(res.convergence_flag) != 0:
            pytest.skip(f"{name} did not converge (flag={res.convergence_flag})")
        assert float(res.params["omega"]) > 0.0, name
        assert float(res.params["alpha[1]"]) >= 0.0, name
        assert float(res.params["beta[1]"]) >= 0.0, name
        persistence = float(
            res.params["omega"] + res.params["alpha[1]"] + res.params["beta[1]"]
        )
        assert np.isfinite(persistence), name
        # Persistence < 1 is *reported*, not enforced: only assert the record is
        # finite; the experiment logs persistence_below_one as a diagnostic.


# --------------------------------------------------------------------------- #
# End-to-end artifact schema (unique run ID, no test-partition use)
# --------------------------------------------------------------------------- #
def test_run_writes_valid_artifacts(tmp_path: Path):
    tables_dir = tmp_path / "tables"
    forecasts_dir = tmp_path / "forecasts"

    payload = run(tables_dir=tables_dir, forecasts_dir=forecasts_dir)

    # JSON artifact with registered schema
    table_files = list(tables_dir.glob("exp003_garch_mean_spec_*.json"))
    assert len(table_files) == 1
    with table_files[0].open(encoding="utf-8") as fh:
        on_disk = json.load(fh)

    assert on_disk["experiment_id"] == "EXP-003"
    assert on_disk["run_id"] == payload["run_id"]
    assert on_disk["hold_back"] == 2
    assert set(on_disk["specs"]) == set(SPECS)
    assert on_disk["data"]["test"]["used_for_fitting_or_selection"] is False

    for spec_name, rec in on_disk["specs"].items():
        if "error" in rec:
            continue  # failures are recorded, not hidden
        for key in (
            "converged", "effective_sample", "loglikelihood", "aic", "bic",
            "parameters", "persistence_omega_alpha_beta",
            "ljung_box_std_resid", "ljung_box_squared_std_resid",
            "arch_lm_std_resid", "arch_lm_raw_resid", "validation_metrics",
        ):
            assert key in rec, f"{spec_name} missing {key}"
        assert rec["effective_sample"] == EXPECTED_BOUNDARIES["train_n"] - 2
        for lag in ("5", "10", "20"):
            assert lag in rec["ljung_box_std_resid"]
            assert lag in rec["ljung_box_squared_std_resid"]
        for pname, prec in rec["parameters"].items():
            assert {"coef", "std_err", "z", "pvalue"} <= set(prec), pname

    # CSV artifact: validation dates only, identical for every candidate
    fc_files = list(forecasts_dir.glob("exp003_validation_forecasts_*.csv"))
    assert len(fc_files) == 1
    fc = pd.read_csv(fc_files[0], parse_dates=["Date"])

    _, validation, _ = chronological_split(load_returns(), 0.70, 0.15)
    assert len(validation) == 603
    assert len(fc) == 603
    assert list(fc["Date"]) == list(validation.index)
    for spec_name in SPECS:
        assert f"mean_forecast_{spec_name}" in fc.columns
        assert f"variance_forecast_{spec_name}" in fc.columns
    assert (fc[[f"variance_forecast_{s}" for s in SPECS]] > 0).all().all()
    assert {"baseline_mean_zero", "baseline_mean_training",
            "baseline_variance_training"} <= set(fc.columns)


def test_run_is_deterministic_in_metrics(tmp_path: Path):
    """Two runs must produce identical metrics (deterministic optimizer)."""
    a = run(tables_dir=tmp_path / "a_t", forecasts_dir=tmp_path / "a_f")
    b = run(tables_dir=tmp_path / "b_t", forecasts_dir=tmp_path / "b_f")
    for spec_name in SPECS:
        if "error" in a["specs"][spec_name] or "error" in b["specs"][spec_name]:
            continue
        assert a["specs"][spec_name]["validation_metrics"] == pytest.approx(
            b["specs"][spec_name]["validation_metrics"], rel=1e-9, abs=1e-9
        ), spec_name

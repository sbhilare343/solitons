"""Tests for EXP-004-S1 (src/classical/garch_constrained.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classical.garch_constrained import (
    PERSISTENCE_CAP,
    fit_constrained,
    run,
    unconstrained_to_natural,
    natural_to_unconstrained,
)
from src.classical.garch_mean_spec import SPECS, fit_spec
from src.classical.sarima import load_returns

FIT_N = 600


@pytest.fixture(scope="module")
def train() -> pd.Series:
    return load_returns().iloc[:FIT_N]


def test_round_trip_theta_to_natural_to_theta(train):
    """Optimizer coordinates must map consistently through the reparameterization."""
    for spec in SPECS:
        base, _ = fit_spec(train, spec)
        index = base.params.index
        theta = natural_to_unconstrained(base.params)
        nat1 = unconstrained_to_natural(theta, index)
        theta2 = natural_to_unconstrained(nat1)
        nat2 = unconstrained_to_natural(theta2, index)
        np.testing.assert_allclose(nat1.to_numpy(), nat2.to_numpy(), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(theta, theta2, rtol=1e-6, atol=1e-8)


def test_constraint_enforced_even_when_unconstrained_persistence_exceeds_one(train):
    base, _ = fit_spec(train, "zero")
    p_unconstrained = float(
        base.params["omega"] + base.params["alpha[1]"] + base.params["beta[1]"]
    )
    theta = natural_to_unconstrained(base.params)  # clamps p>1 to the boundary
    nat = unconstrained_to_natural(theta, base.params.index)
    p_constrained = float(nat["omega"] + nat["alpha[1]"] + nat["beta[1]"])
    assert p_constrained < 1.0
    assert p_constrained <= PERSISTENCE_CAP + 1e-12
    if p_unconstrained >= PERSISTENCE_CAP:
        assert p_constrained == pytest.approx(PERSISTENCE_CAP, abs=1e-9)
    assert float(nat["omega"]) > 0 and float(nat["nu"]) > 2.01


@pytest.mark.parametrize("spec", list(SPECS))
def test_constrained_fit_valid_and_diagnostic(train, spec):
    fixed, opt, warns, base = fit_constrained(train, spec)
    p = float(fixed.params["omega"] + fixed.params["alpha[1]"] + fixed.params["beta[1]"])
    assert p < 1.0, spec
    assert np.isfinite(fixed.loglikelihood), spec
    assert int(fixed.nobs) == FIT_N - 2, spec
    # constrained LL can only be <= unconstrained LL (smaller feasible set)
    assert fixed.loglikelihood <= base.loglikelihood + 1e-6, spec


def test_run_applies_decision_rule_and_writes_artifact(tmp_path: Path):
    payload = run(tables_dir=tmp_path)

    files = list(tmp_path.glob("exp004s1_constrained_garch_*.json"))
    assert len(files) == 1
    on_disk = json.loads(files[0].read_text())
    assert on_disk["run_id"] == payload["run_id"]
    assert on_disk["test_partition_read"] is False
    assert set(on_disk["specs"]) == set(SPECS)

    for spec, rec in on_disk["specs"].items():
        assert "error" not in rec, f"{spec}: {rec.get('error')}"
        assert rec["persistence_below_one"] is True, spec
        assert rec["effective_sample"] == 2815 - 2, spec
        assert rec["converged"] is True, (spec, rec["optimizer_message"])
        assert {"rmse", "mae", "qlike"} <= set(rec["validation_metrics"])

    d = on_disk["decision_rule"]
    # The verdict must follow mechanically from the two booleans (pre-declared rule).
    assert d["verdict"].startswith("IMMATERIAL") == (
        d["rankings_unchanged"] and d["baseline_ordering_unchanged"]
    )
    assert "rule_source" in d
    assert d["mean_rmse_ranking_s1"] and d["qlike_r2_ranking_s1"]

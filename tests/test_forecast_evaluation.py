"""Protocol and arithmetic tests for EXP-006 through EXP-008."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.combination import fit_convex_weight
from src.evaluation.compare import assemble
from src.evaluation.dm_test import automatic_hac_lag, dm_test
from src.evaluation.encompassing import encompassing_regression


def test_convex_weight_finds_exact_interpolant_and_bounds():
    y = np.array([0.5, 1.5, 2.5])
    a = np.array([0.0, 2.0, 4.0])
    b = np.array([1.0, 1.0, 1.0])
    assert fit_convex_weight(y, a, b) == pytest.approx(0.5)
    assert 0 <= fit_convex_weight(y, np.ones(3), np.zeros(3)) <= 1


def test_dm_direction_and_known_arithmetic():
    y = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5])
    better = y.copy()
    worse = np.zeros_like(y)
    result = dm_test(y, better, worse, loss="squared", hac_lag=0)
    assert result["mean_loss_difference_a_minus_b"] < 0
    assert result["n"] == len(y)
    assert result["hln_factor"] == pytest.approx(np.sqrt((len(y)-1)/len(y)))


def test_dm_rejects_misaligned_or_invalid_inputs():
    with pytest.raises(ValueError, match="aligned"):
        dm_test(np.ones(4), np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="loss"):
        dm_test(np.ones(4), np.ones(4), np.zeros(4), loss="bad")
    assert automatic_hac_lag(603) == 5


def test_fair_comparison_rejects_date_mismatch(tmp_path):
    a = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=3),
                      "actual_return_pct": [0, 1, 2], "mean_forecast_zero": [0, 0, 0]})
    b = pd.DataFrame({"Date": pd.date_range("2020-01-02", periods=3),
                      "actual_return_pct": [0, 1, 2], "patchtst_forecast": [0, 0, 0]})
    ap, bp = tmp_path/"a.csv", tmp_path/"b.csv"
    a.to_csv(ap, index=False); b.to_csv(bp, index=False)
    with pytest.raises(ValueError, match="dates/order"):
        assemble(ap, bp)


def test_encompassing_marks_constant_intercept_collinearity():
    y = np.arange(8, dtype=float)
    result = encompassing_regression(y, np.zeros(8), y, "zero", "trend", hac_lag=1)
    assert result["status"] == "not_identified_rank_deficient"
    assert result["rank"] < result["design_columns"]


def test_encompassing_estimates_non_collinear_pair():
    y = np.arange(10, dtype=float) + np.random.default_rng(4).normal(size=10)
    result = encompassing_regression(y, y + np.linspace(0, 1, 10), y[::-1], hac_lag=1)
    assert result["status"] == "estimated"
    assert "joint_p" in result

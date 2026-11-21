"""HAC directional forecast-encompassing regressions (EXP-007)."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm


def encompassing_regression(actual: np.ndarray, forecast_a: np.ndarray,
                            forecast_b: np.ndarray, name_a: str = "a",
                            name_b: str = "b", hac_lag: int = 5) -> dict:
    """Regress y on an intercept and both forecasts, reporting rank failures.

    The incremental-signal null for B is coefficient_b=0. Constant forecast
    columns may be collinear with the intercept; such parameters are unidentified
    and are returned as an explicit status, not an unstable coefficient.
    """
    y, a, b = (np.asarray(x, dtype=float) for x in (actual, forecast_a, forecast_b))
    if y.ndim != 1 or a.ndim != 1 or b.ndim != 1 or not (len(y) == len(a) == len(b)):
        raise ValueError("Encompassing inputs must be aligned one-dimensional arrays.")
    if len(y) < 4 or not np.isfinite(np.column_stack([y, a, b])).all():
        raise ValueError("Need at least four finite aligned observations.")
    if not 0 <= hac_lag < len(y):
        raise ValueError("hac_lag must be in [0, n-1].")
    x = sm.add_constant(np.column_stack([a, b]), has_constant="add")
    rank = int(np.linalg.matrix_rank(x))
    base = {"a": name_a, "b": name_b, "hac_lag": int(hac_lag),
            "rank": rank, "design_columns": int(x.shape[1])}
    if rank < x.shape[1]:
        return base | {"status": "not_identified_rank_deficient"}
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag, "use_correction": True})
    wald = fit.wald_test("x1 = 0, x2 = 0", scalar=True)
    return base | {"status": "estimated", "intercept": float(fit.params[0]),
                    "coef_a": float(fit.params[1]), "se_a_hac": float(fit.bse[1]),
                    "p_a": float(fit.pvalues[1]), "coef_b": float(fit.params[2]),
                    "se_b_hac": float(fit.bse[2]), "p_b": float(fit.pvalues[2]),
                    "joint_wald_stat": float(wald.statistic), "joint_p": float(wald.pvalue)}

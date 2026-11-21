"""Evaluation metrics and volatility proxies (EXP-004).

Point-forecast metrics (RMSE, MAE) and the proxy-based QLIKE loss used to score
one-step-ahead *variance* forecasts, plus the three pre-declared volatility
proxies of EXP-004:

- ``squared_return_proxy``  P1: q_t = r_t^2
- ``parkinson_proxy``       P2: q_t = ln(H_t/L_t)^2 / (4 ln 2), converted from
  decimal^2 to return_pct^2 by x 1e4 (units discipline, the research protocol)
- ``ewma_proxy``            P3: q_t = lam*q_{t-1} + (1-lam)*r_t^2 (RiskMetrics
  lambda = 0.94 fixed by convention, never tuned on validation). This is a
  *measurement* of the latent variance path: it uses the current r_t and is
  therefore only available ex post. It is a proxy, not a forecast.

Forecast baselines live here too (``ewma_forecast``): unlike the proxy, the
forecast uses only information available at the origin.

Note: ``qlike_proxy`` computes mean(q_t/sigma^2_t + ln sigma^2_t), which equals
Patton (2011)'s QLIKE up to terms free of the forecast, so cross-model rankings
are identical. ``src/classical/garch_mean_spec.py`` carries its own r^2-based
`qlike` for EXP-003; the duplication is deliberate (decision D-011) so the
committed EXP-003 code and its tests stay untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# decimal^2 -> percentage-point^2 (return_pct = 100 * log return)
PCT2_SCALE = 100.0 ** 2

# RiskMetrics convention; pre-declared for EXP-004, never fitted.
DEFAULT_EWMA_LAM = 0.94


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Root mean squared error in the units of ``actual``."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {forecast.shape}")
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error in the units of ``actual``."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {forecast.shape}")
    return float(np.mean(np.abs(actual - forecast)))


def qlike_proxy(proxy: np.ndarray, variance: np.ndarray) -> float:
    """QLIKE of a variance forecast given a realized-variance proxy.

    L = mean( q_t / sigma^2_t + ln sigma^2_t )

    Requires strictly positive forecasts and non-negative, finite proxies.
    """
    proxy = np.asarray(proxy, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if proxy.shape != variance.shape:
        raise ValueError(f"Shape mismatch: {proxy.shape} vs {variance.shape}")
    if not np.all(np.isfinite(proxy)) or not np.all(np.isfinite(variance)):
        raise ValueError("QLIKE inputs must be finite.")
    if np.any(variance <= 0):
        raise ValueError("QLIKE requires strictly positive variance forecasts.")
    if np.any(proxy < 0):
        raise ValueError("QLIKE variance proxies must be non-negative.")
    return float(np.mean(proxy / variance + np.log(variance)))


# --------------------------------------------------------------------------- #
# Proxies (measurements of the latent variance; may use the current return)
# --------------------------------------------------------------------------- #
def squared_return_proxy(returns_pct: pd.Series) -> pd.Series:
    """P1: q_t = r_t^2 in return_pct^2 units (noisy, unbiased up to scale)."""
    if not np.isfinite(returns_pct.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite returns in proxy input.")
    return (returns_pct.astype(float) ** 2).rename("proxy_squared_return")


def parkinson_proxy(
    high: pd.Series,
    low: pd.Series,
    scale_to_pct2: bool = True,
) -> pd.Series:
    """P2: Parkinson (1980) high-low range variance.

    q_t = ln(H_t / L_t)^2 / (4 ln 2)  estimates Var(Delta log price) in decimal^2
    under a Brownian-motion efficiency argument; set ``scale_to_pct2`` to convert
    to return_pct^2 (multiply by 1e4). Requires strictly positive prices and
    H >= L.
    """
    high = high.astype(float)
    low = low.astype(float)
    if not (np.isfinite(high).all() and np.isfinite(low).all()):
        raise ValueError("Non-finite prices in Parkinson input.")
    if (high <= 0).any() or (low <= 0).any():
        raise ValueError("Parkinson proxy requires strictly positive H and L.")
    if (high < low).any():
        raise ValueError("High < low detected in Parkinson input.")
    q = np.log(high / low) ** 2 / (4.0 * np.log(2.0))
    if scale_to_pct2:
        q = q * PCT2_SCALE
    return q.rename("proxy_parkinson")


def ewma_proxy(
    evaluation: pd.Series,
    init: float,
    lam: float = DEFAULT_EWMA_LAM,
) -> pd.Series:
    """P3: smoothed measurement q_t = lam*q_{t-1} + (1-lam)*r_t^2.

    ``init`` (required) is the state before the first evaluated observation —
    EXP-004 passes the *training* sample variance so the proxy never uses
    information from outside train+validation. Uses the current r_t: a
    measurement available only after t is observed, hence ex-post scoring.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1).")
    if float(init) <= 0:
        raise ValueError("init must be strictly positive.")
    r2 = evaluation.astype(float) ** 2
    state = float(init)
    out: list[float] = []
    for value in r2:
        state = lam * state + (1.0 - lam) * float(value)
        out.append(state)
    return pd.Series(out, index=evaluation.index, name="proxy_ewma")


# --------------------------------------------------------------------------- #
# Forecasts (must not use the current return)
# --------------------------------------------------------------------------- #
def ewma_forecast(
    evaluation: pd.Series,
    init: float,
    last_history_r2: float,
    lam: float = DEFAULT_EWMA_LAM,
) -> pd.Series:
    """One-step EWMA variance forecast: s_t = lam*s_{t-1} + (1-lam)*r_{t-1}^2.

    The first forecast is ``lam*init + (1-lam)*last_history_r2`` (all information
    available at the first origin); subsequent forecasts use only previously
    *revealed* squared returns. ``init`` (required) is the training variance.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1).")
    if float(init) <= 0 or float(last_history_r2) < 0:
        raise ValueError("init must be > 0 and last_history_r2 >= 0.")

    state = lam * float(init) + (1.0 - lam) * float(last_history_r2)
    out: list[float] = [state]
    values = evaluation.to_numpy(dtype=float)
    for value in values[:-1]:
        state = lam * state + (1.0 - lam) * float(value) ** 2
        out.append(state)
    return pd.Series(out, index=evaluation.index, name="forecast_ewma")

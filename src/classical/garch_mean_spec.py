"""EXP-003: four-way joint conditional-mean specification comparison.

Fit GARCH(1,1) with Student-t innovations on **training data only**, holding the
variance model and effective sample (``hold_back=2``) fixed while varying only the
conditional-mean specification:

    zero     : mu_t = 0
    constant : mu_t = mu
    ar1      : mu_t = mu + phi_1 * r_{t-1}
    ar2      : mu_t = mu + phi_1 * r_{t-1} + phi_2 * r_{t-2}

Diagnostics recorded per fit: convergence flag/message, effective sample,
log-likelihood, AIC/BIC, coefficients, standard errors, z, p-values, degrees of
freedom nu, persistence (omega + alpha + beta), Ljung-Box on standardized and
squared standardized residuals (lags 5/10/20), and Engle ARCH LM (nlags=10) before
and after GARCH.

Validation forecasts use a rolling one-step-ahead protocol with frozen parameters:
each forecast is computed from revealed history only, then the target is revealed and
appended to history (see ``rolling_one_step``). Mean forecasts are scored with
RMSE/MAE; variance forecasts with QLIKE against proxy r_t^2. Baselines: zero
forecast, training-mean forecast, constant variance = training sample variance.

Registered in ``docs/experiment_log.md`` (EXP-003) before execution. Run from the
repository root:

    uv run python -m src.classical.garch_mean_spec
"""

from __future__ import annotations

import json
import subprocess
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.classical.sarima import load_returns
from src.data.splits import chronological_split

# --- Experiment configuration (registered in docs/experiment_log.md, EXP-003) ---

EXPERIMENT_ID = "EXP-003"
HOLD_BACK = 2
LB_LAGS: tuple[int, ...] = (5, 10, 20)
ARCH_LM_NLAGS = 10
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

# Registered partition boundaries (the registered data protocol, re-verified 2026-11-16).
EXPECTED_BOUNDARIES = {
    "train_n": 2815,
    "train_end": "2021-03-11",
    "validation_n": 603,
    "validation_end": "2023-08-03",
    "test_n": 604,
}

TABLES_DIR = Path("results/tables")
FORECASTS_DIR = Path("results/forecasts")

# Conditional-mean candidates. Every candidate shares vol="GARCH", p=1, q=1,
# dist="t", hold_back=HOLD_BACK so that likelihoods use the same effective sample.
SPECS: dict[str, dict[str, Any]] = {
    "zero": {"mean": "Zero"},
    "constant": {"mean": "Constant"},
    "ar1": {"mean": "AR", "lags": 1},
    "ar2": {"mean": "AR", "lags": 2},
}


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_spec(
    train: pd.Series,
    spec_name: str,
    hold_back: int = HOLD_BACK,
) -> tuple[ARCHModelResult, list[str]]:
    """Fit one mean specification on ``train`` with a common ``hold_back``.

    Returns the fitted result and any Python warnings emitted during optimization
    (recorded, never suppressed silently).
    """
    if spec_name not in SPECS:
        raise KeyError(f"Unknown specification {spec_name!r}; expected one of {sorted(SPECS)}")

    cfg = SPECS[spec_name]
    kwargs: dict[str, Any] = dict(
        mean=cfg["mean"],
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
        hold_back=hold_back,
    )
    if "lags" in cfg:
        kwargs["lags"] = cfg["lags"]

    model = arch_model(train, **kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(disp="off")
    return result, [str(w.message) for w in caught]


def mean_function(result: ARCHModelResult) -> Callable[[np.ndarray], float]:
    """Build ``hist -> mu`` from a fitted result.

    ``hist`` is the revealed return history (last element = most recent).
    Mean parameters are those before 'omega' in the parameter index; the constant
    is named 'mu' (constant mean) or 'Const' (AR mean), lag coefficients are
    named '<series>[i]'.
    """
    params = result.params
    omega_pos = list(params.index).index("omega")
    mean_params = params.iloc[:omega_pos]

    if mean_params.empty:
        return lambda hist: 0.0

    if list(mean_params.index) == ["mu"]:
        mu = float(mean_params["mu"])
        return lambda hist: mu

    const = float(mean_params["Const"]) if "Const" in mean_params.index else 0.0
    lag_coefs: dict[int, float] = {}
    for name in mean_params.index:
        if "[" in name and name.endswith("]"):
            lag_coefs[int(name.rsplit("[", 1)[1][:-1])] = float(mean_params[name])
    if not lag_coefs:
        raise ValueError(f"Could not parse AR coefficients from {list(mean_params.index)}")

    def ar_mean(hist: np.ndarray) -> float:
        total = const
        for lag, phi in lag_coefs.items():
            total += phi * float(hist[-lag])
        return total

    return ar_mean


def rolling_one_step(
    result: ARCHModelResult,
    history: pd.Series,
    evaluation: pd.Series,
) -> pd.DataFrame:
    """Rolling one-step-ahead forecasts with frozen parameters (EXP-003 policy).

    At each evaluation date the mean forecast ``mu_t`` and variance forecast
    ``sigma_t^2`` are computed from ``history`` (and previously revealed
    evaluation values) *before* the target at that date is used. After recording
    the forecast, the target is revealed and appended to history, and enters the
    variance recursion as ``eps_t = r_t - mu_t``.

    The variance recursion is initialized from the last in-sample fitted
    conditional variance and residual of ``result``.
    """
    params = result.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])
    mean_fn = mean_function(result)

    hist = history.to_numpy(dtype=float).copy()
    var_prev = float(result.conditional_volatility.iloc[-1]) ** 2
    eps_prev = float(result.resid.iloc[-1])

    rows: list[dict[str, Any]] = []
    for date, y in evaluation.items():
        y_val = float(y)
        mu = mean_fn(hist)                      # forecast from history only
        var = omega + alpha * eps_prev ** 2 + beta * var_prev
        rows.append({"Date": date, "mean_forecast": mu, "variance_forecast": var})
        # target revealed after the forecast for this date
        eps_prev = y_val - mu
        var_prev = var
        hist = np.append(hist, y_val)

    return pd.DataFrame(rows).set_index("Date")


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _lb_table(series: pd.Series) -> dict[str, dict[str, float]]:
    table = acorr_ljungbox(series.dropna(), lags=list(LB_LAGS), return_df=True)
    return {
        str(lag): {
            "stat": float(table.loc[lag, "lb_stat"]),
            "pvalue": float(table.loc[lag, "lb_pvalue"]),
        }
        for lag in LB_LAGS
    }


def _arch_lm(series: pd.Series) -> dict[str, float]:
    stat, pvalue, _, _ = het_arch(series.dropna(), nlags=ARCH_LM_NLAGS)
    return {"stat": float(stat), "pvalue": float(pvalue), "nlags": float(ARCH_LM_NLAGS)}


def spec_diagnostics(
    result: ARCHModelResult,
    fit_warnings: list[str],
    hold_back: int = HOLD_BACK,
) -> dict[str, Any]:
    """Full per-specification diagnostic record for EXP-003."""
    params = result.params
    std_err = result.std_err
    pvalues = result.pvalues

    param_record = {
        name: {
            "coef": float(params[name]),
            "std_err": float(std_err[name]),
            "z": float(params[name] / std_err[name]) if float(std_err[name]) != 0 else None,
            "pvalue": float(pvalues[name]),
        }
        for name in params.index
    }

    persistence = float(
        sum(
            float(params[name])
            for name in params.index
            if name == "omega" or name.startswith("alpha") or name.startswith("beta")
        )
    )

    std_resid = pd.Series(result.std_resid, index=result.resid.index)
    raw_resid = pd.Series(result.resid)

    converged = int(result.convergence_flag) == 0
    opt_message = getattr(getattr(result, "optimization_result", None), "message", None)

    return {
        "converged": converged,
        "convergence_flag": int(result.convergence_flag),
        "optimization_message": str(opt_message) if opt_message is not None else None,
        "warnings": list(fit_warnings),
        "effective_sample": int(result.nobs),
        "hold_back": int(hold_back),
        "loglikelihood": float(result.loglikelihood),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "parameters": param_record,
        "persistence_omega_alpha_beta": persistence,
        "persistence_below_one": bool(persistence < 1.0),
        "ljung_box_std_resid": _lb_table(std_resid),
        "ljung_box_squared_std_resid": _lb_table(std_resid ** 2),
        "arch_lm_std_resid": _arch_lm(std_resid),
        "arch_lm_raw_resid": _arch_lm(raw_resid),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs(actual - forecast)))


def qlike(actual: np.ndarray, variance: np.ndarray) -> float:
    """QLIKE = mean(r^2 / sigma^2 + ln sigma^2); proxy for realized variance is r^2."""
    actual = np.asarray(actual, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if np.any(variance <= 0):
        raise ValueError("QLIKE requires strictly positive variance forecasts.")
    return float(np.mean(actual ** 2 / variance + np.log(variance)))


def validation_metrics(
    actual: np.ndarray,
    mean_fc: np.ndarray,
    variance_fc: np.ndarray,
) -> dict[str, float]:
    return {
        "rmse": rmse(actual, mean_fc),
        "mae": mae(actual, mean_fc),
        "qlike": qlike(actual, variance_fc),
    }


# --------------------------------------------------------------------------- #
# Experiment runner
# --------------------------------------------------------------------------- #
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
    except Exception:  # pragma: no cover - git may be unavailable
        return None


def _check_boundaries(train: pd.Series, validation: pd.Series, test: pd.Series) -> None:
    checks = [
        (len(train) == EXPECTED_BOUNDARIES["train_n"], f"train n={len(train)}"),
        (str(train.index[-1].date()) == EXPECTED_BOUNDARIES["train_end"],
         f"train end={train.index[-1].date()}"),
        (len(validation) == EXPECTED_BOUNDARIES["validation_n"], f"validation n={len(validation)}"),
        (str(validation.index[-1].date()) == EXPECTED_BOUNDARIES["validation_end"],
         f"validation end={validation.index[-1].date()}"),
        (len(test) == EXPECTED_BOUNDARIES["test_n"], f"test n={len(test)}"),
        (test.index[0] > validation.index[-1], "test strictly after validation"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        raise ValueError(
            "Partition boundaries do not match the EXP-003 registration: " + "; ".join(failed)
        )


def run(
    tables_dir: Path = TABLES_DIR,
    forecasts_dir: Path = FORECASTS_DIR,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> dict[str, Any]:
    """Execute EXP-003 end to end and write uniquely-named artifacts."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:4]

    returns = load_returns()
    train, validation, test = chronological_split(
        returns, train_fraction, validation_fraction
    )
    _check_boundaries(train, validation, test)

    actual = validation.to_numpy(dtype=float)
    train_mean = float(train.mean())
    train_var = float(train.var(ddof=1))

    records: dict[str, Any] = {}
    forecast_frames: dict[str, pd.DataFrame] = {}

    for spec_name in SPECS:
        try:
            result, fit_warnings = fit_spec(train, spec_name)
            diag = spec_diagnostics(result, fit_warnings)
            forecasts = rolling_one_step(result, train, validation)
            diag["validation_metrics"] = validation_metrics(
                actual,
                forecasts["mean_forecast"].to_numpy(),
                forecasts["variance_forecast"].to_numpy(),
            )
            records[spec_name] = diag
            forecast_frames[spec_name] = forecasts
        except Exception as exc:  # failures are recorded, not hidden
            records[spec_name] = {"error": f"{type(exc).__name__}: {exc}"}

    # Baselines (no fitting involved; train statistics only).
    baselines = {
        "zero_mean": {"rmse": rmse(actual, np.zeros_like(actual)),
                      "mae": mae(actual, np.zeros_like(actual))},
        "training_mean": {"value": train_mean,
                          "rmse": rmse(actual, np.full_like(actual, train_mean)),
                          "mae": mae(actual, np.full_like(actual, train_mean))},
        "train_unconditional_variance": {"value": train_var,
                                         "qlike": qlike(actual, np.full_like(actual, train_var))},
    }

    payload: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "data": {
            "path": "data/processed/sp500_returns.csv",
            "target": "return_pct",
            "target_units": "percentage-point log return",
            "n_total": int(len(returns)),
            "train": {"n": int(len(train)),
                      "start": str(train.index[0].date()),
                      "end": str(train.index[-1].date())},
            "validation": {"n": int(len(validation)),
                           "start": str(validation.index[0].date()),
                           "end": str(validation.index[-1].date())},
            "test": {"n": int(len(test)),
                     "start": str(test.index[0].date()),
                     "end": str(test.index[-1].date()),
                     "used_for_fitting_or_selection": False},
        },
        "hold_back": HOLD_BACK,
        "specs": records,
        "baselines": baselines,
    }

    tables_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / f"exp003_garch_mean_spec_{run_id}.json"
    forecast_path = forecasts_dir / f"exp003_validation_forecasts_{run_id}.csv"

    with table_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    # Assemble per-date forecast table (identical dates for all candidates).
    out = pd.DataFrame(index=validation.index)
    out["actual_return_pct"] = actual
    for spec_name, fc in forecast_frames.items():
        out[f"mean_forecast_{spec_name}"] = fc["mean_forecast"]
        out[f"variance_forecast_{spec_name}"] = fc["variance_forecast"]
    out["baseline_mean_zero"] = 0.0
    out["baseline_mean_training"] = train_mean
    out["baseline_variance_training"] = train_var
    out.index.name = "Date"
    out.to_csv(forecast_path)

    _print_summary(payload, baselines)
    print(f"\nRun ID          : {run_id}")
    print(f"Table artifact  : {table_path}")
    print(f"Forecast artifact: {forecast_path}")
    return payload


def _print_summary(payload: dict[str, Any], baselines: dict[str, Any]) -> None:
    print(f"\n=== {payload['experiment_id']} : joint mean specification comparison ===")
    print(f"train n={payload['data']['train']['n']} "
          f"({payload['data']['train']['start']} -> {payload['data']['train']['end']}), "
          f"validation n={payload['data']['validation']['n']}, hold_back={payload['hold_back']}")

    header = (f"{'spec':10s} {'conv':5s} {'n_eff':>6s} {'logL':>12s} {'AIC':>10s} {'BIC':>10s} "
              f"{'persist':>8s} {'RMSE':>8s} {'MAE':>8s} {'QLIKE':>10s}")
    print(header)
    print("-" * len(header))
    for spec_name, rec in payload["specs"].items():
        if "error" in rec:
            print(f"{spec_name:10s} ERROR: {rec['error']}")
            continue
        vm = rec["validation_metrics"]
        print(f"{spec_name:10s} {str(rec['converged']):5s} {rec['effective_sample']:6d} "
              f"{rec['loglikelihood']:12.3f} {rec['aic']:10.3f} {rec['bic']:10.3f} "
              f"{rec['persistence_omega_alpha_beta']:8.4f} "
              f"{vm['rmse']:8.4f} {vm['mae']:8.4f} {vm['qlike']:10.4f}")

    b = baselines
    print("-" * len(header))
    print(f"{'zero':10s} {'-':5s} {'-':>6s} {'-':>12s} {'-':>10s} {'-':>10s} {'-':>8s} "
          f"{b['zero_mean']['rmse']:8.4f} {b['zero_mean']['mae']:8.4f}")
    print(f"{'train_mean':10s} {'-':5s} {'-':>6s} {'-':>12s} {'-':>10s} {'-':>10s} {'-':>8s} "
          f"{b['training_mean']['rmse']:8.4f} {b['training_mean']['mae']:8.4f}")
    print(f"{'train_var':10s} (QLIKE baseline) = {b['train_unconditional_variance']['qlike']:.4f}")

    for spec_name, rec in payload["specs"].items():
        if "error" in rec:
            continue
        if rec["warnings"]:
            print(f"[warn] {spec_name}: {rec['warnings']}")
        lb = rec["ljung_box_std_resid"]
        lb_sq = rec["ljung_box_squared_std_resid"]
        print(f"[diag] {spec_name}: LB(z) p@10={lb['10']['pvalue']:.4g} | "
              f"LB(z^2) p@10={lb_sq['10']['pvalue']:.4g} | "
              f"ARCHLM(z) p={rec['arch_lm_std_resid']['pvalue']:.4g} | "
              f"persist<1={rec['persistence_below_one']}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()

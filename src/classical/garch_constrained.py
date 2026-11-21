"""EXP-004-S1: persistence-constrained GARCH(1,1)-t refit (registered sensitivity).

Re-fits the four EXP-003 mean specifications on training data with the inequality
constraint omega + alpha + beta <= 1 imposed by reparameterization:

    p        = 1 - softplus(d)          # persistence, strictly < 1
    omega    = sigmoid(o) * p
    rest     = p - omega                # = alpha + beta
    alpha    = sigmoid(s) * rest
    beta     = rest - alpha
    nu       = softplus(nu_raw) + 2.01  # Student-t degrees of freedom > 2
    mean     = identity                 # mu / Const / y[i] pass through

Optimization: scipy L-BFGS-B on -loglikelihood evaluated through
``arch_model(...).fix(params).loglikelihood`` (feasibility verified 2026-11-16),
started from the unconstrained EXP-003 fit (clamped to the boundary if its
persistence exceeded 1). Standard errors are not computed under inequality
constraints (declared limitation, see docs/experiment_log.md EXP-004-S1).

The pre-declared decision rule is applied in ``run()`` by comparing candidate
rankings (validation RMSE, validation QLIKE r^2) and GARCH-vs-baseline ordering
against the frozen EXP-003/EXP-004 artifacts. Registered in
docs/experiment_log.md (EXP-004 / EXP-004-S1); reproduce with:

    uv run python -m src.classical.garch_constrained
"""

from __future__ import annotations

import json
import subprocess
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arch.univariate.base import ARCHModelResult
from scipy.optimize import minimize
from scipy.special import expit, logit

from src.classical.garch_mean_spec import (
    EXPECTED_BOUNDARIES,
    SPECS,
    _arch_lm,
    _lb_table,
    fit_spec,
    load_returns,
    rolling_one_step,
)
from src.data.splits import chronological_split
from src.evaluation.metrics import ewma_forecast, qlike_proxy, squared_return_proxy

EXPERIMENT_ID = "EXP-004-S1"
HOLD_BACK = 2
NU_FLOOR = 2.01          # nu must exceed 2 for finite variance
PERSISTENCE_CAP = 1 - 1e-6  # boundary used when initializing from a p>1 fit
TABLES_DIR = Path("results/tables")
FORECASTS_DIR = Path("results/forecasts")


# --------------------------------------------------------------------------- #
# Reparameterization
# --------------------------------------------------------------------------- #
def _softplus(x: float) -> float:
    return float(np.logaddexp(0.0, x))


def natural_to_unconstrained(nat: pd.Series) -> np.ndarray:
    """Map arch natural parameters to the unconstrained optimization vector.

    If the natural persistence p > 1 (the EXP-003 case), omega/alpha/beta are
    scaled to the boundary p = 1 - 1e-6 for initialization only.
    """
    idx = list(nat.index)
    i_omega = idx.index("omega")
    mean = nat.iloc[:i_omega].to_numpy(dtype=float)

    omega, alpha, beta = (float(nat["omega"]), float(nat["alpha[1]"]),
                          float(nat["beta[1]"]))
    nu = max(float(nat["nu"]), NU_FLOOR + 1e-3)

    p = omega + alpha + beta
    if p <= 0:
        raise ValueError("Non-positive persistence in natural parameters.")
    if p >= PERSISTENCE_CAP:
        scale = PERSISTENCE_CAP / p
        omega, alpha, beta = omega * scale, alpha * scale, beta * scale
        p = PERSISTENCE_CAP

    o = float(logit(np.clip(omega / p, 1e-12, 1 - 1e-12)))
    s = float(logit(np.clip(alpha / (alpha + beta), 1e-12, 1 - 1e-12)))
    d = float(np.log(np.expm1(1.0 - p)))       # softplus(d) = 1 - p
    nu_raw = float(np.log(np.expm1(nu - NU_FLOOR)))
    return np.concatenate([mean, [o, s, d, nu_raw]])


def unconstrained_to_natural(theta: np.ndarray, index: pd.Index) -> pd.Series:
    """Inverse map; guarantees omega, alpha, beta > 0 and omega+alpha+beta < 1."""
    idx = list(index)
    i_omega = idx.index("omega")
    mean = theta[:i_omega]
    o, s, d, nu_raw = (float(v) for v in theta[i_omega:i_omega + 4])

    p = 1.0 - _softplus(d)                     # strictly < 1
    omega = float(expit(o)) * p
    rest = p - omega                           # alpha + beta
    alpha = float(expit(s)) * rest
    beta = rest - alpha
    nu = _softplus(nu_raw) + NU_FLOOR

    values = np.concatenate([mean, [omega, alpha, beta, nu]])
    if not np.isfinite(values).all():
        raise ValueError("Non-finite natural parameters.")
    if omega <= 0 or alpha < 0 or beta < 0 or p <= 0 or nu <= NU_FLOOR:
        raise ValueError("Parameter outside valid domain.")
    return pd.Series(values, index=index)


def _neg_loglik(theta: np.ndarray, model: Any, index: pd.Index) -> float:
    try:
        nat = unconstrained_to_natural(theta, index)
        ll = float(model.fix(nat.to_numpy()).loglikelihood)
    except Exception:
        return 1e12
    return -ll if np.isfinite(ll) else 1e12


# --------------------------------------------------------------------------- #
# Fitting and diagnostics
# --------------------------------------------------------------------------- #
def fit_constrained(
    train: pd.Series,
    spec_name: str,
    hold_back: int = HOLD_BACK,
) -> tuple[Any, Any, list[str], ARCHModelResult]:
    """Unconstrained start (EXP-003 machinery) + constrained L-BFGS-B refit.

    Returns (fixed_result, scipy_optimum, warnings, unconstrained_start).
    """
    base, base_warnings = fit_spec(train, spec_name, hold_back=hold_back)
    index = base.params.index
    theta0 = natural_to_unconstrained(base.params)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt = minimize(
            _neg_loglik,
            theta0,
            args=(base.model, index),
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12},
        )
    fit_warnings = base_warnings + [str(w.message) for w in caught]

    nat = unconstrained_to_natural(opt.x, index)
    fixed = base.model.fix(nat.to_numpy())
    return fixed, opt, fit_warnings, base


def constrained_diagnostics(fixed: Any, opt: Any, fit_warnings: list[str]) -> dict[str, Any]:
    """Diagnostic record for the constrained fit (no std_err/pvalues - declared)."""
    params = fixed.params
    persistence = float(params["omega"] + params["alpha[1]"] + params["beta[1]"])
    std_resid = pd.Series(fixed.std_resid)

    return {
        "converged": bool(opt.success),
        "optimizer": "L-BFGS-B over arch .fix(params).loglikelihood",
        "optimizer_message": str(opt.message),
        "n_iter": int(getattr(opt, "n_iter", -1)),
        "warnings": list(fit_warnings),
        "effective_sample": int(fixed.nobs),
        "hold_back": HOLD_BACK,
        "loglikelihood": float(fixed.loglikelihood),
        "aic": float(fixed.aic),
        "bic": float(fixed.bic),
        "parameters": {name: float(v) for name, v in params.items()},
        "persistence_omega_alpha_beta": persistence,
        "persistence_below_one": bool(persistence < 1.0),
        "std_err_pvalues": "not computed under inequality constraints (declared)",
        "ljung_box_std_resid": _lb_table(std_resid),
        "ljung_box_squared_std_resid": _lb_table(std_resid ** 2),
        "arch_lm_std_resid": _arch_lm(std_resid),
        "arch_lm_raw_resid": _arch_lm(pd.Series(fixed.resid)),
    }


# --------------------------------------------------------------------------- #
# Experiment runner
# --------------------------------------------------------------------------- #
def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover
        return None


def _latest(pattern_glob: str) -> Path | None:
    matches = sorted(Path(".").glob(pattern_glob))
    return matches[-1] if matches else None


def _rank(values: dict[str, float], keys: list[str]) -> list[str]:
    """Best (lowest) first."""
    return sorted(keys, key=lambda k: values[k])


def apply_decision_rule(
    s1: dict[str, Any],
    exp003: dict[str, Any],
    exp004: dict[str, Any],
) -> dict[str, Any]:
    """Compare S1 against frozen EXP-003/EXP-004 results per the registered rule."""
    specs = list(SPECS)

    rmse_003 = {s: exp003["specs"][s]["validation_metrics"]["rmse"] for s in specs}
    rmse_s1 = {s: s1["specs"][s]["validation_metrics"]["rmse"] for s in specs}
    rank_003, rank_s1 = _rank(rmse_003, specs), _rank(rmse_s1, specs)

    qlike_003 = {s: exp003["specs"][s]["validation_metrics"]["qlike"] for s in specs}
    qlike_s1 = {s: s1["specs"][s]["validation_metrics"]["qlike"] for s in specs}
    qrank_003, qrank_s1 = _rank(qlike_003, specs), _rank(qlike_s1, specs)

    # GARCH-vs-baseline ordering on r^2: EXP-004 (frozen) vs S1 recomputation.
    b1_004 = exp004["qlike"]["p1_squared_return"]["b1_constant_variance"]
    b2_004 = exp004["qlike"]["p1_squared_return"]["b2_ewma_forecast"]
    beats_b1_004 = {s: exp004["qlike"]["p1_squared_return"][s] < b1_004 for s in specs}
    beats_b2_004 = {s: exp004["qlike"]["p1_squared_return"][s] < b2_004 for s in specs}
    b1_s1, b2_s1 = s1["baselines_r2"]["b1"], s1["baselines_r2"]["b2"]
    beats_b1_s1 = {s: qlike_s1[s] < b1_s1 for s in specs}
    beats_b2_s1 = {s: qlike_s1[s] < b2_s1 for s in specs}

    ranking_unchanged = (rank_003 == rank_s1) and (qrank_003 == qrank_s1)
    ordering_unchanged = (
        beats_b1_004 == beats_b1_s1 and beats_b2_004 == beats_b2_s1
    )
    return {
        "mean_rmse_ranking_exp003": rank_003,
        "mean_rmse_ranking_s1": rank_s1,
        "qlike_r2_ranking_exp003": qrank_003,
        "qlike_r2_ranking_s1": qrank_s1,
        "rankings_unchanged": bool(ranking_unchanged),
        "beats_b1_exp004": beats_b1_004,
        "beats_b1_s1": beats_b1_s1,
        "beats_b2_exp004": beats_b2_004,
        "beats_b2_s1": beats_b2_s1,
        "baseline_ordering_unchanged": bool(ordering_unchanged),
        "verdict": (
            "IMMATERIAL: persistence>1 does not affect one-step evaluation; "
            "EXP-003/EXP-004 conclusions stand as reported"
            if (ranking_unchanged and ordering_unchanged)
            else "MATERIAL: rankings/ordering changed; EXP-003 conclusions must be "
                 "qualified in EXP-009"
        ),
        "rule_source": "pre-declared in docs/experiment_log.md EXP-004 (S1)",
    }


def run(
    tables_dir: Path = TABLES_DIR,
    exp003_json: Path | None = None,
    exp004_json: Path | None = None,
) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:4]

    exp003_path = exp003_json or _latest("results/tables/exp003_garch_mean_spec_*.json")
    exp004_path = exp004_json or _latest("results/tables/exp004_variance_proxy_eval_*.json")
    if exp003_path is None or exp004_path is None:
        raise FileNotFoundError("EXP-003/EXP-004 artifacts required for S1 comparison.")
    exp003 = json.loads(exp003_path.read_text())
    exp004 = json.loads(exp004_path.read_text())

    returns = load_returns()
    train, validation, _test = chronological_split(returns, 0.70, 0.15)
    if len(train) != EXPECTED_BOUNDARIES["train_n"]:
        raise ValueError(f"Unexpected train length {len(train)}")

    train_variance = float(train.var(ddof=1))
    last_train_r2 = float(train.iloc[-1]) ** 2
    actual = validation.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(validation.index)

    # Baselines on r^2, identical construction to EXP-004 (frozen definitions).
    b1 = pd.Series(train_variance, index=dates)
    b2 = ewma_forecast(validation, init=train_variance, last_history_r2=last_train_r2)
    p1 = squared_return_proxy(pd.Series(actual, index=dates))
    baselines_r2 = {
        "b1": qlike_proxy(p1.to_numpy(), b1.to_numpy()),
        "b2": qlike_proxy(p1.to_numpy(), b2.to_numpy()),
    }

    records: dict[str, Any] = {}
    for spec_name in SPECS:
        try:
            fixed, opt, fit_warnings, _base = fit_constrained(train, spec_name)
            diag = constrained_diagnostics(fixed, opt, fit_warnings)
            fc = rolling_one_step(fixed, train, validation)
            diag["validation_metrics"] = {
                "rmse": float(np.sqrt(np.mean((actual - fc["mean_forecast"]) ** 2))),
                "mae": float(np.mean(np.abs(actual - fc["mean_forecast"]))),
                "qlike": qlike_proxy(p1.to_numpy(),
                                     fc["variance_forecast"].to_numpy()),
            }
            records[spec_name] = diag
        except Exception as exc:  # recorded, not hidden
            records[spec_name] = {"error": f"{type(exc).__name__}: {exc}"}

    if any("error" in r for r in records.values()):
        raise RuntimeError(f"S1 fit failure(s): "
                           f"{[k for k, v in records.items() if 'error' in v]}")

    s1_payload: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "constraint": "omega + alpha + beta < 1 via p = 1 - softplus(d)",
        "source_artifacts": {"exp003": str(exp003_path), "exp004": str(exp004_path)},
        "train_n": int(len(train)),
        "validation_n": int(len(validation)),
        "test_partition_read": False,
        "specs": records,
        "baselines_r2": baselines_r2,
    }
    s1_payload["decision_rule"] = apply_decision_rule(s1_payload, exp003, exp004)

    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / f"exp004s1_constrained_garch_{run_id}.json"
    out.write_text(json.dumps(s1_payload, indent=2, ensure_ascii=False))

    _print_summary(s1_payload)
    print(f"\nRun ID        : {run_id}")
    print(f"Artifact      : {out}")
    return s1_payload


def _print_summary(p: dict[str, Any]) -> None:
    print(f"\n=== {p['experiment_id']} : persistence-constrained refit ===")
    print(f"{'spec':10s} {'conv':5s} {'n_eff':>6s} {'logL':>12s} {'AIC':>10s} "
          f"{'persist':>8s} {'RMSE':>8s} {'QLIKE':>8s}")
    for spec, rec in p["specs"].items():
        vm = rec["validation_metrics"]
        print(f"{spec:10s} {str(rec['converged']):5s} {rec['effective_sample']:6d} "
              f"{rec['loglikelihood']:12.3f} {rec['aic']:10.3f} "
              f"{rec['persistence_omega_alpha_beta']:8.5f} "
              f"{vm['rmse']:8.4f} {vm['qlike']:8.4f}")
        if rec["warnings"]:
            print(f"           [warn] {rec['warnings']}")
    print(f"baselines (r^2 QLIKE): b1 constant={p['baselines_r2']['b1']:.4f}, "
          f"b2 ewma={p['baselines_r2']['b2']:.4f}")
    d = p["decision_rule"]
    print("\ndecision rule:")
    print(f"  mean RMSE ranking  exp003 -> {d['mean_rmse_ranking_exp003']}")
    print(f"  mean RMSE ranking  s1     -> {d['mean_rmse_ranking_s1']}")
    print(f"  QLIKE r^2 ranking  exp003 -> {d['qlike_r2_ranking_exp003']}")
    print(f"  QLIKE r^2 ranking  s1     -> {d['qlike_r2_ranking_s1']}")
    print(f"  rankings unchanged: {d['rankings_unchanged']} | "
          f"baseline ordering unchanged: {d['baseline_ordering_unchanged']}")
    print(f"  VERDICT: {d['verdict']}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()

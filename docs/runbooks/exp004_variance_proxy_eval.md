# Runbook — EXP-004: variance evaluation across volatility proxies

Registered in `docs/experiment_log.md` (EXP-004) before execution. Depends on the EXP-003
forecast artifact (frozen; this experiment performs **no refit**).

## Environment

- Python 3.13 via `uv`; run **from the repository root**:
  `cd <repository-root>`
- No randomness, no free parameters: λ = 0.94 fixed by RiskMetrics convention.

## Commands

```bash
# 1. Targeted tests (proxy arithmetic + units, EWMA semantics/leakage, OHLC join,
#    registered window, end-to-end schema, determinism)
uv run pytest tests/test_variance_proxy_eval.py -q

# 2. Full suite
uv run pytest -q

# 3. Run the experiment (requires an EXP-003 forecast CSV in results/forecasts/)
uv run python -m src.evaluation.variance_proxy_eval
```

## Inputs (read-only)

- `results/forecasts/exp003_validation_forecasts_<runid>.csv` — most recent is used by default;
  must contain `actual_return_pct`, `mean_forecast_*`, `variance_forecast_*` for all four
  candidates, `baseline_mean_training`; dates must equal the registered window
  2021-03-12 → 2023-08-03 (603 rows) or the run aborts.
- `data/raw/sp500.csv` — High/Low for the Parkinson proxy; **every** validation date must be
  present or the run aborts listing the missing dates.
- `data/processed/sp500_returns.csv` — train slice only (variance + last r² for baseline
  initialization).

## Outputs (fresh unique run ID per run)

- `results/tables/exp004_variance_proxy_eval_<runid>.json` — full QLIKE matrix, mean metrics,
  rankings, stability flag, provenance (source artifact, train init, λ, git commit).
- `results/tables/exp004_qlike_by_proxy_<runid>.csv` — long format (proxy × model → QLIKE).
- `results/tables/exp004_run_log_<runid>.txt` — stdout copy (created by the operator after the
  run: `cp <run-log-path>/exp004_run.log results/tables/exp004_run_log_<runid>.txt`).

## Post-run verification

1. JSON `source.forecast_artifact` points at the intended EXP-003 run.
2. `partition.test_partition_read == false`; validation n = 603.
3. QLIKE matrix: 3 proxies × (4 candidates + 2 baselines), all finite and positive.
4. Compare `candidate_rankings_by_proxy` across proxies (stability flag).
5. Append results to `docs/experiment_log.md` (EXP-004 — Results), entries to
   `docs/experiment_log.md`, and new choices to `docs/decision_log.md`; commit artifacts.

## Failure handling

- Missing/misaligned EXP-003 artifact or missing OHLC dates ⇒ loud `FileNotFoundError` /
  `ValueError` with the offending dates; no partial artifacts are written before both inputs
  validate.
- Nothing in this experiment can "fail to converge" — it is arithmetic over frozen inputs.

## Rollback

All outputs are additive; `git rm` the three artifact files + the two source/test files to
revert. No pre-existing file is written.

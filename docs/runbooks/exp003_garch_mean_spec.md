# Runbook — EXP-003: four-way joint mean specification comparison

Registered in `docs/experiment_log.md` (EXP-003) before execution.

## Environment

- Python 3.13 via `uv` (see `.python-version`, `pyproject.toml`); arch 8.0.0, statsmodels 0.15.0.
- Run **from the repository root**: `cd <repository-root>`
- No randomness: deterministic SLSQP optimization, no seeds required.

## Commands

```bash
# 1. Targeted tests (split boundaries, common effective sample, no-future-leakage,
#    manual recursion ≡ arch one-step forecast, artifact schema)
uv run pytest tests/test_garch_mean_spec.py -q

# 2. Full suite
uv run pytest -q

# 3. Run the experiment (writes fresh uniquely-named artifacts; never overwrites)
uv run python -m src.classical.garch_mean_spec
```

## Inputs (read-only)

- `data/processed/sp500_returns.csv` — target `return_pct` (percentage-point log return).
- Split: train 2,815 (2010-01-05 → 2021-03-11), validation 603 (2021-03-12 → 2023-08-03).
  The **test** partition (604 rows) is not read by the fit or the scoring code.

## Outputs (created per run, unique run ID)

- `results/tables/exp003_garch_mean_spec_<runid>.json` — full diagnostics, parameters, metrics.
- `results/forecasts/exp003_validation_forecasts_<runid>.csv` — per-date mean/variance forecasts
  for all four candidates plus baselines (603 rows, validation dates only).

Run ID = UTC timestamp + 4 hex characters, also printed to stdout and stored inside the JSON.

## Post-run verification

1. In the JSON check, per spec: `converged`, `effective_sample == 2813` (identical across
   specs), `persistence`, all `ljung_box_*` and `arch_lm_*` values, `warnings`.
2. Check `validation_metrics` per spec vs. `baselines` (zero, training_mean) for RMSE/MAE, and
   vs. `train_unconditional_variance` for QLIKE.
3. Append results to `docs/experiment_log.md` (EXP-003 — Results), execution details to
   `docs/experiment_log.md`, and new choices to `docs/decision_log.md`.
4. Commit with the artifact paths referenced.

## Failure handling

- Non-convergence (`convergence_flag != 0`) or warnings are **recorded, not suppressed**; the
  spec remains in the comparison with its status reported.
- If a fit fails outright, the exception is caught per-spec, recorded in the JSON under
  `error`, and the remaining specs still run; the experiment status becomes PARTIAL.

## Rollback

All new files; `git rm` / `git checkout -- <path>` restores prior state. No pre-existing data or
results files are written by this runbook.

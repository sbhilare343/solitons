# Remaining experiment runbook (EXP-001/002, EXP-005…009)

Run all commands from the repository root. The recorded run used Python 3.13 via uv. For a
restricted home directory, use the writable temporary cache shown here.

```bash
export UV_CACHE_DIR=/tmp/solitons-uv-cache
export MPLCONFIGDIR=/tmp/solitons-mpl
uv run python -m src.experiments.complete_retro
uv run python -m src.transformer.train
uv run python -m src.evaluation.compare
uv run python -m src.evaluation.combination
uv run python -m src.evaluation.dm_test
uv run python paper/make_figures.py
uv run pytest -q
```

Compile the paper-style report after rebuilding the seven figures:

```bash
XDG_CACHE_HOME=/tmp/solitons-xdg-cache tectonic -X compile paper/report.tex --outdir paper
```

Each run writes a unique ID to `results/tables/` (and forecasts to `results/forecasts/`). Never
overwrite prior artifacts. The EXP-009 command is intentionally gated:

```bash
uv run python -m src.evaluation.final_test
```

It refuses to run unless `docs/freeze_manifest.json` exists with `status=frozen_before_test` and
`test_gate=true`. The manifest and all model/weight choices must be committed before this command.
The final test runner loads the exact PatchTST checkpoint hash, fits classical parameters only
on training returns, keeps them fixed, and permits each newly observed return only in later
rolling input windows. Do not edit the manifest, models, weights, or weights after test results
exist; record any later correction as a new analysis rather than a reselection.

## Artifacts

- EXP-001/002 metrics: `results/tables/exp001_exp002_metrics_*.json`.
- EXP-005 PatchTST: JSON + validation forecast CSV + `.pt` state dictionary.
- EXP-006 aligned forecast comparison: JSON + CSV.
- EXP-007 combination/encompassing: JSON + CSV.
- EXP-008 DM comparisons: JSON + CSV.
- EXP-009 final test: JSON + forecast CSV.

PyTorch may emit `enable_nested_tensor ... norm_first was True`; this is retained as a warning.
Statsmodels emits FutureWarnings for diagnostic tuple return conventions. Neither is suppressed.

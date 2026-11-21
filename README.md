# Solitons — Forecast Encompassing Study

This repository compares a classical AR-GARCH(1,1) model with Student-t innovations against a compact univariate PatchTST for one-step-ahead S&P 500 daily log-return forecasting. The central question is whether the neural forecast adds information beyond simple and classical benchmarks.

## Main findings

- Sample: 4,022 S&P 500 daily observations, 2010-01-05 to 2025-12-30.
- Chronological partitions: 2,815 training, 603 validation and 604 test observations.
- Validation RMSE: zero-return benchmark 1.1618; PatchTST 1.2952.
- Sealed-test RMSE: AR(1)-GARCH-t 0.9643; frozen zero/training-mean blend 0.9659; PatchTST 1.0997.
- The frozen blend is not statistically distinguishable from AR(1)-GARCH-t on the sealed test (Diebold-Mariano p=0.802).

The result is specific to this asset, sample, daily frequency, one-step horizon and implemented configuration. It does not establish that Transformers are generally ineffective.

## Scope

Implemented: S&P 500 daily returns; zero, constant, AR(1) and AR(2) mean specifications with GARCH(1,1)-Student-t variance; one compact univariate PatchTST; validation-fitted forecast combination; encompassing and Diebold-Mariano comparisons; multiple volatility proxies; residual diagnostics.

Not implemented: NIFTY 50 experiments, intraday forecasting, quantile forecasts, calibrated prediction intervals, VaR backtesting or trading-strategy evaluation. These remain possible extensions and are not presented as completed results. The classical model is AR-GARCH, not seasonal ARIMA-GARCH.

## Repository layout

- `src/` — data processing, classical models, PatchTST and evaluation modules.
- `data/` — archived S&P 500 raw and processed data.
- `results/` — forecast outputs, tables, diagnostics and figures from the recorded experiments.
- `docs/` — research protocol, experiment log, decisions, runbooks and Box-Jenkins diagnostic note.
- `paper/` — report source, compiled report and figures.
- `tests/` — data, model and evaluation tests.

## Environment and installation

The project uses Python 3.13 and `uv`. Install `uv` using its official installation instructions, then from the repository root run:

```bash
uv sync --all-groups
```

Run the test suite:

```bash
uv run pytest -q
```

## Reproducing selected analyses

The bundled data and recorded outputs are provided. Useful commands from the repository root include:

```bash
# Supplementary training-only Box-Jenkins identification and diagnostics
uv run python -m src.classical.box_jenkins

# Rebuild report figures from saved outputs
uv run python paper/make_figures.py
```

The main experiments and their exact run commands are documented in `docs/experiment_log.md` and `docs/runbooks/`. The final test partition was evaluated once after the forecast configuration was frozen. Do not use the sealed test to tune models, choose epochs or fit forecast-combination weights. The final-test module is intentionally a one-time confirmatory step; use the saved test artifacts for inspection.

## Box-Jenkins methodology

The additional `src/classical/box_jenkins.py` performs a training-only ARIMA(p,0,q) screen with p,q in 0..3, ranks candidates by AICc, and checks residual autocorrelation (Ljung-Box with p+q degrees-of-freedom adjustment), ARCH effects and distributional departures. See `docs/BOX_JENKINS_AND_DIAGNOSTICS.md`. This supplementary mean-model screen motivates conditional-variance modelling; it is not selected as the final forecasting model.

## Data source

The archived daily S&P 500 series was retrieved through `yfinance` from Yahoo Finance (`^GSPC`) and is included so the reported experiments can be inspected without an immediate download. The download utility is `src/data/download.py`. Review the data provider's terms before redistribution or commercial reuse.

## Research references

The report cites the foundational GARCH paper, forecast comparison methodology, volatility-proxy scoring and the PatchTST paper. Full citations are listed in `paper/report.tex`.


## Supplementary SARIMA check

A training-only SARIMA seasonal comparison is included in `src/classical/sarima_seasonal.py`. It uses a five-trading-session seasonal period, evaluates validation only, and never reads the sealed test. See `docs/SARIMA_EXPLORATORY_RESULTS.md`. The exploratory SARIMA candidate gives only a negligible validation improvement and fails residual diagnostics, so the primary model remains accurately labelled AR-GARCH(1,1)-Student-t.

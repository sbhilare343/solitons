# Final project audit and scope correction

## Correct project identity

**Forecast Encompassing: Deep Time-Series Transformers vs. Classical AR-GARCH**

The implemented project is an empirical one-step-ahead forecasting study of S&P 500 daily
log returns. The implemented classical system is AR-GARCH(1,1) with Student-t innovations;
the neural comparator is a small univariate PatchTST. Forecast combinations, encompassing
regressions, and Diebold--Mariano tests are implemented.

## Proposal-to-implementation reconciliation

| Proposal item | Actual archive | Final reporting treatment |
|---|---|---|
| S&P 500 and NIFTY 50 | S&P 500 only | Scope limited to S&P 500 |
| Daily, 10+ years | 2010-01-05 to 2025-12-30, 4,022 observations | Implemented |
| Optional 1-minute extension | Not present | Future work |
| SARIMA/GARCH | No seasonal AR or seasonal differencing; AR(0/1/2)-GARCH(1,1)-t | Call it AR-GARCH |
| PatchTST, 8 heads | 4 heads, 2 layers, 69,441 parameters | Report actual architecture |
| AdamW | Adam, lr=0.001, weight decay=0.0001 | Report actual optimizer |
| Quantile loss / interval forecasts | MSE training loss; no quantile output | Not implemented |
| VaR backtesting | Not present | Future work |
| RMSE/MAE and forecast comparisons | Present | Implemented |
| Encompassing and DM | Present | Implemented |

## Important statistical corrections

1. Canonical GARCH(1,1) persistence is alpha+beta. The finite-unconditional-variance
   condition is alpha+beta<1. Do not call omega+alpha+beta the persistence coefficient.
2. EXP-004-S1 used omega+alpha+beta<1. Retain the original estimates as a historical,
   noncanonical sensitivity; do not describe it as the standard stationarity constraint.
3. The test set was opened once under the frozen protocol. Do not reselect models, tune
   hyperparameters, or change the primary forecast based on test results.
4. The validation-fitted combination's RMSE improvement is only 0.000211 percentage points.
   It is a negligible point improvement and should not be marketed as a meaningful gain.
5. A lower test RMSE for AR(1)-GARCH than the frozen blend does not establish superiority:
   DM p=0.802 for the blend-vs-AR(1) comparison.

## Existing methodology components

The repository already contains ADF, ACF/PACF, Ljung--Box, ARCH LM, standardized residual,
squared standardized residual, and QLIKE procedures. The added
`src/classical/box_jenkins.py` makes identification, estimation, and diagnostic checking
explicit and reproducible on the training partition only.

## Final conclusion

The archive supports a narrow result: for this S&P 500 sample, daily frequency, one-step
horizon, and registered model configurations, PatchTST did not improve on simple mean
benchmarks. Classical and simple mean forecasts were statistically indistinguishable on the
sealed test; the frozen combination outperformed the evaluated PatchTST configuration.
This is not evidence that Transformers are universally ineffective or that the forecasts
are profitable trading signals.

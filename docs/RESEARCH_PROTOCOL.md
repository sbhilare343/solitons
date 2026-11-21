# Research protocol

## Scope
This repository evaluates one-step-ahead forecasts of S&P 500 daily log returns. The implemented comparison is between simple mean benchmarks, AR-GARCH(1,1) with Student-t innovations, and a compact univariate PatchTST.

## Data and target
The target `return_pct` is 100 times the first difference of log closing prices and is expressed in percentage-point log-return units. The archived sample contains 4,022 observations from 2010-01-05 through 2025-12-30.

## Chronological partitions
- Training: 2,815 observations, 2010-01-05 through 2021-03-11.
- Validation: 603 observations, 2021-03-12 through 2023-08-03.
- Test: 604 observations, 2023-08-04 through 2025-12-30.

Observations are not randomly shuffled. At each evaluation origin, the forecast is issued before the target is revealed. Once observed, the target may enter the history for the next one-step forecast. Fitted model parameters remain fixed after training. Any learned preprocessing parameters are fitted on training data only.

## Model comparison
Forecasts must use the same target dates and horizon. Report both point metrics and statistical uncertainty. Information criteria are used only to compare compatible likelihood fits on the same effective sample. Validation may be used for model selection and combination weights; the sealed test is reserved for a single confirmatory evaluation.

## GARCH terminology
For GARCH(1,1), the conventional persistence coefficient is `alpha + beta`. Under the standard assumptions, `alpha + beta < 1` is the finite-unconditional-variance condition. The variance intercept `omega` has squared-return units and is not part of the dimensionless persistence coefficient.

## Diagnostic checking
Use residual time plots, ACF and Ljung-Box checks for serial dependence, and ARCH LM / squared-standardized-residual checks for remaining conditional heteroskedasticity. Normality diagnostics such as Jarque-Bera or Q-Q plots should be interpreted as distributional checks, not forecast-accuracy metrics. Failure to reject a diagnostic null does not prove the model is correctly specified.

## Reproduction and test integrity
Consult `docs/experiment_log.md`, `docs/decision_log.md`, `docs/freeze_manifest.json`, and the runbooks for experiment details. Do not use the sealed test to tune parameters, select models, or change forecast combinations.

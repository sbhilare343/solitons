# Supplementary SARIMA experiment

## Why this experiment exists

The proposal names a SARIMA-GARCH classical track. The original implementation in
`src/classical/sarima.py` fits non-seasonal `ARIMA(p,d,q)` candidates; it does not include
seasonal orders. This supplementary experiment explicitly checks whether a five-trading-session
seasonal component adds useful mean-forecast information.

## Protocol

- Training only: 2,815 returns, 2010-01-05 to 2021-03-11.
- Validation: 603 returns, 2021-03-12 to 2023-08-03.
- Sealed test: not read.
- Candidate seasonal period: `s=5` trading sessions.
- All candidates use `d=0`, `D=0`, consistent with modelling stationary returns; no seasonal
  differencing is imposed without evidence.
- Parameters are estimated once on training; validation is forecast one step ahead with
  fixed parameters and state updates as observations arrive.
- Candidate ranking uses training AICc, but the model must also pass residual checks and show
  a meaningful validation gain.

## Findings

The training ACF at lag 5 is approximately 0.0045, which does not suggest a strong weekly
linear seasonal pattern by itself. The tested SARIMA(1,0,1)(1,0,1)[5] has validation RMSE
about 1.161696 and MAE about 0.863337. Non-seasonal ARIMA(1,0,1) has RMSE about 1.161739
and MAE about 0.863841. The absolute RMSE gain is only about 0.000043 percentage points:
negligible in practical terms.

The seasonal candidate's training residual Ljung-Box tests remain strongly significant after
accounting for four fitted AR/MA terms (model_df=4): p approximately 2.39e-4 at lag 5,
4.07e-18 at lag 10 and 6.27e-19 at lag 20. The residual ARCH LM(10) p-value is approximately
2.20e-190. Thus the fitted seasonal mean does not remove serial dependence or volatility
clustering. The model is not an adequate standalone residual model and does not replace the
AR-GARCH framework.

## Recommendation

Do not rename the main implemented model as SARIMA-GARCH. Keep the primary analysis accurately
labelled AR-GARCH(1,1)-Student-t, and present this seasonal model as an exploratory comparator.
If the course marking rubric requires a completed SARIMA-GARCH model, the next defensible step
is to combine a pre-specified SARIMA mean with GARCH conditional variance, fit only on training,
then compare validation forecasts and diagnostics before deciding whether it merits inclusion.
Do not open the sealed test to tune or choose the model.

## Reproduction

From repository root:

```bash
uv run python -m src.classical.sarima_seasonal
```

Outputs are written under `results/sarima/`.

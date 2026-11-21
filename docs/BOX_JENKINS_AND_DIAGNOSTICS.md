# Box--Jenkins methodology and residual diagnostics

## Scope

This supplementary analysis uses only the frozen training partition (2,815 daily S&P 500
percentage-point log returns, 2010-01-05 through 2021-03-11). It does not inspect validation
or test outcomes, does not alter the frozen EXP-003--EXP-009 model selection, and does not
replace the final AR-GARCH/PatchTST comparison.

## Identification

1. Work with daily log returns, not the nonstationary price level.
2. The training-only ADF test rejects a unit root (statistic -11.7049, p=1.54e-21).
3. ACF/PACF are used as identification aids, not as automatic order selectors.
4. A reproducible ARIMA(p,0,q) grid with p,q in {0,1,2,3} and a constant is fitted.
   AICc ranks the candidate mean specifications. The selected screening model is ARMA(2,2).

## Estimation

The ARMA(2,2) fit is estimated on training data only. Its AIC is 8404.261, BIC 8439.917,
and AICc 8404.291. This is a screening model, not the final model. The primary forecasting
pipeline separately compares zero, constant, AR(1), and AR(2) conditional means jointly
with GARCH(1,1)-Student-t variance.

## Diagnostic checking

For the ARMA(2,2) screening residuals:
- Ljung--Box Q(10): statistic 25.712, p=0.000252; Q(20): statistic 41.492, p=0.000470.
  These p-values use the fitted ARMA degrees-of-freedom adjustment `model_df=p+q=4`;
  residual autocorrelation remains.
- Engle ARCH LM(10): p=4.60e-182. Strong conditional heteroskedasticity remains.
- Jarque--Bera: statistic about 15,174.5; p-value numerically underflows to zero.

The screening model is therefore not adequate as a complete stochastic description. The
remaining ARCH effect supports adding a conditional-variance model; heavy tails support
Student-t innovations. This does not prove that GARCH is the unique or optimal volatility
model.

## Final AR-GARCH diagnostics already recorded

For the unconstrained EXP-003 Student-t GARCH candidates, standardized-residual Ljung--Box
p-values at lag 20 range approximately 0.080--0.153; squared-standardized-residual
Ljung--Box p-values range approximately 0.812--0.863; standardized-residual ARCH LM(10)
p-values range approximately 0.565--0.668. These checks do not reject remaining serial
correlation or ARCH effects at 5% for these selected lags. Failure to reject is not proof
that the residuals are white noise.

## GARCH persistence terminology correction

For GARCH(1,1), the conventional persistence coefficient is alpha+beta. The usual
finite-unconditional-variance condition is alpha+beta<1 (with omega>0). Omega is an
intercept with variance units and must not be added to the dimensionless persistence
coefficient. The historical EXP-004-S1 restriction omega+alpha+beta<1 is retained as a
historical sensitivity exercise, but it is not a canonical GARCH stationarity constraint.

## Reproduce

From the repository root:
`uv run python -m src.classical.box_jenkins`

The script is deliberately training-only; Ljung--Box degrees of freedom are adjusted by p+q. It writes
`results/tables/box_jenkins_training_diagnostics.json`.

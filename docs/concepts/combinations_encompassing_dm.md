# Forecast combinations, encompassing, and Diebold–Mariano tests

## Forecast combinations

For aligned forecasts A and B, a convex combination is
`f_t(w)=w f_A,t+(1-w)f_B,t`, with 0≤w≤1. Minimize validation SSE. Let `d=f_A-f_B`; then
`w*=sum(d*(y-f_B))/sum(d^2)`, clipped to [0,1]. Fitting weights on the same validation sample
used to score the combination makes its reported score optimistic. The weights must be frozen
before test evaluation.

Example: if the best convex weight is 0.5 and two forecasts are 0.00 and 0.04, the combination is
0.02. The Solitons validation fit produced weights 0.499943/0.500057 for zero and training mean;
RMSE improved by only 0.000211 percentage points versus the best constituent.

## Forecast encompassing

The regression `y_t = c + b_A f_A,t + b_B f_B,t + u_t` asks whether each forecast retains a
linear signal conditional on the other. A zero `b_B` is the directional null that B adds no
signal beyond A. Newey–West/HAC covariance allows short-range dependence in regression residuals.
If a forecast is constant, it is collinear with the intercept; separate coefficient estimates
are unidentified and must be reported as such. Statistical nonsignificance is not proof that a
forecast contains no information.

## Diebold–Mariano comparison

Define `d_t=L(e_A,t)-L(e_B,t)`. The null is `E[d_t]=0`; negative mean d favors A. The statistic
is `mean(d)/sqrt(Ωhat/T)`, where Ωhat is a Newey–West estimate of long-run variance using Bartlett
weights. For horizon h, the Harvey–Leybourne–Newbold multiplier is
`sqrt((T+1-2h+h(h-1)/T)/T)`; for h=1 this is `sqrt((T-1)/T)`. The implementation uses a
two-sided t reference with T−1 degrees of freedom. RMSE rank alone is not a significance test.

Example: T=603, d mean −0.328, HAC standard error about 0.056. The negative sign means model A
has lower loss than B; the confidence interval excludes zero in validation. Because the many
pairwise validation tests are not multiplicity adjusted, treat the p-value as exploratory.

## Practical interpretation

Always verify identical dates, forecast horizon, and information set first. Read the sign of the
loss difference, its interval, HAC lag, and p-value together. Account for validation selection,
multiple comparisons, asset and period limits, and the fact that one test period cannot establish
universal superiority.

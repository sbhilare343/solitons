# Concept: Joint conditional-mean specification for GARCH(1,1) with Student-t innovations

Used by: EXP-003. Companion to `docs/experiment_log.md` EXP-003 and
`src/classical/garch_mean_spec.py`.

## Plain language

A return series is modelled in two layers: an *average* (conditional mean) and a *spread*
(conditional variance). Volatility clustering — quiet days followed by wild days — is the
variance layer's job; any predictable drift or autocorrelation is the mean layer's job. GARCH(1,1)
says today's variance is a weighted average of yesterday's squared surprise and yesterday's
variance. The "innovations" (surprises) are assumed Student-t rather than Normal so that extreme
days are not infinitely improbable.

The open question in EXP-003: what should the *mean* layer be? Candidates, in order of
flexibility:

1. **Zero** — no drift at all: μ_t = 0. Daily equity returns are often ≈ mean-zero, so this is a
   serious baseline, not a strawman.
2. **Constant** — one fixed drift μ estimated from training data.
3. **AR(1)** — today's expected return depends on yesterday's return: μ_t = μ + φ₁ r_{t−1}.
4. **AR(2)** — plus the day before: μ_t = μ + φ₁ r_{t−1} + φ₂ r_{t−2}.

Everything else (variance dynamics, distribution, effective sample) is held identical, so any
difference in fit or forecast quality is attributable to the mean specification alone.

## Clarification added 2026-11-17: persistence units

In the recursion above, `omega` is measured in return-squared units, while `alpha` and `beta`
are dimensionless coefficients. The canonical GARCH(1,1) condition for a finite unconditional
variance is `alpha + beta < 1` under standard assumptions. The historical EXP-003 output also
records `omega + alpha + beta`, as the registered project diagnostic. That sum depends on the
scale in which returns are expressed and is not the standard persistence parameter. Therefore,
the old observation that this reported sum exceeds one is a valid arithmetic description of that
artifact, but by itself it does not establish nonstationarity. EXP-004-S1 constrained the
registered sum below one; read it as the predeclared sensitivity rule, not as the canonical
GARCH covariance-stationarity constraint. Historical log values and model fits have not been
rewritten.

## Mathematics

Model, for t = 1…T (training only):

- Conditional mean: r_t = μ_t + ε_t, with μ_t as above.
- Conditional variance: ε_t = σ_t z_t,  σ_t² = ω + α ε_{t−1}² + β σ_{t−1}²,
  where ω > 0 (variance intercept), α ≥ 0 (ARCH/reaction to the last surprise),
  β ≥ 0 (GARCH/persistence of past variance). Constraints are enforced by the optimizer;
  stationarity of σ_t² requires ω + α + β < 1 (persistence < 1) — **reported, not enforced**,
  and recorded per fit.
- Innovations: z_t standardized Student-t, density parameterized by degrees of freedom ν > 2
  (ν → ∞ approaches Normal). Heavier tails ⇒ fatter loss function in the tails ⇒ less influence
  of single crashes on parameter estimates.
- Quasi-likelihood: parameters are Maximum-Likelihood estimates under the assumed t-density;
  with a possibly misspecified density these are quasi-MLEs, for which standard asymptotic
  (std error, z, p-value) results still hold under correct conditional mean/variance
  specification.

Estimation sample control — `hold_back = 2`: the first 2 observations are excluded from the
likelihood for **every** candidate (also enough to initialize the AR(2) recursion). Effective
sample = 2,815 − 2 = 2,813 for all four. Without this, AR models would silently use fewer
observations than zero/constant models and AIC/BIC comparisons would be invalid (the research protocol).

Diagnostics recorded per fit:

- **Ljung–Box** on standardized residuals z_t at lags k ∈ {5, 10, 20}:
  Q = n(n+2) Σ_{k=1}^{K} ρ̂_k / (n−k) ~ χ²_K under H₀: no residual autocorrelation
  ⇒ tests whether the *mean* specification left linear structure behind.
- **Ljung–Box** on z_t² at the same lags ⇒ tests whether squared residuals still cluster
  (variance misspecification).
- **Engle ARCH LM** (nlags = 10) on raw residuals (before GARCH) and standardized residuals
  (after GARCH): H₀: no ARCH effects up to lag 10.
- **Persistence** ω̂ + α̂ + β̂: near-1 ⇒ shocks decay very slowly (long memory); >1 ⇒
  non-stationary variance, a specification failure to report.

Forecast scoring (validation only, parameters frozen):

- Mean: RMSE = √mean((r_t − μ̂_t)²), MAE = mean|r_t − μ̂_t|, in **percentage-point** log-return
  units (`return_pct`).
- Variance: QLIKE(r_t, σ̂_t²) = mean( r_t²/σ̂_t² + ln σ̂_t² ).
  QLIKE is minimized in expectation by the true conditional variance even when the "realized
  variance" proxy (here r_t²) is noisy (Patton 2011); the squared return is unbiased for one-day
  variance up to a constant, and QLIKE's log term penalizes under-stating volatility.

Baselines: zero forecast (μ̂_t ≡ 0), training-mean forecast (μ̂_t ≡ mean of training returns),
constant variance forecast (variance = training sample variance) for QLIKE.

**Selection rule (pre-declared):** AIC/BIC are recorded but are *not* sufficient. A candidate
"wins" only by (a) converging, (b) passing residual diagnostics (no significant Ljung–Box/ARCH-LM
after correction for multiple lags), and (c) improving validation RMSE/MAE over both baselines.
Ties in diagnostics ⇒ prefer the simpler mean (parsimony).

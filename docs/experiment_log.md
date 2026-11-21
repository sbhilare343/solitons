# Experiment Log (append-only)

Register each experiment **before** running it, then append results. Corrections are appended, never
silently edited. Each record states the question, data, protocol, model settings, metrics, results,
interpretation, and limitations.

---

## EXP-000 — Data provenance and integrity (retrospective registration)

- **Status:** PARTIAL / retrospective registration. The initial integrity checks predated the
  standardized experiment-record format; subsequent checks and findings are recorded below.
- **Question:** Is `data/processed/sp500_returns.csv` internally consistent with
  `data/raw/sp500.csv`, and are trading dates real and unique?
- **Observations of record:** processed file has 4,022 rows, 2010-01-05 → 2025-12-30, columns
  Date, Close, log_price, log_return, return_pct; raw file has 4,023 OHLCV rows from 2010-01-04
  (first row lost to differencing); split recomputed live as train 2,815 / val 603 / test 604 with
  boundaries matching the plan snapshot.
- **Gaps:** raw-vs-processed Close equality, `return_pct = 100·Δlog(Close)` identity, and exchange
  holiday integrity have **not** been asserted in an automated test. Open.

  **GAP CLOSED (2026-11-16):** `tests/test_data_provenance.py` now asserts all of
  these — raw≡processed Close (exact), log_price ≡ log(Close), return_pct ≡ 100·Δlog(Close) and
  ≡100·log_return (max diff 8.9e-16), OHLC sanity (H≥max(O,C), L≤min(O,C), positive prices,
  non-negative volume), weekday-only strictly-increasing unique dates, registered row counts
  (4,023 raw / 4,022 processed), processed dates = raw dates minus the first. Residual gap:
  no external exchange-calendar cross-check (weekday assertion only) — recorded in D-013.
  Status: DONE-in-substance.

---

## EXP-001 — Exploratory stationarity and dependence diagnostics (retrospective registration)

- **Status:** PARTIAL / retrospective registration. Run unlogged; code `src/classical/diagnostics.py`,
  artifacts `results/figures/*.png` (dated 2025-09-18).
- **Question:** Are returns (near-)stationary, and is there linear and squared-residual dependence
  (ARCH effects) to justify a GARCH variance model?
- **Recorded metrics:** ADF, Ljung-Box on returns and squared returns (lags 5/10/20), Engle ARCH LM
  (nlags=10) — printed only, not persisted as machine-readable values. Open: persist these numbers.

---

## EXP-002 — Conditional mean candidates vs. zero and training-mean baselines (retrospective registration)

- **Status:** PARTIAL / retrospective registration. Run unlogged; code `src/classical/sarima.py`,
  artifact `results/forecasts/validation_arima_candidates.csv` (603 validation rows,
  2021-03-12 → 2023-08-03; columns: zero_forecast, training_mean_forecast, ARIMA 0/0/0, 1/0/0,
  1/0/1, 2/0/0, 2/0/1).
- **Question:** Do small ARIMA mean models beat the zero and training-mean baselines on validation
  one-step RMSE/MAE?
- **Gap:** metric summary was not persisted; candidate ranking by AIC was reported but not persisted in a result artifact.

---

## EXP-003 — Four-way joint conditional-mean specification comparison (GARCH(1,1)-t)

- **Status:** REGISTERED (pre-run) on 2026-11-16. Code: `src/classical/garch_mean_spec.py`.
- **Question:** With the GARCH(1,1)-Student-t variance held fixed, does the conditional-mean
  specification (zero vs. constant vs. AR(1) vs. AR(2)) materially change convergence, diagnostics,
  and validation one-step forecast quality?
- **Objective:** Fit all four candidates on **training data only** under an identical effective
  sample (`hold_back=2`), record full diagnostics, and compare validation forecasts against zero
  and training-mean baselines — without selecting on AIC alone.
- **Testable hypothesis (H1):** At least one non-zero mean specification passes convergence and
  white-noise diagnostics that the zero-mean model fails, and improves validation RMSE/MAE over
  both baselines. Null / admissible negative result: mean dynamics beyond a constant are not
  detectable and no candidate beats the baselines on validation.
- **Concepts / equations:**
  - r_t = μ_t + ε_t, ε_t = σ_t z_t, z_t ~ standardized Student-t(ν)
  - σ_t² = ω + α ε_{t-1}² + β σ_{t-1}², ω > 0, α, β ≥ 0
  - μ_t ∈ {0, μ, μ + φ₁r_{t-1}, μ + φ₁r_{t-1} + φ₂r_{t-2}}
  - Persistence = ω + α + β (variance stationarity requires < 1, not enforced by the optimizer —
    reported, not assumed)
  - Ljung–Box Q = n(n+2) Σ_k ρ_k/(n−k); Engle ARCH LM; QLIKE(r_t, σ_t²) = r_t²/σ_t² + ln σ_t²
  - Common `hold_back=2` ⇒ identical effective sample across candidates, which is the precondition
    for even within-family AIC/BIC comparison.
- **Dataset / partition:** `data/processed/sp500_returns.csv`, target `return_pct` (percentage-point
  log return). Train 2,815 obs (2010-01-05 → 2021-03-11) for fitting; validation 603 obs
  (2021-03-12 → 2023-08-03) for forecast comparison; **test partition is not read by this
  experiment**.
- **Forecast origin / horizon / update policy:** rolling one-step ahead; parameters fixed after
  training; at each origin the forecast is computed from history only, then the target is revealed
  and appended to history. Variance recursion initialized from the last in-sample fitted σ² and ε.
- **Candidates:** arch GARCH(1,1), dist="t", rescale=False, hold_back=2; mean ∈ {Zero, Constant,
  AR lags=1, AR lags=2}. Baselines: zero forecast, training-mean forecast, constant variance =
  training sample variance (for QLIKE only).
- **Configuration / seeds:** deterministic optimization (SLSQP, arch defaults); no RNG used.
- **Metrics:** diagnostics (convergence flag/message, effective sample, log-likelihood, AIC/BIC,
  coefficients, std errors, z, p-values, ν, persistence, Ljung–Box on standardized and squared
  standardized residuals at lags 5/10/20, ARCH LM nlags=10 before and after); validation RMSE/MAE
  (mean) and QLIKE (variance, proxy = r_t²).
- **Null hypotheses:** (mean) no improvement over baselines; (diagnostics) residual autocorrelation
  and ARCH effects are absent (reject ⇒ specification inadequate).
- **Outputs:** `results/tables/exp003_garch_mean_spec_<runid>.json`,
  `results/forecasts/exp003_validation_forecasts_<runid>.csv` (unique run IDs, no overwrites).
- **Acceptance criteria:** all four fits report convergence status honestly (failures recorded);
  effective sample identical across candidates; forecasts computed before reveal (unit-tested);
  results not selected on AIC alone; this entry updated with actual results after the run.


### EXP-003 — Results (appended post-run, 2026-11-16)

- **Run ID:** `20261116T113919Z_6cb9`; git at run time: `b03b9ed` (pre-experiment source revision; this batch
  of changes was not yet committed when the artifacts were produced).
- **Command:** `uv run python -m src.classical.garch_mean_spec` (repo root). Exit 0.
- **Warnings during fits:** none for any specification. Runtime warning observed (not suppressed):
  statsmodels `FutureWarning` — `acorr_lm` return-type change planned for 0.16 (from `het_arch`),
  emitted 12× in tests, 8× in the run.
- **Artifacts:**
  - `results/tables/exp003_garch_mean_spec_20261116T113919Z_6cb9.json`
  - `results/forecasts/exp003_validation_forecasts_20261116T113919Z_6cb9.csv` (603 rows)
  - `results/tables/exp003_run_log_20261116T113919Z_6cb9.txt` (stdout copy)

**Fit results (train n=2,815, hold_back=2, n_eff=2,813 for all; all `convergence_flag=0`,
"Optimization terminated successfully", zero warnings):**

| spec | logL | AIC | BIC | persistence ω+α+β | persistence < 1 |
|---|---|---|---|---|---|
| zero | −3458.991 | 6925.982 | 6949.750 | 1.0164 | **no** |
| constant | −3429.715 | 6869.429 | 6899.139 | 1.0201 | **no** |
| ar1 | −3423.881 | 6859.763 | 6895.415 | 1.0200 | **no** |
| ar2 | −3423.840 | 6861.680 | 6903.274 | 1.0200 | **no** |

Key coefficients (coef, s.e., p):

- constant: μ̂ = 0.09227 (0.01164, p=2.2e−15)
- ar1: Const = 0.09949 (0.01216, p=2.9e−16), φ̂₁ = −0.06631 (0.01863, p=3.7e−4)
- ar2: Const = 0.10017 (p=3.5e−15), φ̂₁ = −0.06657 (p=3.5e−4), φ̂₂ = −0.00566 (0.02045, p=0.78)
- all specs: ω̂ ≈ 0.0256–0.0267 (p<1e−4), α̂ ≈ 0.173–0.187 (p<1e−14), β̂ ≈ 0.806–0.818
  (p≈0), ν̂ ≈ 4.83–5.10 (p<1e−23) ⇒ heavy tails strongly supported (Normal rejected in spirit).

**Residual diagnostics (all specs):**

- Ljung–Box on standardized residuals: all p ≥ 0.078 (lags 5/10/20) ⇒ no leftover linear
  dependence.
- Ljung–Box on squared standardized residuals: all p ≥ 0.54 ⇒ no leftover volatility clustering.
- ARCH LM (nlags=10): raw residuals p ≈ 1e−220 (ARCH present before GARCH), standardized
  residuals p ∈ [0.56, 0.67] (removed after GARCH).
- ⇒ **all four candidates pass the pre-declared diagnostic gate**; diagnostics do not
  differentiate them.

**Validation one-step forecasts (603 dates, 2021-03-12 → 2023-08-03, frozen parameters):**

| candidate | RMSE | MAE | QLIKE (proxy r²) |
|---|---|---|---|
| zero | **1.1618** | **0.8636** | 1.1256 |
| constant | 1.1637 | 0.8651 | 1.1153 |
| ar1 | 1.1669 | 0.8692 | 1.1122 |
| ar2 | 1.1666 | 0.8689 | 1.1122 |
| baseline: zero forecast | **1.1618** | **0.8636** | — |
| baseline: training mean (0.0443) | 1.1618 | 0.8638 | — |
| baseline: constant variance (1.2220) | — | — | 1.3050 |

**Interpretation (vs. pre-declared hypothesis):**

1. **H₁ is not supported for the mean.** No non-zero mean specification beats the zero or
   training-mean baseline on validation RMSE/MAE; all are 0.16–0.44 % *worse*. The in-sample
   significance of μ̂ (p≈1e−15) and φ̂₁ (p≈4e−4) does **not** translate into forecast gains —
   a clean negative result, recorded as such.
2. **The variance layer is decisively useful.** QLIKE falls from 1.3050 (constant training
   variance) to 1.1122–1.1256 (≈ 13.8–14.8 % reduction) for every candidate.
3. **AIC alone would have selected `ar1`** (AIC 6859.8 < constant 6869.4 < zero 6926.0), yet
   `ar1` is the *worst* mean forecaster on validation — direct evidence for the pre-declared
   "do not select on AIC alone" rule (the research protocol).
4. **Persistence > 1 for all four fits** (1.016–1.020): the fitted variance process is slightly
   non-stationary (near-IGARCH behaviour on this sample). One-step rolling forecasts are
   unaffected (recursion re-anchored daily), but any long-horizon variance extrapolation would
   explode. Reported, not hidden; ω+α+β < 1 was a diagnostic, not an enforced constraint.
5. **Per the pre-declared rule no candidate "wins" the mean comparison**; parsimony favours the
   zero (or constant) mean for point forecasting. The ar1/ar2 QLIKE edge (1.1122 vs 1.1256)
   comes from their variance recursion, not from mean accuracy.

**Limitations:** single chronological split; QLIKE proxy r² is noisy (level inflated, ranking
preserved per Patton 2011); RMSE/MAE differences among candidates are *not yet* tested for
statistical significance (Diebold–Mariano is EXP-008); persistence > 1 unexplained; no test-set
data read; no combination weights or Transformer involved yet.

**Open questions:** (a) Is persistence > 1 a sample artifact (2010–2021 low-vol regime) or
misspecification (e.g. need for E-GARCH/APARCH or a variance-mean interplay)? (b) Are the tiny
RMSE differences distinguishable from zero under HAC-corrected DM tests? (c) Does the constant
mean's validation RMSE improve in subsamples with drift?

**Reproduction:** `uv run python -m src.classical.garch_mean_spec` (metrics deterministic;
artifacts carry fresh run IDs — see `docs/runbooks/exp003_garch_mean_spec.md`).

---

## EXP-004 — Validation variance evaluation across justified volatility proxies

- **Status:** REGISTERED (pre-run) on 2026-11-16. Code: `src/evaluation/variance_proxy_eval.py`
  + primitives in `src/evaluation/metrics.py`.
- **Question:** Do the frozen EXP-003 GARCH variance forecasts beat *strong* variance baselines —
  not just a constant — and is their ranking robust to the choice of volatility proxy?
- **Objective:** Score the **unchanged** EXP-003 one-step variance forecasts on the same 603
  validation dates with QLIKE under three pre-declared proxies, add two variance-forecast
  baselines, and persist consolidated mean (RMSE/MAE) and variance (QLIKE) tables. **No refit** —
  forecasts are read from the EXP-003 artifact so the two experiments score literally identical
  numbers. Test partition remains unread.
- **Testable hypotheses:**
  - **H₁ (variance value):** every GARCH candidate achieves lower QLIKE than *both* baselines
    (constant training variance; EWMA(0.94) one-step forecast) under *every* proxy.
  - **H₂ (proxy robustness):** the ordering of the four candidates is identical under all three
    proxies.
  - Nulls / admissible negatives: GARCH does not beat the EWMA(0.94) forecast on some or all
    proxies (EWMA is a serious competitor — same recursion family, one tuned-free constant);
    candidate ranking changes with proxy ⇒ metric-dependent conclusions, reported as such.
- **Concepts / equations:**
  - QLIKE-with-proxy: L(σ̂²; q) = mean( q_t/σ̂_t² + ln σ̂_t² ); identical to Patton (2011)'s
    QLIKE up to terms free of σ̂², so cross-model rankings coincide.
  - P1 squared return: q_t = r_t² (noisy, unit-consistent with `return_pct`²).
  - P2 Parkinson range: q_t = ln(H_t/L_t)² / (4 ln 2), computed from **raw** OHLC and converted
    decimal² → return_pct² by ×10⁴ (units discipline, the research protocol); lower sampling noise than
    r² under a Brownian-motion efficiency argument.
  - P3 smoothed proxy: q_t = 0.94·q_{t−1} + 0.06·r_t² (EWMA of squared returns, λ = 0.94 fixed
    by RiskMetrics convention — **not** tuned on validation); initialized with the training
    sample variance. Uses the current r_t (it is a *measurement*, available only ex post).
  - Forecast baselines (genuine one-step forecasts, no future information):
    B1 constant σ̂² = training sample variance; B2 EWMA forecast
    σ̂²_t = 0.94·σ̂²_{t−1} + 0.06·r²_{t−1}, initialized with the training sample variance.
  - Unit check: all q_t and σ̂²_t in return_pct² (percentage-point²) units; QLIKE's ln σ̂² term
    is unit-dependent in level, so mixed units would silently distort comparisons.
- **Dataset / partition:** EXP-003 forecast artifact
  (`results/forecasts/exp003_validation_forecasts_<runid>.csv`, 603 rows) + actual
  `return_pct` from `data/processed/sp500_returns.csv` + raw OHLC from `data/raw/sp500.csv`
  (read-only). Validation only; **test partition not read**.
- **Forecast origin / horizon / update policy:** inherited unchanged from EXP-003 — rolling
  one-step, frozen parameters, forecast-before-reveal, identical 603 dates.
- **Candidates / baselines:** the four EXP-003 mean specs (zero, constant, ar1, ar2) as variance
  carriers; baselines B1, B2; mean metrics (RMSE/MAE) re-persisted for consolidation.
- **Configuration / seeds:** λ = 0.94 pre-declared convention; no RNG; no free parameters
  estimated in this experiment.
- **Metrics:** QLIKE per (candidate ∪ baseline) × (P1, P2, P3); RMSE/MAE per candidate for the
  mean; relative QLIKE change vs B2 per candidate.
- **Acceptance criteria:** all proxies defined on all 603 registered dates (join asserted, raw
  OHLC missing dates reported); forecasts byte-identical to EXP-003 artifact (read, not
  recomputed); unit conversion verified by test; results appended below regardless of sign.
- **Pre-registered robustness sensitivity (persistence > 1, from EXP-003 finding D-007):**
  **EXP-004-S1** — refit the four specs with ω+α+β ≤ 1 imposed by reparameterization
  (e.g. α+β = 1−δ, ω = δ·σ̄² with δ > 0 via softplus; optimized with scipy over
  `arch_model(...).fix(params).loglikelihood`, feasibility verified 2026-11-16), same
  `hold_back=2`, same protocol; if the constrained fit is numerically unstable, fall back to
  EGARCH(1,1)-t and record the substitution. Decision rule declared now: **if candidate
  rankings and GARCH-vs-baseline ordering are unchanged vs EXP-003/EXP-004, persistence>1 is
  immaterial for one-step evaluation and stands as reported; if they change, the EXP-003
  conclusion must be qualified in EXP-009.** S1 runs only after EXP-004 results are frozen.
- **Outputs:** `results/tables/exp004_variance_proxy_eval_<runid>.json`,
  `results/tables/exp004_qlike_by_proxy_<runid>.csv`.


### EXP-004 — Results (appended post-run, 2026-11-16)

- **Run ID:** `20261116T124356Z_d85f`; git at run time: `447d1b5`.
- **Command:** `uv run python -m src.evaluation.variance_proxy_eval` (repo root). Exit 0,
  **no warnings emitted** (this experiment performs no model fitting).
- **Source:** frozen EXP-003 artifact
  `results/forecasts/exp003_validation_forecasts_20261116T113919Z_6cb9.csv` — forecasts were
  *read, not recomputed*; 603 registered validation dates; raw OHLC join 603/603 complete;
  train used only for initialization (variance 1.22198, ddof=1); test partition unread.
- **Artifacts:**
  - `results/tables/exp004_variance_proxy_eval_20261116T124356Z_d85f.json`
  - `results/tables/exp004_qlike_by_proxy_20261116T124356Z_d85f.csv` (18 rows = 3 proxies × 6 models)
  - `results/tables/exp004_run_log_20261116T124356Z_d85f.txt` (stdout copy)

**QLIKE by proxy (lower = better; levels are NOT comparable across proxies — only rankings
within a proxy column are):**

| model | P1 r² | P2 Parkinson | P3 EWMA(0.94) |
|---|---|---|---|
| zero | 1.1256 | 0.7147 | 1.1052 |
| constant | 1.1153 | **0.7050** | 1.1124 |
| ar1 | 1.1122 | 0.7061 | 1.1117 |
| ar2 | 1.1122 | 0.7057 | 1.1117 |
| b1 constant variance | 1.3050 | 0.8821 | 1.3222 |
| b2 EWMA(0.94) forecast | **1.1116** | 0.7172 | **1.0749** |

**Relative QLIKE improvement vs. b2 (positive = candidate beats the EWMA forecast baseline):**

| candidate | P1 | P2 | P3 |
|---|---|---|---|
| zero | −1.259 % | +0.355 % | −2.812 % |
| constant | −0.336 % | **+1.707 %** | −3.482 % |
| ar1 | −0.060 % | +1.547 % | −3.421 % |
| ar2 | −0.059 % | +1.601 % | −3.419 % |

**Candidate rankings (best → worst) per proxy:**

- P1 squared return: `ar2 > ar1 > constant > zero`
- P2 Parkinson: `constant > ar2 > ar1 > zero`
- P3 EWMA: `zero > ar2 > ar1 > constant`

**Mean forecasts (consolidated from the same artifact):** zero 1.1618/0.8636 (RMSE/MAE) —
unchanged from EXP-003, confirming no recomputation drift; all candidates 0.16–0.44 % worse.

**Interpretation vs. pre-declared hypotheses:**

1. **H₁ rejected.** Not every candidate beats both baselines under every proxy. What *does*
   hold robustly: **every candidate beats b1 (constant variance) under every proxy** by
   13.8–20.5 % — the EXP-003 variance-layer conclusion generalizes across proxies. But the
   serious baseline **b2 EWMA(0.94) beats every candidate on P1** (by 0.06–1.26 %) **and on P3**
   (by 2.8–3.5 %); only under P2 (Parkinson) do all four candidates beat b2 (+0.4…+1.7 %).
2. **H₂ rejected.** Candidate rankings are *not* proxy-invariant — the ordering flips
   completely across the three proxies (P1 favours ar2, P2 favours constant, P3 favours zero).
   However, the candidate spread is tiny (≤1.4 % within any proxy), so these are plausibly
   near-ties whose ordering is noise; **no "best spec" claim is made without EXP-008 DM tests.**
3. **Proxy choice changes conclusions** — the experiment's central practical lesson: with the
   conventional r² proxy, a bare EWMA(0.94) forecast is at least as good as GARCH(1,1) here;
   with the lower-noise independent Parkinson range proxy, GARCH wins modestly.
4. **Structural caveat (recorded honestly):** P3 is an EWMA-smoothed measurement built with the
   *same* λ as the b2 forecast (b2 is essentially the P3 recursion lagged one day), so b2's P3
   advantage is partly mechanical proxy-alignment, not independent evidence. P2 — the only proxy
   not built from returns — is the least aligned with any candidate and is where GARCH wins.

**Limitations:** QLIKE point estimates only (no HAC-corrected significance — EXP-008 pending);
single split; λ = 0.94 a fixed convention (not tuned, but not varied either); Parkinson assumes
Brownian motion within the day (jumps/overnight gaps bias it); P1/P3 built from returns share
data with the forecast recursion; b1 uses training variance only (a deliberately weak baseline);
test partition unread.

**Open questions:** (a) is b2's P1 edge (0.06 %) statistically distinguishable from zero?
(b) does the GARCH-vs-b2 ordering hold under an independent high-frequency variance proxy?
(c) does λ variation (0.94 vs 0.97) change P3 conclusions — pre-declare any such sweep before
running it.

**Reproduction:** `uv run python -m src.evaluation.variance_proxy_eval` (deterministic given the
EXP-003 artifact; a fresh EXP-003 run would produce fresh run IDs on both sides).

---

## EXP-004-S1 — Persistence-constrained refit (registered inside EXP-004 above)

- **Status:** INVOKED (pre-run) on 2026-11-16, after EXP-004 results were frozen in commit
  `48edd1c`. Code: `src/classical/garch_constrained.py`.
- **Scope:** as registered in the EXP-004 entry — re-fit the four mean specs with ω+α+β ≤ 1
  imposed, same `hold_back=2`, same train-only data, same rolling one-step validation protocol;
  apply the **pre-declared decision rule** (rankings + GARCH-vs-baseline ordering unchanged ⇒
  persistence>1 immaterial for one-step; changed ⇒ qualify EXP-003 in EXP-009).
- **Implementation deviation (disclosed, registered example was "e.g."):** the illustrative
  parametrization ω = δ·σ̄² does **not** guarantee ω+α+β ≤ 1 when σ̄² > 1 (here σ̄² = 1.22).
  Actual parametrization enforces the constraint directly:
  p = 1 − softplus(d) < 1; ω = sigmoid(o)·p; α = sigmoid(s)·(p−ω); β = (p−ω) − α;
  ν = softplus(ν_raw) + 2.01; mean parameters pass through unchanged. Optimized with scipy
  L-BFGS-B over `arch_model(...).fix(params).loglikelihood`.
- **Known limitation declared in advance:** standard errors / p-values are **not** computed under
  the inequality constraints (no delta-method Hessian implemented); S1 compares convergence,
  LL/AIC/BIC, parameters, persistence, residual diagnostics, and validation forecasts only —
  which is all the decision rule requires.

### EXP-004-S1 — Results (appended post-run, 2026-11-16)

- **Run ID:** `20261116T131945Z_8ec9`; git at run time: `879d49d`. Command:
  `uv run python -m src.classical.garch_constrained` → exit 0. 4× `FutureWarning` from
  `het_arch` (reported); **no optimizer warnings**; all 4 fits converged
  ("CONVERGENCE: RELATIVE REDUCTION OF F <= ...").
- **Artifacts:** `results/tables/exp004s1_constrained_garch_20261116T131945Z_8ec9.json`,
  `results/tables/exp004s1_run_log_20261116T131945Z_8ec9.txt`.
- **Constraint binding:** persistence = 1.000000 (boundary) for all specs — the unconstrained
  MLE wanted p ∈ [1.016, 1.020], so the constraint binds exactly as expected. α falls
  (e.g. zero: 0.1730 → 0.1528), β rises (0.8179 → 0.8205), ν rises (5.10 → 5.62).
- **Cost of the constraint:** ΔlogL = −1.46…−1.84, ΔAIC = +2.9…+3.7 (constrained fits worse in
  sample — the data *want* p > 1).
- **Validation forecasts:** mean RMSE/MAE unchanged to ≤3e-5 (zero spec identical by
  construction); QLIKE(r²) degrades slightly (e.g. ar1 1.11224 → 1.11413). Diagnostics still
  pass (LB(z) p@10 ≥ 0.150; ARCH-LM on standardized residuals p ≥ 0.470).
- **Pre-declared decision rule outcome:**
  - mean RMSE ranking: `[zero, constant, ar2, ar1]` both sides — **unchanged**;
  - QLIKE(r²) ranking: exp003 `[ar2, ar1, constant, zero]` vs s1 `[ar1, ar2, constant, zero]`
    — **ar1/ar2 swapped** (they were tied to 1.3e-5 apart; now 3.2e-5 apart);
  - baseline ordering: all candidates beat b1, none beat b2 — **unchanged**;
  - `rankings_unchanged = false`, `baseline_ordering_unchanged = true`
    ⇒ **VERDICT: MATERIAL — EXP-003 conclusions must be qualified in EXP-009.**
- **Qualification text (carried into EXP-009 and the report):** *the persistence>1 constraint
  is binding and costs ≈1.5–1.8 log-likelihood points; it leaves the mean-forecast conclusion
  (no candidate beats the zero baseline) and the variance-baseline ordering (candidates beat
  constant variance, lose to EWMA(0.94) on r²) fully intact; the only ranking change is the
  ar1↔ar2 variance near-tie, which is not interpretable as a substantive difference without
  significance testing. Any EXP-009 statement about ar1 vs ar2 variance superiority must be
  flagged as persistence-specification-sensitive.*
- **Limitations:** no std_err/p-values under inequality constraints (declared); boundary optimum
  makes the effective model IGARCH(1,1); single validation window.


_(pending placeholder from pre-run registration retained above; this entry supersedes it.)_

---

## EXP-005 — PatchTST: small univariate one-step forecaster

- **Status:** REGISTERED (pre-run) on 2026-11-16. Code: `src/transformer/{patchtst,dataset,train}.py`.
- **Question:** Can a deliberately small PatchTST learn predictive structure in daily returns
  that survives out-of-sample one-step evaluation — without any hyperparameter search?
- **Objective:** train one univariate PatchTST on **train data only** (including an internal
  early-stopping split *inside* train so the registered validation window stays untouched),
  then produce rolling one-step validation forecasts on the identical 603 dates/origins as the
  classical models.
- **Testable hypothesis (H₁):** PatchTST validation RMSE ≤ zero-baseline RMSE (1.1618) and is
  competitive with the best classical mean spec. Admissible negative result: PatchTST fails to
  beat the zero baseline — a Transformer that cannot beat "predict zero" on daily returns is a
  publishable negative finding for this sample.
- **Concepts / equations:** patching (split lookback into overlapping patches, stride s →
  P = (L−p)/s + 1 tokens), channel independence (trivial for univariate), Transformer encoder
  (multi-head self-attention over patches), flatten head → next-step point forecast; z-score
  scaling μ̂, σ̂ fit on train only: z = (r−μ̂)/σ̂, forecasts un-scaled as ŷ = ŷ_z·σ̂ + μ̂;
  MSE loss on scaled targets; early stopping = model selection (permitted on internal split,
  not on registered validation).
- **Pre-declared configuration (no tuning loop; everything recorded):** lookback L = 96;
  patch_len = 16; stride = 8 (P = 11 tokens); d_model = 64; heads = 4; layers = 2; FFN = 128;
  dropout = 0.1; learnable positional embedding (N(0, 0.02)); head = Flatten → Linear(d_model·P,
  1); optimizer Adam lr = 1e-3, weight_decay = 1e-4; batch 64; max epochs 50; early-stopping
  patience 8 on internal-val MSE (restore best); **seed 42** (python/numpy/torch);
  **device CPU** (determinism over speed; CUDA present but not used); loss MSE.
- **Two-phase protocol:** Phase 1 — find best epoch via early stopping on the internal split
  (last 15% ≈ 422 train observations as targets; training targets strictly before it).
  Phase 2 — retrain a fresh seeded model on **all** train windows for exactly best_epoch
  epochs (uses the full 2,815 like the classical fits; no validation contact).
- **Forecast policy:** identical to classical — rolling one-step, weights frozen after Phase 2,
  window for origin t = scaled series [t−96, t−1] (revealed history only; batched construction
  is protocol-equivalent to looping because windows end at t−1), forecast before reveal.
- **Scope reductions vs. the PatchTST paper (declared):** no RevIN, no HLF multi-scale encoder,
  no channel-mixing (univariate), single pre-declared config — this is the registered
  "small first model", not a tuned benchmark system.
- **Metrics:** validation RMSE/MAE (percentage points) on identical dates; per-date loss
  differentials saved for EXP-008 DM tests.
- **Outputs:** `results/forecasts/exp005_patchtst_validation_<runid>.csv`,
  `results/tables/exp005_patchtst_<runid>.json` (full config, seed, scaler params, phase-1
  curve + best epoch, phase-2 losses, git commit, source artifact linkage).
- **Acceptance criteria:** config fully recorded; determinism test passes (two seeded runs →
  identical losses); no-future-leakage test passes; scaler fitted on train only (asserted);
  test partition unread; results appended below regardless of sign.


- **Protocol correction before first execution (2026-11-16):** `src/transformer/train.py`
  originally shuffled supervised windows within the training partition. This was changed to
  chronological window order to honor the no-shuffling rule in the research protocol. Batches remain contiguous;
  the registered architecture, optimizer, epoch-selection split, and validation origins are
  unchanged. This correction is pre-run, not a post-result adjustment.

### EXP-005 — Results (appended post-run)

_(pending)_

---

## EXP-007 — Validation-only forecast combinations and encompassing

- **Status:** REGISTERED before execution (2026-11-16). Inputs: EXP-003 classical validation
  forecasts and EXP-005 PatchTST validation forecast; test partition remains sealed.
- **Question/objective:** Do fixed convex combinations improve validation squared-error loss,
  and does either forecast add conditional predictive information to the other?
- **Hypotheses:** H1: at least one predeclared pair's validation-fitted convex combination has
  lower MSE than both constituents. H0: no improvement. Encompassing H0 for candidate B given A:
  its incremental coefficient in `y_t = c + b_A f_A,t + b_B f_B,t + u_t` is zero; also report
  joint forecast-signal test and HAC uncertainty. Regressions are descriptive on validation,
  not proof of generalization.
- **Concepts/equations:** `f_t(w)=w f_A,t+(1-w)f_B,t`, `w∈[0,1]`, choose w by minimizing
  validation SSE; closed-form unconstrained `w*=Σ(a−b)(y−b)/Σ(a−b)^2`, then clip to [0,1].
  All weight fitting occurs on validation and is frozen before test. Encompassing regressions
  use OLS with intercept and Newey–West HAC standard errors (Bartlett, lag 4); nulls tested at
  5%, with coefficients and confidence intervals reported.
- **Dataset/partition/protocol:** `return_pct`, 603 validation dates 2021-03-12…2023-08-03,
  one-step, rolling revealed history, frozen model parameters; identical origins only.
- **Candidates:** each EXP-003 mean forecast and zero/training-mean baselines against PatchTST;
  combinations are validation-trained for all eligible pairs, with the best pair identified by
  validation MSE and selection optimism disclosed. No variance forecasts are combined here.
- **Seeds/config:** none (deterministic convex optimization and OLS); no tuning beyond
  validation weight estimation.
- **Metrics/nulls:** RMSE/MAE; combination MSE vs each constituent (paired loss difference);
  encompassing coefficient and joint-signal Wald tests with HAC covariance.
- **Outputs/acceptance:** unique JSON/CSV, exact date equality required, weights within [0,1],
  no test data in inputs. Report every pair and null result.
- **Limitations:** one validation period; many pairwise comparisons; in-sample weight fitting
  makes combination metrics optimistic; HAC inference is low-power and not multiplicity-adjusted.
- **Reproduction:** `uv run python -m src.evaluation.combination`.

## EXP-008 — Diebold–Mariano forecast-loss comparisons

- **Status:** REGISTERED before execution (2026-11-16); test application only after freeze.
- **Question/objective:** Are validation loss differences between the predeclared mean forecasts
  distinguishable from zero after allowing short-range serial dependence?
- **Hypothesis/null:** for each ordered pair, `H0:E[d_t]=0`, where d is squared-error loss
  differential (primary) or absolute-error differential (secondary); two-sided alternative.
- **Concepts/equations:** `DM = mean(d)/sqrt(Ω̂/T)`; Ω̂ is Bartlett Newey–West long-run variance
  with predeclared lag `floor(4*(T/100)^(2/9))` (capped below T). Apply Harvey–Leybourne–Newbold
  finite-sample factor for horizon h=1 (equals 1); t reference df T−1; report p-values and 95%
  intervals. Paired models must have identical dates. Validation is primary; test is one sealed
  confirmatory application after freeze; no test-driven changes.
- **Dataset/protocol:** same validation 603 origins; one-step; then the frozen forecast set on
  test 604 dates exactly once. Forecasts use revealed prior outcomes and fixed fitted weights.
- **Models/metrics:** all mean forecasts in EXP-006/007; MSE and MAE differentials. Variance
  QLIKE sensitivity comparisons may be reported by proxy only when aligned forecasts exist.
- **Outputs/acceptance:** pairwise JSON/CSV, date and horizon checks, HAC lag and correction
  recorded; no significance claim from raw RMSE alone.
- **Limitations:** correlated pairwise tests, no familywise correction, one asset and horizon.
- **Reproduction:** `uv run python -m src.evaluation.dm_test`.

## EXP-009 — Sealed final test (registration; run only after freeze)

- **Status:** REGISTERED; execution gate: `docs/freeze_manifest.json` must be written and
  committed before the test partition is read. The manifest freezes target/units, selected
  single model and combination, all parameters, training scaler, epoch budget, validation
  weights, one-step update policy, date window, and analysis code hash.
- **Question/objective:** Does the validation-frozen forecast system retain its predictive
  performance on 604 untouched dates 2023-08-04…2025-12-30?
- **Hypothesis/null:** primary comparison is frozen selected system vs zero-return baseline in
  squared-error DM test, `H0:E[d_t]=0`; secondary comparisons use frozen constituents and the
  predeclared model set. One pass only; no refitting, reweighting, or reselection after outcomes.
- **Concepts/equations:** validation-frozen weights and settings; rolling one-step forecasts use
  each newly revealed return only as history for the next origin; scores and HAC DM as EXP-008.
- **Outputs:** unique forecast CSV, metrics/DM JSON, run log; actual outcomes appended after run.
- **Limitations:** one test interval, potential regime dependence, no external-market claim.
- **Reproduction:** `uv run python -m src.evaluation.final_test`.

### EXP-001/002 — Results persisted (retrospective gap closure, 2026-11-16)

- **Artifact:** `results/tables/exp001_exp002_metrics_20261116T165350Z_33d6.json`.
- **EXP-001 partition:** training only, n=2,815, 2010-01-05…2021-03-11. ADF statistic and
  p-value = recorded artifact (p=1.54e-21); reject unit-root null. ARCH LM(10) p=3.69e-232,
  strong conditional heteroskedasticity evidence. Ljung–Box(returns and squared returns), skew,
  excess kurtosis, sample boundaries, and critical values are in JSON. These support conditional
  mean/variance modeling; they do not prove a particular model is correct.
- **EXP-002 partition:** existing frozen validation forecasts, n=603, 2021-03-12…2023-08-03.
  RMSE (percentage points): zero 1.161796; training mean 1.161796; ARIMA(0,0,0) 1.161796;
  ARIMA(1,0,0) 1.176051; ARIMA(1,0,1) 1.179388; ARIMA(2,0,0) 1.183054; ARIMA(2,0,1)
  1.182860. None of the dynamic candidates beats the simple baselines. Result is reconstructed
  from the pre-existing forecast CSV, not a refit; test partition not used.
- **Limitations:** EXP-001's original exploratory plots/console run were retrospective and used
  the full available history; the persisted replacement statistics use training only. No external
  exchange calendar was checked; weekday-only date checks remain the documented limit.
- **Reproduction:** `UV_CACHE_DIR=/tmp/solitons-uv-cache MPLCONFIGDIR=/tmp/solitons-mpl uv run
  python -m src.experiments.complete_retro` (two statsmodels FutureWarnings, preserved).

### EXP-005 — Results (run 2026-11-16; supersedes pending placeholder)

- **Outcome:** completed the predeclared two-phase CPU training with seed 42, train-only scaler
  (mean 0.044268, SD 1.105236), 2,296 phase-1 train windows, 423 internal validation windows,
  selected epoch 10, and phase-2 retraining on all 2,719 train windows. PatchTST has 69,441
  trainable parameters. Validation RMSE=1.295198 and MAE=1.000554 percentage points versus
  zero RMSE=1.161796: H1 rejected; the registered model does not beat zero.
- **Runs/artifacts:** initial run `20261116T164753Z_4e0b` (forecast/JSON, no checkpoint), then
  deterministic checkpoint run `20261116T164839Z_17b6` with forecast CSV, JSON, and weights
  `results/tables/exp005_patchtst_weights_20261116T164839Z_17b6.pt` (checkpoint hash recorded
  in its JSON). Forecasts/metrics agree across runs.
- **Failures/corrections:** first run failed because `nn.Unfold` interpreted the temporal patch as
  a 16×16 kernel. Corrected to a singleton-height 2-D temporal kernel. First focused test run
  found float32 exact-equality assertions and an invalid test-boundary fixture; assertions now
  use float tolerances and fixture ends at validation boundary. The model warning about nested
  tensors with `norm_first=True` is emitted and retained.
- **Test partition:** not accessed by model inputs; runner now constructs inputs only through
  train+validation.
- **Reproduction:** `UV_CACHE_DIR=/tmp/solitons-uv-cache MPLCONFIGDIR=/tmp/solitons-mpl uv run
  python -m src.transformer.train` (exit 0; two documented PyTorch nested-tensor warnings).

### EXP-006 — Results (2026-11-16; supersedes pending placeholder)

- **Artifact:** `results/tables/exp006_fair_comparison_20261116T165032Z_b8f8.json` and CSV;
  same 603 dates, 2021-03-12…2023-08-03, source dates and actuals asserted equal.
- **Validation RMSE/MAE:** zero 1.161796/0.863599; training mean 1.161796/0.863793;
  GARCH means constant 1.163701/0.865138, AR1 1.166885/0.869240, AR2 1.166600/0.868866;
  PatchTST 1.295198/1.000554. Point-estimate H1 (PatchTST beats all) rejected. Zero and
  training mean differ by less than 0.000001 RMSE; treat them as a practical tie.
- **Limitations:** point estimates only; one validation period; PatchTST is one untuned small
  configuration. EXP-008 supplies paired HAC inference.
- **Reproduction:** `UV_CACHE_DIR=/tmp/solitons-uv-cache MPLCONFIGDIR=/tmp/solitons-mpl uv run
  python -m src.evaluation.compare` (exit 0; test partition not read).

### EXP-007 — Results (2026-11-16; supersedes pending placeholder)

- **Artifacts:** `results/tables/exp007_combinations_20261116T165108Z_75d9.json` and CSV.
- **Combination:** best validation pair is zero + training mean, with weights 0.499943 and
  0.500057, respectively; RMSE=1.161585, MAE=0.863545. RMSE gain is only 0.000211 over the
  better constituent (training mean); this tiny in-sample weight-fit gain is not strong evidence
  of generalizable value. All 15 pair weights and scores are retained in the artifact.
- **Encompassing:** HAC lag 5. Regressions where an expert forecast is constant are rank
  deficient with an intercept and are explicitly recorded `not_identified_rank_deficient`;
  only three non-collinear pairs were estimated. For AR1+PatchTST and AR2+PatchTST, joint
  forecast-signal p-values are 0.983 and 0.992. The results do not show incremental linear
  signal from these pairs. Pairwise tests are exploratory and not multiplicity adjusted.
- **Execution issue:** first run exposed rank deficiency and an index error when constant columns
  caused `add_constant` to omit the intercept. Runner now forces an intercept and records
  rank-deficient designs instead of estimating unidentified coefficients. The failed attempt is
  retained in this history; no result artifact from that failed run was created.
- **Reproduction:** `UV_CACHE_DIR=/tmp/solitons-uv-cache MPLCONFIGDIR=/tmp/solitons-mpl uv run
  python -m src.evaluation.combination` (successful run exit 0).

### EXP-008 — Validation results (2026-11-16; supersedes pending placeholder)

- **Artifact:** `results/tables/exp008_validation_dm_20261116T165239Z_63a2.json` and CSV.
- **Method:** paired squared/absolute losses, Bartlett Newey–West HAC lag 5, h=1 HLN correction
  factor `sqrt((T-1)/T)`, t reference df=602, two-sided tests, no multiplicity correction.
- **Key squared-loss result:** with d = loss(simple) − loss(PatchTST), d̄ is −0.327769
  (zero vs PatchTST; 95% t/HAC CI [−0.438633, −0.216905], p=1.06e-8) and −0.327769
  (training mean vs PatchTST; CI [−0.439293, −0.216245], p=1.29e-8). Negative d means the
  first model has lower loss. This validation window strongly disfavors PatchTST relative to
  simple baselines under this declared loss/HAC method. Other pairwise results are in the file.
- **Limitations:** exploratory multiple comparisons; p-values are unadjusted; asymptotic HAC
  assumptions and one validation regime; no claim about the sealed test.
- **Reproduction:** `UV_CACHE_DIR=/tmp/solitons-uv-cache MPLCONFIGDIR=/tmp/solitons-mpl uv run
  python -m src.evaluation.dm_test` (exit 0; test not read).

### EXP-007 — Deterministic rerun after module separation (correction/addendum)

- **Date:** 2026-11-16. The encompassing helper was moved from `combination.py` into the
  dedicated `encompassing.py` and covered by focused rank-deficiency tests (D-017). The runner
  was rerun; no estimator or selection rule changed.
- **New immutable artifact:** `results/tables/exp007_combinations_20261116T165831Z_7e52.json`
  (SHA256 `fd3b26a76ac4ed1d7b2c7bc0168e27b8991ea47542670b038342182ad1a31ea8`) and matching CSV.
  Best pair/weights/scores are bit-for-bit the same as the prior successful run. The freeze
  manifest points to this rerun. Earlier EXP-007 artifact remains preserved.

### EXP-009 — Sealed test results (2026-11-16; one run after freeze commit)

- **Freeze gate:** `docs/freeze_manifest.json` was committed in commit `6fb7b57` before this
  command opened the test partition. The test runner verified the manifest and checkpoint SHA256.
- **Artifacts:** `results/tables/exp009_test_results_20261116T182517Z_13fd.json` and
  `results/forecasts/exp009_test_forecasts_20261116T182517Z_13fd.csv`.
- **Window:** 604 dates, 2023-08-04…2025-12-30. The frozen combination is primary. No weights,
  model, scaler, epoch, or forecast policy were reselected after observing outcomes.
- **RMSE/MAE (percentage points):** zero 0.967302/0.652373; training mean
  0.965081/0.647674; constant GARCH mean 0.964965/0.645014; AR(1)-GARCH mean
  0.964290/0.646271; AR(2)-GARCH mean 0.964722/0.646694; PatchTST 1.099655/0.785724;
  frozen zero/training-mean combination 0.965939/0.649630.
- **Primary DM:** frozen combination minus each comparator under squared loss, Bartlett HAC lag
  5, h=1 HLN, t(df=603). Versus zero, mean differential −0.002636, p=0.104; versus training
  mean +0.001656, p=0.307; versus AR(1) +0.003183, p=0.802; versus AR(2) +0.002349, p=0.851.
  These intervals include zero. Versus PatchTST, differential −0.276203, p=1.25e-5; the
  combination has lower squared loss. Pairwise p-values are exploratory and not multiplicity
  adjusted.
- **Interpretation:** the validation-selected combination remains close to simple/classical
  means; its small validation advantage did not translate into a statistically distinguishable
  test gain. AR(1) has the lowest test RMSE point estimate, but the primary DM comparison does not
  distinguish it from the frozen combination. PatchTST underperforms the primary system on this
  test. No conclusion extends beyond this asset, period, horizon, or fixed configuration.
- **Warning:** PyTorch emitted the expected `enable_nested_tensor` warning when loading the
  transformer. It is preserved in the run record; no fitting warning was emitted by the classical
  test candidates.
- **Reproduction:** `UV_CACHE_DIR=/tmp/solitons-uv-cache MPLCONFIGDIR=/tmp/solitons-mpl uv run
  python -m src.evaluation.final_test` (run once only; test outcomes now exist).

---

## Report revision — descriptive figures and mathematical clarification (maintenance)

- **Objective:** expand `paper/report.tex` into an in-depth paper-style report and produce
  publication-quality descriptive figures from the already saved data and artifacts.
- **Planned figures:** date/split overview and rolling volatility; training return distribution;
  mean forecast RMSE by partition; volatility QLIKE heatmap; PatchTST epoch curves; rolling
  validation/test RMSE; and test DM confidence intervals.
- **Inputs:** processed return data for descriptive figures, EXP-003/004/005/006/007/008/009
  machine-readable artifacts and forecast CSVs. Test scores/forecasts will only be plotted from
  the frozen EXP-009 output; no new test forecast or model selection will be run.
- **Statistical clarification:** report the canonical GARCH(1,1) variance persistence condition
  `alpha+beta<1` separately from the project's recorded `omega+alpha+beta` diagnostic. The
  latter includes a variance intercept and is scale dependent; EXP-004-S1's constraint is
  therefore described as the pre-registered sensitivity rule, not a canonical stationarity
  restriction. Existing outputs and log history remain unchanged.
- **Reproduction:** `UV_CACHE_DIR=/tmp/solitons-uv-cache MPLCONFIGDIR=/tmp/solitons-mpl uv run
  python paper/make_figures.py`; then `XDG_CACHE_HOME=/tmp/solitons-xdg-cache tectonic -X compile
  paper/report.tex --outdir paper`.
- **Status:** registered as documentation/maintenance before figure generation; figures will be
  persisted under `paper/figures/` and their exact source artifacts logged below.

### Report revision — completion record

- **Status:** complete; documentation/maintenance only. No forecast, model, or test evaluation
  was run for this revision.
- **Artifacts:** `paper/report.tex`, `paper/report.pdf`, `paper/make_figures.py`, and seven PNGs:
  `fig1_returns_volatility.png`, `fig2_train_distribution.png`,
  `fig3_rmse_validation_test.png`, `fig4_qlike_by_proxy.png`,
  `fig5_patchtst_learning_curves.png`, `fig6_rolling_rmse.png`, and
  `fig7_test_dm_intervals.png` under `paper/figures/`.
- **Results/procedure:** figure generation read existing processed data and saved experiment
  outputs only. It completed successfully. Tectonic compiled the expanded report to an 18-page
  A4 PDF without warnings. `pdfinfo` and extracted text were checked; the PDF contains the seven
  result figures, equations, experiment tables, interpretation guide, limitations, appendices,
  and reproduction commands.
- **Mathematical clarification:** the report applies the standard GARCH(1,1) finite-unconditional-
  variance condition `alpha+beta<1`. Historical `omega+alpha+beta` values remain transcribed as
  the registered scale-dependent diagnostic; EXP-004-S1 is interpreted as sensitivity to that
  predeclared rule, not as a canonical stationarity-constrained refit (see D-018 and the concept
  note correction).
- **Commands:** `UV_CACHE_DIR=/tmp/solitons-uv-cache MPLCONFIGDIR=/tmp/solitons-mpl uv run
  python paper/make_figures.py` → exit 0; `XDG_CACHE_HOME=/tmp/solitons-xdg-cache tectonic -X
  compile paper/report.tex --outdir paper` → exit 0, no warnings.
- **Interpretation/limits:** figures visualize the already logged outcomes. The report preserves
  the conclusions and limitations of EXP-001…009; in particular, it does not claim a general
  Transformer result from one asset, horizon, and fixed architecture. Sealed-test outcomes were
  plotted from the frozen output only and were not recomputed.


---

## EXP-006 — Fair comparison on identical dates, origins, horizons

- **Status:** REGISTERED (pre-run) on 2026-11-16 (depends on EXP-005 artifact). Code:
  `src/evaluation/compare.py`.
- **Question:** On the *same* 603 validation dates with the *same* one-step protocol, does
  PatchTST add predictive information relative to the classical candidates and baselines?
- **Objective:** assemble a single machine-readable comparison table (classical candidates +
  zero/training-mean baselines + PatchTST), verify date/origin/horizon identity across sources,
  and record per-date loss differentials.
- **Testable hypothesis (H₁):** PatchTST has lower validation RMSE than every classical mean
  candidate. Null (H₀): PatchTST does not — formal inference deferred to EXP-008 (DM); EXP-006
  reports point estimates only and must not be over-read.
- **Concepts:** identical-sample comparison requirement (the research protocol); MSE loss differential
  d_t = (r_t−ŷ¹_t)² − (r_t−ŷ²_t)²; MAE differential analog; fairness = same dates, same
  origins, same horizon, same information set (train-fitted only).
- **Procedure:** load EXP-003 forecast CSV + EXP-005 CSV; assert exact date equality
  (603 dates, same order); compute RMSE/MAE per model; compute d_t of PatchTST vs each
  candidate/baseline; persist.
- **Outputs:** `results/tables/exp006_fair_comparison_<runid>.json` (+ CSV table).
- **Acceptance criteria:** date-identity assertion passes or run aborts; all metrics from the
  same array of actuals; test unread; results appended below.
- **Limitations pre-declared:** point estimates only (no HAC inference — EXP-008); single
  validation window; PatchTST config not tuned (registered as such in EXP-005).


### EXP-006 — Results (appended post-run)

_(pending)_


---

## EXP-SARIMA-01 — Supplementary seasonal mean specification check

- **Status:** Completed exploratory check; not promoted to the primary model.
- **Question:** Does a five-trading-session seasonal ARMA component improve one-step S&P 500
  return forecasts beyond non-seasonal ARIMA candidates?
- **Protocol:** Fit candidates on the frozen 2,815-observation training partition only;
  evaluate one-step forecasts on 603 validation observations; test partition not read.
- **Candidate:** SARIMA(1,0,1)(1,0,1)[5], plus simpler seasonal and non-seasonal comparators.
- **Finding:** SARIMA(1,0,1)(1,0,1)[5] validation RMSE ≈ 1.161696 and MAE ≈ 0.863337;
  non-seasonal ARIMA(1,0,1) RMSE ≈ 1.161739 and MAE ≈ 0.863841. The difference is
  negligible in practical terms.
- **Diagnostics:** After accounting for four AR/MA terms, residual Ljung–Box p-values are
  approximately 2.39e-4 (lag 5), 4.07e-18 (lag 10), and 6.27e-19 (lag 20). ARCH LM(10)
  p ≈ 2.20e-190. The seasonal mean does not adequately whiten residuals or remove ARCH.
- **Decision:** Retain AR-GARCH(1,1)-Student-t as the primary model; report this SARIMA as an
  exploratory comparator only. AICc improvement is not sufficient evidence of adequacy.
- **Artifacts:** `results/sarima/` and `docs/SARIMA_EXPLORATORY_RESULTS.md`.

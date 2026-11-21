# Decision Log (append-only)

Choices, rationale, alternatives, consequences. Corrections are appended, never silently edited.

---

## D-001 — Rolling one-step forecasts computed by manual fixed-parameter recursion

- **Date:** 2026-11-16
- **Decision:** In `src/classical/garch_mean_spec.py`, validation one-step mean/variance forecasts
  are computed by an explicit recursion over the evaluation window using **frozen** training MLE
  parameters: μ_t from the fitted mean formula, σ_t² = ω̂ + α̂ε_{t-1}² + β̂σ_{t-1}², with ε_{t-1} =
  r_{t-1} − μ_{t-1} formed only from *revealed* history. The recursion is initialized from the
  last in-sample fitted σ² and residual.
- **Alternatives:**
  1. `result.forecast(horizon=1, reindex=False)` per origin — requires re-fitting or data
     mutation each step; arch's `forecast` is designed for in-sample/start-index use, not a
     growing external panel.
  2. `model.fix(params)` + repeated forecast — plausible, but the manual recursion is transparent,
     unit-testable, and identical in spirit.
- **Rationale:** the contract fixes the policy "forecast before revealing the target; after it is
  observed it may be used as history" — a manual recursion makes that ordering explicit and lets a
  unit test prove no-future-leakage. Equality with `res.forecast(horizon=1)` at the train-end
  origin is asserted in tests.
- **Consequence:** we own the recursion (bug risk), mitigated by the equivalence test; arch's
  backcast initialization is *not* reproduced for the pre-train window (we start from fitted
  in-sample state — standard practice).

## D-002 — QLIKE volatility proxy = squared return r_t²

- **Date:** 2026-11-16
- **Decision:** Validation variance forecasts are scored with QLIKE using r_t² as the realized
  volatility proxy: QLIKE = mean(r_t²/σ̂_t² + ln σ̂_t²).
- **Alternatives:** |r_t| (equivalent under monotone transforms for ranking only when used inside
  MSE-type losses — not for QLIKE), realized range/high-low (raw file has OHLC but the processed
  target does not carry it into this experiment), EWMA proxy (introduces a tuned decay parameter).
- **Rationale:** QLIKE is loss-consistent for conditional variance even when the proxy is a noisy
  unbiased measure of it (Patton 2011); r_t² is available for every validation date without extra
  data or tuning. Consequence: high proxy noise inflates the *level* of QLIKE but not its
  cross-model ranking; documented as a limitation.

## D-003 — Common effective sample via `hold_back=2` for all four candidates

- **Date:** 2026-11-16
- **Decision:** `hold_back=2` is passed to the `arch_model` constructor (not `fit`) for every
  specification; effective sample = 2,815 − 2 = 2,813 for all candidates; recorded per fit.
- **Alternatives:** arch's default (0) or per-model automatic lag hold-back — would give unequal
  samples and invalidate AIC/BIC comparison.
- **Rationale:** the research protocol forbids comparing information criteria across different effective
  samples; equal samples make AIC/BIC *within this family* comparable (still not across
  implementations, e.g. vs. statsmodels ARIMA).
- **Consequence:** two observations are discarded for every model, including those that don't
  need them — small cost, big comparability gain.

## D-004 — Leave `src/classical/garch.py` untouched; new module for EXP-003

- **Date:** 2026-11-16
- **Decision:** EXP-003 lives in a new file `src/classical/garch_mean_spec.py`; the existing
  `garch.py` (in-sample ARIMA(2,0,0)+GARCH on the *full* series) is not modified.
- **Alternatives:** refactor `garch.py` in place — riskier (it is the provenance of
  `garch_fitted_volatility.csv`) and mixes exploratory in-sample code with a registered
  experiment.
- **Rationale:** minimal modular changes; preserve prior artifacts; full-sample fitting in
  `garch.py` is exploratory-only and must not leak into evaluation — separate module enforces the
  train-only path.
- **Consequence:** some duplication of fit/diagnostic code; acceptable and logged.

## D-005 — Unique run IDs for artifacts; no overwrites

- **Date:** 2026-11-16
- **Decision:** artifacts are named `exp003_garch_mean_spec_<runid>.json` and
  `exp003_validation_forecasts_<runid>.csv`, runid = UTC timestamp + 4 hex chars. Existing files
  (`garch_fitted_volatility.csv`, `validation_arima_candidates.csv`) are never written.
- **Rationale:** the research protocol — never silently overwrite experiment artifacts.
- **Consequence:** disk usage grows per run; the run ID is recorded in the experiment log entry.

## D-006 — Retrospective registration of EXP-000…EXP-002 instead of invented history

- **Date:** 2026-11-16
- **Decision:** prior unlogged experiments are registered retroactively in
  `docs/experiment_log.md` and marked "retrospective registration / run unlogged", with known
  artifacts listed and gaps stated.
- **Alternatives:** (a) pretend they were registered — dishonest; (b) delete/ignore them — loses
  provenance; (c) backfill detailed results not in evidence — fabrication risk.
- **Rationale:** append-only honesty; the project documentation policy forbids rewriting history.
- **Consequence:** three entries carry weaker evidential status than EXP-003 onward.

## D-007 — Persistence > 1 recorded as a finding; no mid-experiment model change

- **Date:** 2026-11-16 (after EXP-003 run)
- **Context:** all four EXP-003 fits returned ω̂+α̂+β̂ ∈ [1.0164, 1.0201] > 1 — the fitted variance
  recursion is slightly non-stationary (near-IGARCH). AIC, diagnostics, and validation metrics
  were otherwise complete and converged.
- **Decision:** do **not** re-fit with an enforced ω+α+β ≤ 1 constraint, do not switch variance
  family, and do not discard the run. Report the value per spec (already in the JSON and the
  experiment log) and carry the issue into EXP-004/EXP-009 robustness analysis.
- **Alternatives:** (a) impose stationarity now and re-run — changes the pre-registered
  configuration after seeing results (garden-of-forking-paths risk); (b) drop the run — hides a
  real diagnostic; (c) silently continue — violates the research protocol.
- **Rationale:** one-step rolling forecasts re-anchor daily, so persistence > 1 does not corrupt
  EXP-003's validation comparisons; only long-horizon extrapolation would be affected, which this
  experiment does not perform. Changing the model post hoc would break pre-registration.
- **Consequence:** EXP-003 stands as run; EXP-004 must include a stationarity-robustness variant
  (e.g. constrained GARCH or E-GARCH) as a pre-declared follow-up.

## D-008 — EXP-004 consumes the frozen EXP-003 artifact instead of refitting

- **Date:** 2026-11-16
- **Decision:** variance forecasts are read from
  `results/forecasts/exp003_validation_forecasts_<runid>.csv` (most recent), never recomputed;
  the source path and EXP-003 run ID are recorded in the EXP-004 JSON.
- **Alternatives:** (a) refit the four models inside EXP-004 — duplicated computation and a real
  risk that numerical drift makes "EXP-004 scores EXP-003's forecasts" false; (b) import
  `rolling_one_step` and re-forecast — same drift risk.
- **Rationale:** the two experiments must score *literally identical numbers*; consuming the
  artifact also exercises the machine-readable pipeline. Consequence: EXP-004 cannot run
  without EXP-003 (dependency recorded in the runbook; loud error if missing).

## D-009 — Proxy selection: r², Parkinson range, EWMA(0.94); |r| rejected; λ fixed by convention

- **Date:** 2026-11-16 (pre-run, in registration)
- **Decision:** P1 = r²; P2 = Parkinson ln(H/L)²/(4ln2) from raw OHLC ×10⁴ into return_pct²;
  P3 = EWMA(0.94) of r² initialized with training variance. λ = 0.94 (RiskMetrics convention,
  declared, never tuned). |r| rejected.
- **Alternatives:** |r| (redundant with r² up to a constant under QLIKE — see concept note);
  Garman–Klass (needs Open too — available, but Parkinson isolates the H/L information with
  fewer terms and is sufficient for a three-proxy design); EWMA λ sweep (would be a *tuned*
  choice on validation — explicitly deferred and pre-declared as requiring registration first).
- **Rationale:** three genuinely different information sources (close-to-close, intraday range,
  smoothed), each with a stated bias; units converted explicitly (the research protocol).
- **Consequence:** conclusions are proxy-conditional by construction — EXP-004 reports all three
  and treats cross-proxy instability as a finding, not a nuisance.

## D-010 — EWMA(0.94) one-step forecast admitted as a first-class baseline (B2)

- **Date:** 2026-11-16 (pre-run)
- **Decision:** score GARCH not only against constant training variance (B1) but against a
  frozen EWMA(0.94) variance *forecast* (B2), recursion initialized with training variance and
  the last training r².
- **Alternatives:** only B1 (weak — would flatter GARCH); EWMA λ tuned on validation (forbidden
  by the research protocol without pre-registration).
- **Rationale:** a baseline that a practitioner can run with zero estimation is the honest
  benchmark for "does GARCH earn its complexity on variance?" B2 proved strong (EXP-004
  results) — exactly why it belongs.
- **Consequence:** any future claim of GARCH variance superiority must beat B2, and must state
  the proxy (B2 wins P1/P3, loses P2).

## D-011 — Fill empty `src/evaluation/metrics.py`; accepted duplication of rmse/mae/qlike

- **Date:** 2026-11-16
- **Decision:** EXP-004 primitives live in the previously-empty `src/evaluation/metrics.py`;
  `src/classical/garch_mean_spec.py` keeps its own internal `rmse`/`mae`/`qlike` (r²-based)
  untouched, with the duplication documented in the metrics module docstring.
- **Alternatives:** refactor `garch_mean_spec.py` to import shared metrics — churn on committed,
  tested EXP-003 code for zero behavioural gain; shared module importing *from* classical
  (wrong dependency direction for a package meant to serve both model families).
- **Rationale:** minimal-change discipline; EXP-003's tests pin its behaviour; the evaluation
  package is the correct long-term home (EXP-006+ will import from it).
- **Consequence:** two qlike implementations exist (`qlike` in garch_mean_spec for r² scoring,
  `qlike_proxy` generalizing to any proxy); a future consolidation entry may merge them once
  EXP-003 is frozen.

## D-012 — Operationalizing the EXP-004-S1 MATERIAL verdict (qualification scope)

- **Date:** 2026-11-16 (after S1 run `20261116T131945Z_8ec9`)
- **Context:** the pre-declared rule fired MATERIAL because the ar1/ar2 QLIKE(r²) ranking
  swapped (near-tie, 1.3e-5 apart), while mean rankings and GARCH-vs-baseline ordering were
  invariant and mean forecasts were unchanged to ≤3e-5.
- **Decision:** carry **only the scoped qualification** into EXP-009 and the report: (1) mean
  conclusion untouched; (2) baseline ordering untouched; (3) ar1-vs-ar2 *variance* superiority
  flagged as persistence-specification-sensitive and never claimed without EXP-008 significance
  tests. Do **not** re-run EXP-003 under the constraint or alter frozen artifacts.
- **Alternatives:** (a) treat MATERIAL as "invalidate everything" — disproportionate, contradicts
  the observed invariance of the substantive conclusions; (b) override the rule as "just a
  tie" — forbidden, the rule was pre-declared and must be honoured as written.
- **Rationale:** the rule's purpose is to force disclosure, not to pick winners; disclosing the
  swap with its magnitude is exactly the honest reading.
- **Consequence:** EXP-009/report include the qualification verbatim; the ar1/ar2 pair is
  reported as statistically indistinguishable pending EXP-008.

## D-013 — EXP-000 gap closed with automated provenance tests (no pipeline rewrite)

- **Date:** 2026-11-16
- **Decision:** close the registered EXP-000 gap by asserting the *existing* artifacts'
  identities in `tests/test_data_provenance.py` (raw Close ≡ processed Close, log_price ≡
  log(Close), return_pct ≡ 100·Δlog(Close), OHLC sanity, weekday-only monotonic unique dates,
  registered row counts), rather than re-deriving the processed file from raw.
- **Alternatives:** rebuild processed from raw (risk of overwriting a committed artifact —
  forbidden without approval); leave the gap open.
- **Rationale:** the identities were verified against the source data (max diff 8.9e-16); tests freeze that
  fact against future drift.
- **Consequence:** EXP-000 promoted to DONE-in-substance (provenance identities); the
  holiday-calendar check remains limited to weekday assertions (full exchange-calendar
  validation would need an external calendar — noted as residual gap).

## D-014 — Preserve chronological order in PatchTST window training

- **Date:** 2026-11-16, before EXP-005 execution.
- **Decision:** DataLoader batches retain chronological window order (`shuffle=False`) to satisfy
  the research protocol's no-random-shuffling rule. Fix PatchTST's temporal Unfold to use a 2-D kernel with
  singleton height; construct model inputs only through the validation endpoint.
- **Alternatives:** shuffle windows for conventional minibatch optimization; train with a library
  default patch operation. Both violate either the explicit chronology rule or model shape.
- **Rationale:** chronology is part of the research protocol, and model input dates must be
  auditable. The focused tests first exposed the kernel and partition-boundary defects.
- **Consequence:** training uses an ordered SGD path; runtime is slower than shuffled batches.
  Architecture and registered seed/hyperparameters were unchanged.

## D-015 — Freeze the validation-selected mean combination for EXP-009

- **Date:** 2026-11-16, after EXP-006…008 validation analysis and before test access.
- **Decision:** primary test forecast is the convex zero/training-mean combination with frozen
  weights 0.4999432456972872 and 0.5000567543027128. Also retain training-mean as the selected
  single-model reference and PatchTST/classical candidates as predeclared secondary comparisons.
- **Alternatives:** pick PatchTST despite worse validation; pick an AR/GARCH mean by AIC; ignore
  the minuscule combination gain and freeze zero only.
- **Rationale:** validation MSE is the declared selection criterion; simple baselines tie for
  best single forecast, and the convex pair is the lowest validation RMSE. The manifest exposes
  the tiny margin and selection optimism so it cannot be mistaken for robust evidence.
- **Consequence:** no weights/model may change after test opens; all test analysis is confirmatory
  and one-pass. The selected combination is essentially a constant forecast near 0.0221.

## D-016 — Correct the finite-sample DM adjustment and disclose failed attempts

- **Date:** 2026-11-16.
- **Decision:** use the h=1 Harvey–Leybourne–Newbold multiplier `sqrt((T-1)/T)` (not 1) and
  t reference df T−1; report previous failed execution and warnings in the experiment log.
- **Rationale:** the registered expression was written incorrectly; correcting it before test
  evaluation preserves the stated finite-sample method. Validation DM was rerun to persist the
  corrected intervals.
- **Consequence:** validation DM significance values change slightly; test code and report use
  the corrected formula. No model selection is altered by this correction.

## D-017 — Keep forecast-encompassing logic in its dedicated evaluation module

- **Date:** 2026-11-16, before sealed test.
- **Decision:** implement the regression helper in `src/evaluation/encompassing.py` and call it
  from the combination runner, after initially colocating the code in `combination.py`.
- **Rationale:** the repository already has a dedicated module and separation makes the
  rank-identification rule independently testable.
- **Consequence:** no estimator, HAC lag, or reported validation coefficient changed. The
  deterministic EXP-007 rerun produced the same best pair/weights; the later artifact is the
  source named by the freeze manifest.

## D-018 — Distinguish canonical GARCH persistence from the registered sum including omega

- **Date:** 2026-11-17, during report quality review.
- **Context:** prior experiment records called `omega + alpha + beta` persistence and described
  values above one as a variance nonstationarity signal. In a GARCH(1,1) variance recursion,
  omega has variance units while alpha and beta are dimensionless coefficients.
- **Decision:** preserve all historical estimates and the EXP-004-S1 registered constraint, but
  the report must identify `alpha + beta < 1` as the canonical finite-unconditional-variance
  condition (under standard assumptions) and explain that `omega + alpha + beta` is a
  scale-dependent project diagnostic. Treat EXP-004-S1 as a sensitivity to its registered rule,
  not as a conventional stationarity-constrained refit.
- **Alternatives:** silently edit the experiment logs/results; retain the imprecise stationarity
  claim; rerun or modify model selection after the sealed test.
- **Rationale:** dimensional consistency and correct econometric interpretation. The correction
  changes interpretation, not fitted values, validation/test forecasts, or the frozen system.
- **Consequence:** persistence-based qualifications in the report are narrower; see the appended
  correction in the experiment log and the concept note.

# Concept: Volatility proxies and proxy-based QLIKE

Used by: EXP-004. Companion to `docs/experiment_log.md` EXP-004,
`src/evaluation/metrics.py`, and `docs/concepts/garch_mean_specifications.md`.

## Plain language

A variance forecast σ̂²_t cannot be scored directly — the true variance is never observed.
We must compare it to something that *stands in for* it: a **proxy**. Three properties matter:

1. **Unbiasedness (approximate):** E[proxy | F] ≈ σ², so forecasts that are closer to the truth
   score better.
2. **Noise level:** the noisier the proxy, the less power we have to rank models (and the more
   inflated the absolute loss).
3. **Independence from the forecast:** a proxy built from the same machinery as a candidate
   forecast quietly favours that candidate ("proxy alignment"). This is the trap EXP-004's P3
   partially falls into with the EWMA baseline — recorded as a caveat, not hidden.

**QLIKE** (Quasi-LIKelihood) is the loss used for variance forecasts:

    L(σ̂²; q) = mean( q_t / σ̂²_t + ln σ̂²_t )

It is the expected-loss-minimizing loss for conditional variance even when q_t is a *noisy,
unbiased* measure of σ²_t (Patton, 2011) — unlike MSE or MAE between σ̂ and a noisy proxy, which
are biased by proxy noise. Written against Patton's exact form it differs only by terms free of
σ̂², so cross-model rankings are identical. QLIKE's log term makes it one-sided in practice:
*under-stating* variance is punished more than over-stating it.

**Units matter:** QLIKE mixes the proxy with σ̂² inside both `q/σ²` (scale-invariant only if both
share units) and `ln σ̂²` (unit-*dependent* in level). All EXP-004 quantities are in
**return_pct²** (percentage-point²) units — mixing decimal² and pct² would silently distort
every comparison (the research protocol).

Also note: **QLIKE levels are comparable only within a proxy.** Different proxies have
different scales and noise, shifting the whole column up or down; only the *ranking* inside a
column carries information.

## The three pre-declared proxies (EXP-004)

| # | Proxy | Formula | Units fix | Why |
|---|---|---|---|---|
| P1 | Squared return | q_t = r_t² | none (already pct²) | standard, unbiased up to scale; but very noisy (a single day's draw) |
| P2 | Parkinson range | q_t = ln(H_t/L_t)² / (4 ln 2) | × 10⁴ (decimal² → pct²) | uses intraday High/Low ⇒ far lower sampling noise; built from **prices, not returns** ⇒ independent of the return-based forecast recursion (Brownian-motion efficiency argument; biased by jumps/overnight gaps) |
| P3 | EWMA(0.94) of r² | q_t = 0.94·q_{t-1} + 0.06·r_t² | none | smooths out single-day noise to approximate the latent variance *path*; λ = 0.94 is the RiskMetrics convention, fixed — never tuned on validation. Uses the **current** r_t ⇒ a measurement, available only ex post |

Why not |r|? |r_t| is (approximately) a proxy for **σ**, not σ²; squaring it returns r² up to a
constant, making it redundant with P1 under QLIKE's `q/σ²` term — rejected as a third proxy in
favour of genuinely different information (intraday range, smoothing).

## The two forecast baselines (not proxies!)

- **B1 constant variance:** σ̂² = training sample variance (ddof = 1), frozen. Deliberately
  weak — it answers "does *any* dynamics beat a constant?"
- **B2 EWMA(0.94) forecast:** σ̂²_t = 0.94·σ̂²_{t-1} + 0.06·r²_{t-1}, first value
  = 0.94·train_var + 0.06·r²_{train,last}. Uses only information available at the origin — a
  genuine one-step forecast (leakage-tested). This is the *serious* competitor: same recursion
  family as GARCH with one fixed constant and no estimation.

The proxy/forecast distinction is the crux: **P3 may use r_t²; B2 may not.** Their formulas look
almost identical — one index differs — and that index is the difference between a measurement
and a forecast.

## References

- Patton, A. J. (2011). "Volatility forecast comparison using imperfect volatility proxies."
  *Journal of Econometrics* 160(1).
- Parkinson, M. H. (1980). "The extreme value method for estimating the variance of the rate of
  return." *Journal of Business* 53(1).
- Zumbach, G. (2000). "The RiskMetrics approach to risk measurement" (EWMA λ = 0.94 convention).

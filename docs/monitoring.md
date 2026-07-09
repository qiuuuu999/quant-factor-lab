# Factor Monitoring

This document defines `quantlab.monitor` — IC decay detection and market-regime
fit — and the conventions it follows.

---

## Why factors decay

A factor's historical IC is a backward-looking average; it is not a guarantee
that the same edge is live today. Three distinct mechanisms erode a working
factor, and they call for different responses:

* **Crowding.** Once a signal is well known and widely traded, the capital
  chasing it bids up the names it favors *in advance* of the rebalance that
  would have captured the edge, and compresses the return spread it used to
  earn. This tends to show up as a gradual IC decline over years, not a sharp
  break — the closest thing to disproving "the strategy stopped working" is
  ruling out mechanism 3 (below) first.
* **Arbitrage / structural elimination.** If the factor captures a genuine
  mispricing (as opposed to a real risk premium), the first movers who
  identify it are, in aggregate, self-defeating: trading on it moves prices
  toward fair value and closes the gap that produced the IC in the first
  place. Unlike crowding this can be fast — once enough capital arrives, the
  edge can vanish over quarters rather than years, which is exactly the
  sharp, single-break pattern the CUSUM test below is built to catch.
* **Structural / regime change.** The factor's edge was real and is not
  arbitraged away, but the *conditions* it depended on changed — a shift in
  monetary regime, market microstructure (e.g. decimalization, HFT
  liquidity), or the composition of the tradable universe. This is why
  regime-conditioned IC (the second half of this module) matters alongside a
  raw decay test: a factor that looks broken in the full-sample average may
  simply be out of its regime, and due to *return*, not retirement, when
  conditions revert.

`quantlab.monitor` cannot distinguish these mechanisms from IC alone — that
requires domain judgment (AUM estimates of similar strategies, transaction-cost
trends, macro regime narratives). What it *can* do reliably is flag **that**
something changed and **when**, and show whether the factor's performance is
regime-dependent — the two inputs a researcher needs to start that diagnosis.

---

## IC decay detection

`quantlab.monitor.decay`

### Rolling IC

`rolling_ic(ic, window=36, min_periods=None)` is the trailing mean of the raw
per-period IC series over `window` periods (36 months / 3 years by default —
long enough to smooth out the month-to-month noise in a single-factor IC,
short enough to still move within one market cycle). This is the "current
form" number a health report leads with: it answers *how has this factor
performed lately*, as opposed to the full-sample average which can hide a
recent decline behind a long earlier track record.

### CUSUM change-point test

`cusum_test(x, confidence=0.95)` is a retrospective (single, unknown-time)
test for a break in the mean of a series. For `x_1, ..., x_n` with sample mean
`x̄` and sample standard deviation `s`, the cumulative sum of mean-centred
observations

```
S_k = sum_{i=1}^{k} (x_i - x̄),   k = 1..n
```

drifts away from zero around a true (unknown) change point and returns toward
zero elsewhere, so the `k` that maximises `|S_k|` is the maximum-likelihood
estimate of a single break point (Page, 1954). Normalized,

```
statistic = max_k |S_k| / (s * sqrt(n))
```

converges, under the "no break" null, to the supremum of a Brownian bridge —
the same asymptotic distribution used for the two-sample Kolmogorov-Smirnov
test — giving parameter-free critical values:

| Confidence | Critical value |
| --- | --- |
| 90% | 1.22 |
| 95% | 1.36 |
| 99% | 1.63 |

If `statistic` exceeds the critical value, the series has a statistically
significant break at the estimated change point; comparing the mean before vs.
after that point shows whether it is a **decay** (mean drops) or an
improvement (mean rises) — only the former is flagged as `is_decay`. This test
makes no distributional assumption on the IC series itself beyond finite
variance, and — being retrospective, not sequential — is re-run over the full
history collected so far each time a health report is refreshed, not
evaluated online as new data streams in.

### Factor health report

`factor_health_report(name, ic, window=36, confidence=0.95)` packages both
into a `FactorHealthReport`: current rolling IC vs. historical mean IC,
whether a decay alert is triggered (`decay_alert`), and the alert date
(`alert_date`, the CUSUM-estimated break point, only set when `decay_alert` is
true).

**Correctness check.** `tests/test_monitor.py` builds a synthetic IC series
with a known engineered break (60 periods of positive-mean IC, then 60 periods
of negative-mean IC) and asserts the CUSUM test triggers, correctly flags it
as decay, and locates the change point within a handful of periods of the true
break; a stationary (no-break) synthetic series and an *improving* (negative
-> positive) break are checked as negative controls.

---

## Market-regime detection

`quantlab.monitor.regime`

### Classification

`classify_regime(prices, vol_window=21, trend_window=200)` labels each
trading day of a benchmark series (SPY) into one of four regimes, crossing two
independent signals:

* **Volatility level** — 21-trading-day rolling realised (annualised)
  volatility, split into *low* / *high* at its own **sample median**. Using
  the median (rather than a fixed threshold like 15% or 20% annualised) is a
  deliberate choice: it makes the classification self-referential to the
  benchmark's own history, so it adapts across very different macro
  volatility eras (e.g. the low-rate 2010s vs. 2022+) without a hand-tuned
  cutoff, at the cost of only being interpretable as *relative* (this
  benchmark's calm vs. turbulent periods), not an absolute vol level.
* **Trend direction** — price above / below its 200-day moving average, the
  standard "risk-on / risk-off" trend filter.

giving `low_vol_up`, `low_vol_down`, `high_vol_up`, `high_vol_down`. The first
`max(vol_window, trend_window)` days of the input have no rolling window and
are dropped rather than guessed at.

`regime_as_of(regime, dates)` forward-fills the daily regime labels onto an
arbitrary (e.g. monthly rebalance) calendar — the regime label active at a
rebalance date is whatever was last known at or before that date.

### Factor-regime fit matrix

`factor_regime_matrix(ic_by_factor, regime_by_date)` cross-tabulates each
factor's per-period IC by the regime active on that date, producing a
factor x regime mean-IC table (`RegimeICMatrix.mean_ic`), each factor's best
and worst regime (`.best_regime` / `.worst_regime`), and the observation count
backing each cell (`.counts`, since an empirically-derived regime split is not
guaranteed to be even — see the note below).

**Correctness check.** `tests/test_monitor.py` builds synthetic price paths
with four consecutive, deliberately engineered segments (one per regime, each
with a known drift/volatility combination) and asserts `classify_regime`
recovers the correct label for at least 90% of days deep into each segment
(allowing for the 200-day trend-filter's lag at each transition); a synthetic
factor with an engineered regime-dependent IC checks that
`factor_regime_matrix` recovers the correct best/worst regime.

**Real-data note.** Over SPY's 2013-2025 history, `low_vol_down` is rare by
construction: the sharp drawdowns in this sample (2020, 2022) were also
high-volatility episodes, so a "calm decline" regime is nearly absent
(`reports/monitor/summary.md` shows `NaN` cells for it in every factor —
too few rebalance-date observations to average, not a bug). This is itself a
useful finding: genuinely low-volatility bear markets are uncommon in equity
history, which is worth remembering when interpreting any regime-conditioned
backtest.

---

## Demo: four price factors, 2015-2025

`scripts/run_factor_monitoring.py` runs both pieces on the same four factors
scored in `scripts/run_factor_evaluation.py` (12-1 momentum, low volatility,
1-month reversal, Amihud illiquidity) over the point-in-time S&P 500,
2015-2025, and writes, under `reports/monitor/`:

- `<factor>_health.png` — per-period IC, rolling mean, CUSUM alert marker (if
  triggered)
- `regime_heatmap.png` — factor x regime mean-IC heatmap
- `summary.md` — both tables

None of the four factors trip a 95%-confidence CUSUM decay alert over this
window (the closest is 1-month reversal, statistic 1.35 vs. critical 1.36) —
consistent with `docs/factors.md`'s IC-based evaluation of these factors as
modest but not obviously broken over 2015-2025. The regime fit matrix is more
informative than the raw decay test here: momentum and low-volatility are both
best in calm uptrends and worst in high-volatility regimes (procyclical), while
reversal and illiquidity favor high-volatility regimes (their edge — short-term
overreaction and forced-liquidation illiquidity premia — is mechanically a
volatility-driven effect).

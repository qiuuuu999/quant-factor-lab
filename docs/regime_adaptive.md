# Regime-Adaptive Factor Weighting

This document defines `quantlab.portfolio.regime_adaptive` — dynamic,
regime-conditioned factor combination — and reports the 2017-2025 backtest
result honestly, including where it does *not* beat a simpler baseline.

---

## Motivation

`docs/monitoring.md` and `reports/monitor/regime_heatmap.png` establish, in
retrospect, that each of the platform's four price factors has a
regime-dependent edge: momentum and low-volatility do best in calm uptrends
and worst in high-volatility regimes; reversal and illiquidity do the
opposite. A static, always-equal combination of the four factors averages
across regimes and by construction cannot exploit this — in a
momentum-favoring regime it is diluted by three factors that (on the
historical record) are dead weight or worse; in a reversal-favoring regime,
the reverse.

The question this module and backtest answer: **does turning that
retrospective observation into a live, point-in-time allocation rule actually
improve the resulting portfolio** — not "would it have, with hindsight" (a
much easier and less meaningful bar) but "would it have, using only
information available at each decision"?

---

## Design

`quantlab.portfolio.regime_adaptive`

### Point-in-time discipline

Two look-ahead traps had to be closed relative to the retrospective analysis
in `quantlab.monitor`:

1. **The regime label.** `quantlab.monitor.regime.classify_regime`'s default
   mode splits volatility into low/high at the *full-sample* median — correct
   for a retrospective report, wrong for a 2018 trading decision that cannot
   know the median volatility of 2015-2025. `classify_regime` gained an
   `expanding: bool = False` parameter (default preserves the existing
   retrospective behaviour used by `scripts/run_factor_monitoring.py`);
   `pit_regime_by_date` always calls it with `expanding=True`, so day `t`'s
   volatility-regime label depends only on prices up to and including `t`.
2. **The factor-regime fit table.** `quantlab.monitor.regime.factor_regime_matrix`
   averages IC over the *entire* sample. `regime_conditioned_factor_weights`
   instead computes, for a decision at `as_of`, each factor's mean IC using
   only observations dated **strictly before** `as_of`, restricted to periods
   that were themselves in the current regime. The strict inequality matters:
   an IC observation at formation date `d` requires the forward return from
   `d` to the *next* formation date, which only realizes at that next date's
   close — so by decision time `as_of`, everything dated `< as_of` is known
   and `as_of`'s own IC is not yet computable, only usable one period later.

Both properties are checked directly in `tests/test_regime_adaptive.py`, not
just asserted in prose: `test_expanding_classification_unaffected_by_future_data`
confirms extending the SPY series into the future leaves past labels
unchanged, and `test_regime_conditioned_weights_ignore_future_ic_observations`
/ `test_regime_conditioned_weights_exclude_current_period_own_ic` confirm
corrupting IC data at or after the decision date does not move the computed
weights.

### Weighting rule

For each factor, `regime_conditioned_factor_weights` computes the
regime-conditioned, strictly-prior mean IC, then:

* A factor with fewer than `min_obs` regime-matched prior observations is
  **excluded** for that rebalance — not assumed to have zero edge, just
  insufficient evidence yet.
* A surviving factor gets a **signed** weight equal to its mean IC. Positive
  IC → used as-is; negative IC → either inverted (`negative_handling="invert"`,
  the default — a factor "working backwards" in a regime still carries
  information) or dropped (`negative_handling="zero"`, more conservative).
* Weights are normalized so `sum(|weight|) == 1`.
* If every factor is excluded (early warm-up, or a regime with too little
  accumulated history), the rule falls back to an equal, uninverted
  combination rather than an empty composite score.

`composite_score(exposures, weights)` combines that rebalance's z-scored
exposures into one number per name; the top 20% by composite score is
selected and equal-weighted, identical to every other selection rule already
in this platform (`scripts/run_momentum_backtest.py`,
`scripts/run_portfolio_comparison.py`).

`build_static_multifactor_weights` is the fixed control group: the same
selection mechanics, but with an always-equal (`1/4` each, unsigned)
combination — no regime awareness, no IC history — isolating what
regime-conditioning specifically contributes versus simply combining several
factors.

---

## Backtest setup

`scripts/run_regime_adaptive_backtest.py`, four strategies on one tearsheet:

| | Regime-Adaptive | Static Multi-Factor | Pure Momentum | SPY |
| --- | --- | --- | --- | --- |
| Factors | all 4, regime-conditioned-IC-weighted | all 4, fixed 1/4 each | momentum only | — |
| Selection | top 20% composite score | top 20% composite score | top 20% momentum | buy & hold |
| Weighting | equal-weight selected names | equal-weight selected names | equal-weight selected names | market-cap |

* **Warm-up**: factor IC and the SPY regime series are built starting
  2015-01, but no capital is deployed until 2017-01 — by the first live
  rebalance, `regime_conditioned_factor_weights` already has two full years
  of real (not synthetic) regime-conditioned IC history to condition on.
* **Live window**: 2017-01 – 2025-12, identical for all four strategies.
* **Execution**: signal on month-end close, fill at next session's open
  (`execution_lag=1`), halted opens rolled forward — the same no-look-ahead
  execution convention as every other backtest on this platform (see
  `docs/execution_timing.md`).
* **Parameters used, not tuned**: `negative_handling="invert"`, `min_obs=6`.
  These were chosen for a defensible prior (six regime-matched observations
  is a low but non-trivial bar; inverting rather than dropping keeps a
  negative-IC factor's information rather than discarding it) and were **not**
  swept or selected by backtest performance — doing so on a single 9-year
  historical draw would be exactly the kind of in-sample overfitting this
  platform's PIT discipline is designed to avoid elsewhere.

---

## Results, 2017-2025

| Metric | Regime-Adaptive | Static Multi-Factor | Pure Momentum | SPY |
| --- | --- | --- | --- | --- |
| Total Return | 174.65% | 70.45% | 163.39% | 249.70% |
| CAGR | 11.92% | 6.12% | 11.40% | 14.97% |
| Annual Volatility | 24.80% | 19.46% | 20.45% | 18.48% |
| Sharpe Ratio | 0.58 | 0.40 | 0.63 | 0.85 |
| Max Drawdown | -49.30% | -42.52% | -37.73% | -33.72% |
| Calmar Ratio | 0.24 | 0.14 | 0.30 | 0.44 |
| Monthly Win Rate | 65.42% | 57.94% | 64.49% | 71.03% |
| Annual Turnover | 14.02x | 10.36x | 5.45x | 0.00x |

Figures and the full table: `reports/regime_adaptive/equity_curve.png`,
`comparison_table.png`, `summary.md`; the regime-adaptive sleeve's per-period
factor-weight history (how the mix actually shifted) is in
`regime_adaptive_factor_weights.csv`.

## Analysis — honest, not cherry-picked

**Regime-conditioning clearly beats the naive static combination.**
Regime-Adaptive roughly 2.5x'd Static Multi-Factor's total return (174.65%
vs. 70.45%), with a materially better Sharpe (0.58 vs. 0.40) and Calmar (0.24
vs. 0.14). This is the comparison that isolates the mechanism this module
adds — conditioning the factor mix on the current regime, instead of a fixed
1/4-1/4-1/4-1/4 blend — and it supports the core hypothesis: the
retrospective factor-regime fit in `reports/monitor/regime_heatmap.png` is
not just a hindsight artifact, using it point-in-time added real value over
the static alternative.

**But it does not beat the simplest baseline.** Pure Momentum — one factor,
no regime machinery at all — posted a *higher* Sharpe (0.63 vs. 0.58) and a
much shallower max drawdown (-37.73% vs. -49.30%) than Regime-Adaptive, on a
similar total return. All the added complexity — three extra factors, a
point-in-time regime classifier, an expanding-window IC history, sign
inversion — did not clear the bar of beating the platform's original,
simplest strategy on a risk-adjusted basis, in this one historical sample.
This is the result that matters most and the one worth stating plainly rather
than leading with the more flattering static-multi-factor comparison above.

**All four strategies badly lag SPY.** Every constructed strategy trails
buy-and-hold SPY on every metric except turnover, by a wide margin (Sharpe
0.85 for SPY vs. 0.40-0.63 for the constructed strategies). This is
consistent with, and not a surprising strategy-specific failure given, the
well-documented mega-cap concentration of the 2017-2025 market (particularly
2020 onward): a small number of mega-cap names drove a large share of
cap-weighted index return, and any equal-weight, broad-selection long-only
strategy — factor-based or not — fights that concentration structurally,
independent of how good the underlying factor signal is. This is a property
of the comparison period and universe, not a defect specific to
regime-adaptive weighting; it is included here for completeness rather than
omitted because it is unflattering.

**Turnover is the visible cost of the added machinery.** Regime-Adaptive
trades at 14.02x annualized versus Static's 10.36x and Pure Momentum's 5.45x.
The mechanism is direct: `negative_handling="invert"` means a factor's
contribution to the composite score can flip sign entirely when its
regime-conditioned IC crosses zero, which can rotate a meaningful share of
the selected names in a single rebalance — a structurally higher-turnover
design than a static blend, let alone a single, slower-moving factor. The
reported metrics already include the platform's standard cost model
(commission + slippage), so this turnover cost is *inside* the Sharpe/Calmar
numbers above, not an unaccounted-for caveat on top of them.

## Conclusion

Regime-conditioning the factor mix, done with real point-in-time discipline,
measurably improves on the naive static-multi-factor alternative — that part
of the hypothesis holds up. It does not, in this single 2017-2025 sample,
improve on the platform's simplest single-factor strategy once turnover and
costs are accounted for, and none of the constructed long-only strategies
kept pace with the cap-weighted benchmark over this concentration-heavy
period. The honest takeaway is that regime-awareness is a genuine
improvement *relative to how you'd otherwise combine several factors*, not a
free upgrade relative to not bothering with multi-factor combination at all
— the extra moving parts (four factors, an inversion rule, a turnover-heavy
composite score) have to earn their cost, and in this sample they did not,
against the simplest available baseline.

### What would be worth trying next (not implemented here, to avoid
over-fitting this single backtest)

* A turnover penalty or hysteresis band on the composite score, so a factor's
  weight does not flip sign on a marginal regime-conditioned IC crossing —
  directly targeting the turnover gap versus Pure Momentum identified above.
* Testing `negative_handling="zero"` head-to-head, which would mechanically
  reduce turnover (dropped factors don't need to flip) at the cost of
  discarding information from inverted factors.
* A longer or different historical window (this platform's price history
  starts 2014; a genuinely different regime mix, e.g. one with a real
  `low_vol_down` episode — see `docs/monitoring.md`'s note that this regime
  is nearly absent in the 2013-2025 SPY sample — would be a meaningfully
  different test, not achievable by re-fitting parameters on the same data).

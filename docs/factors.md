# Factor Library

This document defines the factors implemented in `quantlab.factors` and the
conventions they follow.

## Framework and point-in-time contract

Every factor subclasses `quantlab.factors.base.Factor` and implements a single
method:

```python
def _compute(self, prices, universe, as_of_date) -> pd.Series: ...
```

Callers invoke the public template method `compute(prices, universe, as_of_date)`,
which returns a factor value for **every** universe member (`NaN` where the
factor is undefined) indexed by ticker.

**No look-ahead bias.** A factor value for `as_of_date` may use only data
available on or before `as_of_date`. The base class enforces this in two layers:

1. **Interception** — if `prices` contains any row dated after `as_of_date`,
   `compute` raises `LookaheadBiasError`.
2. **Defense in depth** — the frame passed to `_compute` is sliced to
   `date <= as_of_date`, so a subclass physically cannot read future rows.

Input `prices` is the long frame from `quantlab.data.prices.get_prices`
(`date`, `ticker`, `adj_close`, …). Prices are **adjusted** closes, so splits and
dividends do not create spurious returns.

---

## Momentum (12-1)

`quantlab.factors.momentum.MomentumFactor`

### Definition

Cross-sectional price momentum: the cumulative total return over the past
`lookback` months, **excluding** the most recent `skip` month(s). With the
defaults (`lookback=12`, `skip=1`) this is the canonical **12-1 momentum**:

$$
\text{mom}_{12\text{-}1}(t) = \frac{P^{\text{adj}}_{t-1\,\text{month}}}{P^{\text{adj}}_{t-12\,\text{months}}} - 1
$$

where $P^{\text{adj}}$ is the month-end adjusted close.

### Why skip the most recent month?

The one-month gap removes the well-documented **short-term reversal** effect
(Jegadeesh 1990; Lehmann 1990): stocks tend to mean-revert over horizons of a
few weeks due to microstructure/liquidity effects. Skipping the last month
isolates the medium-term *continuation* (momentum) signal from that short-term
noise. The 12-1 window therefore covers **11 months** of return (from `t-12` to
`t-1`).

### Implementation notes

- Daily adjusted closes are resampled to **month-end** (`resample("ME").last()`)
  and only observations up to `as_of_date` are used.
- A stock needs at least `lookback + 1` month-end observations up to the
  formation date; otherwise its factor value is `NaN` (recent IPOs, delisted
  names with short history). This is handled gracefully — no exception.
- Worked example: a stock returning exactly **+1% every month** has a 12-1
  momentum of $(1.01)^{11} - 1 \approx 11.57\%$. This exact identity is asserted
  in the unit tests (`tests/test_factors.py`).

### Reference

> Jegadeesh, N., & Titman, S. (1993). *Returns to Buying Winners and Selling
> Losers: Implications for Stock Market Efficiency.* The Journal of Finance,
> 48(1), 65–91. https://doi.org/10.1111/j.1540-6261.1993.tb04702.x

Related:

> Jegadeesh, N. (1990). *Evidence of Predictable Behavior of Security Returns.*
> The Journal of Finance, 45(3), 881–898. (short-term reversal — motivates the
> one-month skip)

---

## Low volatility

`quantlab.factors.low_volatility.LowVolatilityFactor`

### Definition

The **negative** of trailing daily-return volatility, so that a *higher* factor
value means *lower* risk — and, per the anomaly, *higher* risk-adjusted return:

$$
\text{low\_vol}(t) = -\;\operatorname{stdev}\!\big(r_d : d \in \text{last } w \text{ trading days} \le t\big)
$$

where $r_d$ is the daily total return from **adjusted** closes and $w$ is the
window (default `252` ≈ one year).

### Why negate?

Ang, Hodrick, Xing & Zhang (2006) document the **low-volatility anomaly**:
high-volatility stocks earn *lower* subsequent returns than the CAPM predicts.
Negating the volatility keeps the library's convention that a higher factor value
maps to the long leg. We use *total* return volatility as the simplest price-only
proxy for the idiosyncratic-volatility measure Ang et al. emphasise; the two are
highly correlated in the cross-section.

### Implementation notes

- Daily adjusted-close returns over the trailing `window` trading days; sample
  standard deviation (`ddof=1`).
- Fewer than `min_periods` (default `window // 2`) returns in the window ⇒ `NaN`.

### Reference

> Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). *The Cross-Section of
> Volatility and Expected Returns.* The Journal of Finance, 61(1), 259–299.
> https://doi.org/10.1111/j.1540-6261.2006.00836.x

---

## Short-term reversal (1-month)

`quantlab.factors.reversal.ShortTermReversalFactor`

### Definition

The **negative** of the trailing one-month return (so recent *losers* score high):

$$
\text{reversal}(t) = -\left(\frac{P^{\text{adj}}_{t}}{P^{\text{adj}}_{t-1\,\text{month}}} - 1\right)
$$

from month-end adjusted closes.

### Why negate?

Jegadeesh (1990) shows a stock's most-recent-month return is *negatively*
related to its next-month return — a liquidity/microstructure mean-reversion.
Negating the past return puts last month's losers on the long leg. This is the
exact effect that **12-1 momentum deliberately skips** with its one-month gap, so
reversal and momentum are near-orthogonal by construction.

### Implementation notes

- Needs at least `lookback_months + 1` month-end observations ⇒ else `NaN`.
- Highest turnover of the four factors (the ranking reshuffles every month); the
  evaluation report's rank-autocorrelation quantifies this.

### Reference

> Jegadeesh, N. (1990). *Evidence of Predictable Behavior of Security Returns.*
> The Journal of Finance, 45(3), 881–898.
> https://doi.org/10.1111/j.1540-6261.1990.tb05110.x

---

## Liquidity — Amihud illiquidity

`quantlab.factors.liquidity.AmihudIlliquidityFactor`

### Definition

Amihud's **ILLIQ** — the average daily ratio of absolute return to dollar volume
(price impact per dollar traded):

$$
\text{ILLIQ}(t) = \operatorname{mean}\!\left(\frac{|r_d|}{P_d \cdot V_d} : d \in \text{last } w \text{ days} \le t\right)\times \text{scale}
$$

where $r_d$ is the daily adjusted-close return, $P_d \cdot V_d$ is the day's
**dollar volume** (raw close × raw share volume), $w$ is the window (default
`21` ≈ one month), and `scale` (default `1e6`, as in Amihud 2002) only rescales
for readability — being monotone it changes no rank, IC, or decile result.

### Why *not* negate?

Illiquid stocks command an **illiquidity premium** — higher expected return to
compensate for higher trading cost — so a higher ILLIQ *already* corresponds to
the long leg; no sign flip is needed.

### Implementation notes

- Requires raw `close` and `volume` columns (dollar volume), not just
  `adj_close`; raises `ValueError` if they are absent.
- Zero-dollar-volume days are dropped from the average; fewer than `min_periods`
  (default `window // 2`) valid days ⇒ `NaN`.
- Note the S&P 500 is a large-cap universe, so the illiquidity *dispersion*
  available here is small versus a full-market universe.

### Reference

> Amihud, Y. (2002). *Illiquidity and stock returns: cross-section and
> time-series effects.* Journal of Financial Markets, 5(1), 31–56.
> https://doi.org/10.1016/S1386-4181(01)00024-6

---

## Factor evaluation

`quantlab.factors.evaluation`

Before a factor earns a full backtest, this toolkit answers four questions from a
point-in-time factor panel (`build_factor_panel` produces aligned `date × ticker`
factor-value and forward-return frames):

| Analytic | Function | What it measures |
|----------|----------|------------------|
| **Information Coefficient** | `information_coefficient` → `ICResult` | Per-period Spearman corr(factor, next-period return). Mean = strength/sign; **ICIR** = mean/std = consistency; t-stat = ICIR·√n; hit-rate = fraction of periods with IC>0. |
| **Quantile returns** | `decile_returns` → `DecileResult` | Equal-weight forward return of each of `n` factor buckets, rebuilt each period. A good factor is **monotone** across buckets with a positive top−bottom (long/short) spread. |
| **Rank autocorrelation** | `rank_autocorrelation` | Period-over-period Spearman corr of the factor's own ranks — a **turnover** proxy (near 1 ⇒ stable/cheap; near 0 ⇒ reshuffled/expensive). |
| **Factor correlation** | `factor_correlation` | Average cross-sectional rank correlation *between* factors — how much unique information each adds. |

**Point-in-time and honest about what it is.** Factor values pass through
`Factor.compute` with the look-ahead guard on; forward returns look *strictly
forward* (formation close → next-formation close) and are never part of the
signal. All analytics are rank-based, hence invariant to
winsorization/standardisation. This is a **signal-quality diagnostic, not a
tradable P&L** — the backtest engine (t+1 open fills, costs) remains the tradable
measure.

Run `scripts/run_factor_evaluation.py` to score all four factors over 2015-2025
and write figures + tables to `reports/factor_eval/`. Correctness of the IC and
quantile logic is asserted against constructed panels in `tests/test_evaluation.py`.

### Results, 2015-2025 (S&P 500, point-in-time)

Over this large-cap US decade **all four classic price factors were weak-to-negative**
— mean ICs within ±0.01 of zero, t-stats insignificant (|t| < 1), and decile
long/short spreads negative. This is not a bug: the same era shows long-only 12-1
momentum *underperforming* SPY (see `reports/momentum_12_1/`), and the period was
dominated by liquid, high-volatility mega-cap growth — precisely the names the
low-volatility and illiquidity factors are short. The toolkit's job is to surface
that verdict cleanly, and the rank-autocorrelations line up with theory
(reversal ≈ 0.27 turnover-heavy vs low-vol ≈ 0.99 sticky). Full numbers and
figures: `reports/factor_eval/summary.md`.

---

## Preprocessing utilities

`quantlab.factors.preprocess` provides standard cross-sectional cleaning steps,
applied to a raw factor Series. All ignore `NaN` and preserve the index.

| Function | Purpose | Default |
|----------|---------|---------|
| `winsorize(s, lower, upper)` | Clip extreme values to quantile bounds | 1% / 99% |
| `zscore(s, ddof)` | Standardize to mean 0, std 1 | `ddof=1` |
| `deciles(s, n)` | Assign equal-count buckets `1..n` (n = highest) | `n=10` |

Typical order of operations for a raw factor: **winsorize → z-score** (for
regression/risk-model inputs) or **winsorize → deciles** (for long/short
portfolio sorts). `deciles` uses first-tie-broken ranks so buckets stay
(near-)equal in count even with duplicate factor values.

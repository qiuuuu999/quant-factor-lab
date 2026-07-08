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

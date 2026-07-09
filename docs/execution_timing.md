# Execution Timing

This document records the execution-timing design used by the backtest engine
(`quantlab.backtest.engine`) and the reasoning behind it. It complements the
point-in-time contract enforced in the factor layer (see
[`factors.md`](factors.md)); together they define where look-ahead bias can and
cannot enter a backtest.

## The problem: same-bar look-ahead

A cross-sectional signal for date `t` is, by construction, a function of prices
*up to and including* `t`'s close (e.g. 12-1 momentum reads the month-end close).
The first version of the momentum backtest then **filled orders at that same
`t` close**. That is a subtle but real look-ahead: the fill price is one of the
inputs that produced the signal, so the strategy trades on information at the
exact instant it is revealed — impossible in practice, since you cannot know the
closing print until the session is over and the auction has cleared.

The bias is different from — and finer than — the coarse look-ahead the factor
layer already blocks (reading rows dated after `as_of_date`). Here every price is
"as of `t` or earlier"; the leak is purely in the *timing of execution* relative
to the *timing of the decision*.

## The rule: decide on `t` close → fill on `t+1` open

The engine separates the **decision** date from the **execution** date:

- **Decision** — the signal is formed on session `t` from data ≤ `t`. Weights are
  keyed by `t` (a `rebalance_calendar` month-end).
- **Execution** — the fill happens on session `t+1`, at that morning's **open**.
  Marking to market still uses the daily **close**.

So the daily cycle is: *form the signal on tonight's close → trade at tomorrow's
open → mark at every close.* No order is ever filled on a price that helped form
its own signal, and no order uses a price the strategy could not have observed
before committing to the trade.

This is implemented with two orthogonal knobs on `run_backtest`, both with
backward-compatible defaults (`execution_lag=0`, `execution_prices=None`
reproduce the legacy same-close behaviour):

| Knob | Meaning | Momentum backtest |
| --- | --- | --- |
| `execution_lag` | trading-day offset from the decision date to the fill day | `1` |
| `execution_prices` | separate panel to fill against (marking still uses `prices`) | adjusted opens |
| `defer_halted` | roll a halted-open leg to the next available open | `True` |

A decision keyed on the **final** session of the window has no `t+1` and is
dropped with a warning (you cannot act on a signal formed on the last day) — this
is why the 2015-2025 run executes 131 rebalances from 132 month-ends.

## Adjusted opens: deriving the fill price

yfinance is fetched with `auto_adjust=False`, so the stored `open/high/low/close`
are **raw** (split/dividend-unadjusted) prices while `adj_close` is fully
adjusted. Filling at the raw open while marking at the adjusted close would mix
two price bases and inject spurious jumps around every split/dividend.

The fix keeps a single total-return basis. The daily adjustment factor is
`adj_close / close`; applying it to any raw field yields its adjusted
counterpart:

```
adj_open = open * (adj_close / close)
```

`quantlab.data.prices.get_prices(..., field="adj_open")` computes this on read
(also `adj_high`, `adj_low`). The default long frame is unchanged — the derived
fields appear only when explicitly requested — so existing readers are
unaffected.

## Halts at the open: roll forward, don't drop

A name can have no open print on its intended fill day (a trading halt or
suspension). Two policies are supported:

- `defer_halted=False` (default) — the leg is skipped and recorded as a
  `SkippedOrder`; the position is left untouched until the next rebalance.
- `defer_halted=True` (used by the momentum book) — the leg is **rolled forward**
  and retried at each subsequent session's open until it fills, or until the next
  rebalance supersedes it (whichever comes first). A leg superseded before it ever
  trades is still recorded as a `SkippedOrder`, so nothing is silently dropped.

Rolling forward is the honest behaviour when executing at the open: a name with
no print this morning should trade at the next available open, not be excluded
from the portfolio for the entire period. Fills are always sized against the
portfolio value *at fill time*, so a deferred leg is internally consistent with
the book it joins.

## Impact: before vs. after (12-1 momentum, S&P 500, 2015-2025)

Removing the same-close look-ahead and moving to `t+1`-open execution changed the
headline metrics only marginally:

| Metric | Before (fill at `t` close) | After (fill at `t+1` open) |
| --- | --- | --- |
| CAGR | 9.98% | 10.16% |
| Sharpe | 0.59 | 0.59 |
| Max Drawdown | −36.78% | −37.73% |
| Total Return | 183.98% | 189.01% |
| Excess CAGR vs SPY | −3.48% | −3.31% |
| Information Ratio | −0.35 | −0.32 |

The small magnitude is itself the useful finding: for a **monthly**-rebalanced,
equal-weight book of **liquid** large caps, the overnight gap between the signal
close and the next open is close to random and roughly averages out — so this
particular look-ahead was not materially inflating the original result. The
correction matters far more for higher-frequency or less-liquid strategies, where
the same-bar fill can flatter returns substantially; the engine now bars that
leak by construction regardless of the strategy plugged into it.

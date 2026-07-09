"""engine — a day-by-day, event-driven backtest loop.

The engine replays a daily price panel in chronological order through a simple
cash + positions account, rebalancing to caller-supplied **target weights** on a
rebalance calendar and marking the book to market every session. It deliberately
knows nothing about *how* the weights were formed (a factor, an optimiser, a
hand-typed dict) — this module is the execution/accounting core only.

Accounting model
----------------
* The book holds ``cash`` and a ``positions`` map of ``ticker -> shares``
  (fractional shares allowed by default).
* Every trading day the portfolio is marked to market at that day's price:
  ``nav = cash + Σ shares * price``. Marking uses a **forward-filled** price so a
  temporary halt (a missing session) holds the last observed price rather than
  dropping the name to zero. Forward-fill only ever looks backwards, so no
  look-ahead is introduced.
* On a rebalance date the target weights are applied against the *pre-trade*
  portfolio value ``V``: ``target_shares_i = w_i * V / price_i``. The delta vs.
  the current holding is filled at that day's price and the
  :class:`~quantlab.backtest.costs.CostModel` cost is subtracted from cash on top
  of the share cash-flow. Weights need not sum to 1 (``< 1`` leaves cash, ``> 1``
  is leverage); costs may push cash slightly negative — that residual is the
  honest frictional drag and shows up directly in NAV.

Halts / missing prices at rebalance
-----------------------------------
A target (or an existing holding that needs selling) with **no price on the
rebalance day** cannot be traded. The engine skips that leg, keeps the current
holding untouched, and records a :class:`SkippedOrder` — it never fabricates a
fill. The next rebalance retries from the then-current state.

Rebalance calendar
------------------
:func:`rebalance_calendar` returns the last trading day of each period
(month/quarter/week) from the price index. Target weights are supplied keyed by
date; a key that falls on a non-trading day is deferred to the next available
session, so a weekend/holiday rebalance simply executes at the next open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

__all__ = [
    "Trade",
    "SkippedOrder",
    "BacktestResult",
    "rebalance_calendar",
    "run_backtest",
]

log = logging.getLogger("quantlab.backtest.engine")

#: Positions smaller than this many shares are treated as flat (float dust).
_SHARE_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

@dataclass
class Trade:
    """A single executed fill."""

    date: pd.Timestamp
    ticker: str
    shares: float          # signed: > 0 buy, < 0 sell
    price: float
    notional: float        # signed cash flow from shares alone (= -shares*price)
    commission: float
    slippage: float
    cost: float            # commission + slippage (always >= 0)


@dataclass
class SkippedOrder:
    """A leg that could not be executed (no price / halted on the rebalance day)."""

    date: pd.Timestamp
    ticker: str
    target_weight: float
    reason: str


@dataclass
class BacktestResult:
    """Standardised output of a backtest run.

    Attributes
    ----------
    nav:
        Daily portfolio net asset value, indexed by trading day.
    cash, positions_value:
        Daily cash and marked position value; ``cash + positions_value == nav``.
    turnover:
        Per-rebalance turnover, indexed by rebalance day. Defined as the total
        absolute traded notional divided by the pre-trade portfolio value (an
        initial full deployment from all-cash is ``1.0``).
    trades:
        Every executed :class:`Trade`, in chronological order.
    skipped:
        Every :class:`SkippedOrder` (halted / missing-price legs).
    initial_capital:
        Starting cash.
    """

    nav: pd.Series
    cash: pd.Series
    positions_value: pd.Series
    turnover: pd.Series
    trades: list[Trade] = field(default_factory=list)
    skipped: list[SkippedOrder] = field(default_factory=list)
    initial_capital: float = 0.0

    @property
    def returns(self) -> pd.Series:
        """Daily simple returns of the NAV series."""
        return self.nav.pct_change().dropna()

    @property
    def total_return(self) -> float:
        """Cumulative return over the whole run."""
        if self.nav.empty:
            return 0.0
        return float(self.nav.iloc[-1] / self.initial_capital - 1.0)

    def trades_frame(self) -> pd.DataFrame:
        """All trades as a tidy DataFrame."""
        cols = ["date", "ticker", "shares", "price", "notional",
                "commission", "slippage", "cost"]
        if not self.trades:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([vars(t) for t in self.trades])[cols]

    def skipped_frame(self) -> pd.DataFrame:
        """All skipped legs as a tidy DataFrame."""
        cols = ["date", "ticker", "target_weight", "reason"]
        if not self.skipped:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([vars(s) for s in self.skipped])[cols]


# --------------------------------------------------------------------------- #
# Price panel helpers
# --------------------------------------------------------------------------- #

def _to_wide(prices: pd.DataFrame, price_field: str) -> pd.DataFrame:
    """Coerce a long or wide price frame into a wide (date x ticker) panel.

    A *long* frame (has ``date``/``ticker``/``<price_field>`` columns) is
    pivoted; anything already indexed by date with one column per ticker is
    returned as-is (sorted).
    """
    cols = set(prices.columns)
    if {"date", "ticker"} <= cols and price_field in cols:
        wide = prices.pivot_table(
            index="date", columns="ticker", values=price_field, aggfunc="last"
        )
        wide.columns.name = None
    else:
        wide = prices.copy()
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def rebalance_calendar(trading_days: pd.DatetimeIndex, freq: str = "month_end") -> list[pd.Timestamp]:
    """Last trading day of each period in ``trading_days``.

    ``freq`` is one of ``month_end`` / ``quarter_end`` / ``week_end``. The result
    is a subset of ``trading_days`` (real sessions), so there is no need to snap
    calendar-period ends onto the exchange calendar afterwards.
    """
    alias = {"month_end": "ME", "quarter_end": "QE", "week_end": "W"}
    if freq not in alias:
        raise ValueError(f"unknown freq {freq!r}; choose from {sorted(alias)}")
    idx = pd.DatetimeIndex(trading_days).sort_values()
    if idx.empty:
        return []
    # Group each session into its period bucket and keep the last session seen.
    periods = idx.to_period({"ME": "M", "QE": "Q", "W": "W"}[alias[freq]])
    s = pd.Series(idx, index=periods)
    return list(s.groupby(level=0).last())


# --------------------------------------------------------------------------- #
# Core loop
# --------------------------------------------------------------------------- #

def _normalize_targets(
    target_weights: dict, trading_days: pd.DatetimeIndex
) -> list[tuple[pd.Timestamp, dict[str, float]]]:
    """Map each requested rebalance date to the trading day it executes on.

    A key on a non-trading day (weekend/holiday) defers to the next session;
    a key after the last session is dropped with a warning. Returns a list of
    ``(execution_day, weights)`` sorted by execution day.
    """
    days = pd.DatetimeIndex(trading_days).sort_values()
    out: list[tuple[pd.Timestamp, dict[str, float]]] = []
    for key, weights in target_weights.items():
        ts = pd.Timestamp(key).normalize()
        pos = days.searchsorted(ts, side="left")
        if pos >= len(days):
            log.warning("rebalance date %s is after the last trading day; dropped", ts.date())
            continue
        out.append((days[pos], dict(weights)))
    out.sort(key=lambda kv: kv[0])
    return out


def _rebalance(
    day: pd.Timestamp,
    targets: dict[str, float],
    cash: float,
    positions: dict[str, float],
    exec_px: pd.Series,
    mark_px: pd.Series,
    cost_model,
    allow_fractional: bool,
) -> tuple[float, float, list[Trade], list[SkippedOrder]]:
    """Trade ``positions`` toward ``targets`` at ``day``'s prices.

    Returns ``(new_cash, traded_notional, trades, skipped)``. ``positions`` is
    mutated in place.
    """
    # Pre-trade portfolio value marks every held name at its (ffilled) price.
    pv = cash
    for t, sh in positions.items():
        px = mark_px.get(t, np.nan)
        if not np.isnan(px):
            pv += sh * px

    trades: list[Trade] = []
    skipped: list[SkippedOrder] = []
    traded_notional = 0.0

    # Consider every target plus any current holding (weight 0 => liquidate).
    tickers = list(dict.fromkeys([*targets.keys(), *positions.keys()]))
    for t in tickers:
        w = targets.get(t, 0.0)
        px = exec_px.get(t, np.nan)
        if np.isnan(px) or px <= 0.0:
            # Halted / no price today: cannot trade this leg, keep the holding.
            if w != 0.0 or abs(positions.get(t, 0.0)) > _SHARE_EPS:
                skipped.append(SkippedOrder(day, t, w, "no price / halted"))
            continue

        target_shares = (w * pv) / px
        if not allow_fractional:
            target_shares = float(int(target_shares))  # truncate toward zero

        delta = target_shares - positions.get(t, 0.0)
        if abs(delta) <= _SHARE_EPS:
            continue

        cost = cost_model.cost(delta, px)
        commission = cost_model.commission(delta)
        slippage = cost_model.slippage(delta, px)
        share_flow = -delta * px            # buy (delta>0) => cash out (negative)
        cash += share_flow - cost
        traded_notional += abs(delta * px)

        new_shares = positions.get(t, 0.0) + delta
        if abs(new_shares) <= _SHARE_EPS:
            positions.pop(t, None)
        else:
            positions[t] = new_shares

        trades.append(Trade(
            date=day, ticker=t, shares=delta, price=float(px),
            notional=share_flow, commission=commission,
            slippage=slippage, cost=cost,
        ))

    return cash, traded_notional, trades, skipped


def run_backtest(
    prices: pd.DataFrame,
    target_weights: dict,
    *,
    initial_capital: float = 1_000_000.0,
    cost_model=None,
    price_field: str = "adj_close",
    allow_fractional: bool = True,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
) -> BacktestResult:
    """Run an event-driven backtest over a daily price panel.

    Parameters
    ----------
    prices:
        Either a *wide* panel indexed by date with one column per ticker, or a
        *long* frame with ``date``/``ticker``/``<price_field>`` columns (as
        returned by :func:`quantlab.data.prices.get_prices`).
    target_weights:
        Mapping ``rebalance_date -> {ticker: weight}``. On each mapped date (or
        the next trading day if it falls on a holiday) the book is rebalanced to
        those weights. Provide keys from :func:`rebalance_calendar` for a monthly
        cadence.
    initial_capital:
        Starting cash.
    cost_model:
        A :class:`~quantlab.backtest.costs.CostModel`; ``None`` means frictionless.
    price_field:
        Column used when ``prices`` is long (default ``adj_close``).
    allow_fractional:
        If ``False``, position sizes are truncated to whole shares.
    start, end:
        Optional inclusive bounds to restrict the simulation window.

    Returns
    -------
    BacktestResult
        Daily NAV/cash/position-value series, per-rebalance turnover, and the
        full trade and skipped-order logs.
    """
    from quantlab.backtest.costs import CostModel

    cost_model = cost_model or CostModel()

    wide = _to_wide(prices, price_field)
    if start is not None:
        wide = wide[wide.index >= pd.Timestamp(start)]
    if end is not None:
        wide = wide[wide.index <= pd.Timestamp(end)]
    if wide.empty:
        empty = pd.Series(dtype=float)
        return BacktestResult(nav=empty, cash=empty, positions_value=empty,
                              turnover=empty, initial_capital=initial_capital)

    trading_days = wide.index
    mark_panel = wide.ffill()   # hold last price through halts for marking only
    schedule = dict(_normalize_targets(target_weights, trading_days))

    cash = float(initial_capital)
    positions: dict[str, float] = {}

    nav_idx, cash_hist, pos_val_hist = [], [], []
    turn_idx, turn_hist = [], []
    trades: list[Trade] = []
    skipped: list[SkippedOrder] = []

    for day in trading_days:
        exec_px = wide.loc[day]
        mark_px = mark_panel.loc[day]

        if day in schedule:
            # Pre-trade value for the turnover denominator.
            pv = cash + sum(
                sh * mark_px.get(t, np.nan)
                for t, sh in positions.items()
                if not np.isnan(mark_px.get(t, np.nan))
            )
            cash, traded_notional, day_trades, day_skipped = _rebalance(
                day, schedule[day], cash, positions,
                exec_px, mark_px, cost_model, allow_fractional,
            )
            trades.extend(day_trades)
            skipped.extend(day_skipped)
            turn_idx.append(day)
            turn_hist.append(traded_notional / pv if pv > 0 else 0.0)

        # End-of-day mark to market.
        pos_val = sum(
            sh * mark_px.get(t, np.nan)
            for t, sh in positions.items()
            if not np.isnan(mark_px.get(t, np.nan))
        )
        nav_idx.append(day)
        cash_hist.append(cash)
        pos_val_hist.append(pos_val)

    idx = pd.DatetimeIndex(nav_idx)
    cash_s = pd.Series(cash_hist, index=idx, name="cash")
    pos_s = pd.Series(pos_val_hist, index=idx, name="positions_value")
    nav_s = (cash_s + pos_s).rename("nav")
    turn_s = pd.Series(turn_hist, index=pd.DatetimeIndex(turn_idx), name="turnover")

    return BacktestResult(
        nav=nav_s,
        cash=cash_s,
        positions_value=pos_s,
        turnover=turn_s,
        trades=trades,
        skipped=skipped,
        initial_capital=float(initial_capital),
    )

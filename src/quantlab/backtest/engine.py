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
* On a rebalance the target weights are applied against the *pre-trade*
  portfolio value ``V``: ``target_shares_i = w_i * V / exec_price_i``. The delta
  vs. the current holding is filled at the **execution price** (see below) and
  the :class:`~quantlab.backtest.costs.CostModel` cost is subtracted from cash on
  top of the share cash-flow. Weights need not sum to 1 (``< 1`` leaves cash,
  ``> 1`` is leverage); costs may push cash slightly negative — that residual is
  the honest frictional drag and shows up directly in NAV.

Execution timing (avoiding look-ahead)
--------------------------------------
Signals are formed from information available *up to and including* a given
session's close, so filling on that same session's close would trade on a price
that co-determines the signal — a subtle look-ahead. The engine therefore
separates the **decision** date from the **execution** date via two knobs:

* ``execution_lag`` — how many trading days after the keyed decision date the
  fill happens. ``0`` (default) keeps the legacy same-session behaviour;
  ``1`` implements the realistic *decide on ``t`` close → trade on ``t+1``* rule.
* ``execution_prices`` — an optional separate price panel to fill against (e.g.
  the adjusted **open**), while the primary ``prices`` panel (adjusted close)
  continues to mark the book to market. When omitted, fills use the same panel
  as marking.

The canonical realistic setup for a momentum book is thus ``execution_lag=1``
with ``execution_prices`` = adjusted opens: form the signal on the month-end
close, execute at the next morning's open, mark every night at the close.

Halts / missing prices at execution
-----------------------------------
A leg with **no execution price on its execution day** (a suspended/halted name)
cannot be filled. Behaviour is controlled by ``defer_halted``:

* ``False`` (default) — skip the leg, keep the current holding, and record a
  :class:`SkippedOrder`; the next rebalance retries from the then-current state.
* ``True`` — *roll the leg forward*: retry it at each subsequent session's
  execution price until it fills, or until the next rebalance supersedes it
  (whichever comes first). A leg that is superseded before it ever trades is
  recorded as a :class:`SkippedOrder`. This is the right policy when executing at
  the open, where a name with no print that morning should simply trade at the
  next available open rather than be dropped for the whole period.

Rebalance calendar
------------------
:func:`rebalance_calendar` returns the last trading day of each period
(month/quarter/week) from the price index. Target weights are supplied keyed by
date; a key that falls on a non-trading day is deferred to the next available
session, and ``execution_lag`` is then counted from that session.
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
    target_weights: dict,
    trading_days: pd.DatetimeIndex,
    execution_lag: int = 0,
) -> list[tuple[pd.Timestamp, dict[str, float]]]:
    """Map each requested decision date to the trading day it executes on.

    A key on a non-trading day (weekend/holiday) first snaps to the next
    session; ``execution_lag`` trading days are then counted from there, so
    ``lag=1`` means "decide on this session's close, fill on the next session".
    A key whose execution day would fall after the last session is dropped with
    a warning. Returns a list of ``(execution_day, weights)`` sorted by execution
    day (later duplicates on the same execution day win).
    """
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0")
    days = pd.DatetimeIndex(trading_days).sort_values()
    out: list[tuple[pd.Timestamp, dict[str, float]]] = []
    for key, weights in target_weights.items():
        ts = pd.Timestamp(key).normalize()
        pos = int(days.searchsorted(ts, side="left")) + execution_lag
        if pos >= len(days):
            log.warning("rebalance date %s (+%d lag) is after the last trading "
                        "day; dropped", ts.date(), execution_lag)
            continue
        out.append((days[pos], dict(weights)))
    out.sort(key=lambda kv: kv[0])
    return out


def _portfolio_value(cash: float, positions: dict[str, float], mark_px: pd.Series) -> float:
    """Cash plus every held name marked at its (ffilled) price; NaN marks skipped."""
    pv = cash
    for t, sh in positions.items():
        px = mark_px.get(t, np.nan)
        if not np.isnan(px):
            pv += sh * px
    return pv


def _execute_legs(
    day: pd.Timestamp,
    legs: dict[str, float],
    cash: float,
    positions: dict[str, float],
    exec_px: pd.Series,
    mark_px: pd.Series,
    cost_model,
    allow_fractional: bool,
) -> tuple[float, float, float, list[Trade], dict[str, float]]:
    """Trade the named ``legs`` toward their target weights at ``day``'s prices.

    Only the tickers in ``legs`` are touched — holdings absent from ``legs`` are
    left alone (callers wanting a name liquidated must pass it with weight 0).
    Each leg is sized to ``w * pv`` shares against the pre-trade portfolio value
    ``pv``. A leg whose execution price is missing/non-positive and that still
    needs trading cannot fill; it is returned in ``halted`` for the caller to
    skip or defer.

    Returns ``(new_cash, traded_notional, pv, trades, halted)``. ``positions`` is
    mutated in place.
    """
    pv = _portfolio_value(cash, positions, mark_px)

    trades: list[Trade] = []
    halted: dict[str, float] = {}
    traded_notional = 0.0

    for t, w in legs.items():
        px = exec_px.get(t, np.nan)
        if np.isnan(px) or px <= 0.0:
            # Halted / no execution print today: cannot trade this leg.
            if w != 0.0 or abs(positions.get(t, 0.0)) > _SHARE_EPS:
                halted[t] = w
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

    return cash, traded_notional, pv, trades, halted


def run_backtest(
    prices: pd.DataFrame,
    target_weights: dict,
    *,
    initial_capital: float = 1_000_000.0,
    cost_model=None,
    price_field: str = "adj_close",
    execution_prices: pd.DataFrame | None = None,
    execution_price_field: str = "adj_open",
    execution_lag: int = 0,
    defer_halted: bool = False,
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
        returned by :func:`quantlab.data.prices.get_prices`). This panel marks
        the book to market every session (and, unless ``execution_prices`` is
        given, is also filled against).
    target_weights:
        Mapping ``decision_date -> {ticker: weight}``. Each key is snapped to a
        trading day and then offset by ``execution_lag`` sessions to get the fill
        day. Provide keys from :func:`rebalance_calendar` for a monthly cadence.
    initial_capital:
        Starting cash.
    cost_model:
        A :class:`~quantlab.backtest.costs.CostModel`; ``None`` means frictionless.
    price_field:
        Column used for marking when ``prices`` is long (default ``adj_close``).
    execution_prices:
        Optional separate price panel (long or wide) to fill trades against — e.g.
        adjusted opens. When ``None``, fills use ``prices`` itself. Marking always
        uses ``prices``.
    execution_price_field:
        Column used for fills when ``execution_prices`` is long (default
        ``adj_open``).
    execution_lag:
        Trading-day offset from each decision date to its fill day. ``0``
        (default) fills on the decision session itself; ``1`` fills on the next
        session — the realistic "decide on ``t`` close, trade on ``t+1``" rule.
    defer_halted:
        When ``True``, a leg with no execution price on its fill day is rolled
        forward to the next session that prices it (until the next rebalance
        supersedes it), instead of being skipped outright. See the module
        docstring. Recommended together with open-price execution.
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

    # Execution panel: a distinct fill panel (e.g. adjusted opens) if supplied,
    # otherwise the marking panel. Aligned to the trading calendar; deliberately
    # NOT forward-filled, so a genuine gap reads as a halt (skip/defer).
    if execution_prices is not None:
        exec_panel = _to_wide(execution_prices, execution_price_field)
        exec_panel = exec_panel.reindex(trading_days)
    else:
        exec_panel = wide

    schedule = dict(_normalize_targets(target_weights, trading_days, execution_lag))

    cash = float(initial_capital)
    positions: dict[str, float] = {}
    # Legs deferred because they were halted on their fill day (defer_halted).
    pending: dict[str, float] = {}
    pending_since: dict[str, pd.Timestamp] = {}

    nav_idx, cash_hist, pos_val_hist = [], [], []
    turn_idx, turn_hist = [], []
    trades: list[Trade] = []
    skipped: list[SkippedOrder] = []

    for day in trading_days:
        exec_px = exec_panel.loc[day]
        mark_px = mark_panel.loc[day]

        if day in schedule:
            # A fresh rebalance supersedes any still-pending deferred legs.
            for t, w in pending.items():
                skipped.append(SkippedOrder(
                    pending_since[t], t, w,
                    "halted; superseded by next rebalance before it could fill"))
            pending.clear()
            pending_since.clear()

            targets = schedule[day]
            # Held names not in the new targets get weight 0 (=> liquidate).
            legs = {**{t: 0.0 for t in positions}, **targets}
            cash, traded_notional, pv, day_trades, halted = _execute_legs(
                day, legs, cash, positions,
                exec_px, mark_px, cost_model, allow_fractional,
            )
            trades.extend(day_trades)
            turn_idx.append(day)
            turn_hist.append(traded_notional / pv if pv > 0 else 0.0)

            for t, w in halted.items():
                if defer_halted:
                    pending[t] = w
                    pending_since[t] = day
                else:
                    skipped.append(SkippedOrder(day, t, w, "no price / halted"))

        elif pending:
            # Retry deferred legs at today's execution price.
            cash, traded_notional, pv, day_trades, still_halted = _execute_legs(
                day, dict(pending), cash, positions,
                exec_px, mark_px, cost_model, allow_fractional,
            )
            if day_trades:
                trades.extend(day_trades)
                turn_idx.append(day)
                turn_hist.append(traded_notional / pv if pv > 0 else 0.0)
            # Keep only the legs that still could not fill (preserve since-dates).
            pending = {t: w for t, w in still_halted.items()}
            pending_since = {t: pending_since[t] for t in pending}

        # End-of-day mark to market.
        pos_val = sum(
            sh * mark_px.get(t, np.nan)
            for t, sh in positions.items()
            if not np.isnan(mark_px.get(t, np.nan))
        )
        nav_idx.append(day)
        cash_hist.append(cash)
        pos_val_hist.append(pos_val)

    # Any legs still pending after the last session never filled.
    for t, w in pending.items():
        skipped.append(SkippedOrder(pending_since[t], t, w,
                                    "halted; never filled before end of data"))

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

"""Tests for the backtest engine and cost model.

Hermetic: every scenario uses a hand-built price panel for a tiny 3-stock
universe (AAA / BBB / CCC) with prices chosen so the correct cash flows, cost
deductions, NAV, and turnover are known in closed form and asserted exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quantlab.backtest import (
    BacktestResult,
    CostModel,
    load_cost_model,
    rebalance_calendar,
    run_backtest,
)
from quantlab.backtest.costs import default_config_path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _wide(prices_by_ticker: dict[str, list[float]], dates: list[str]) -> pd.DataFrame:
    """Build a wide (date x ticker) price panel from lists of daily prices."""
    return pd.DataFrame(prices_by_ticker, index=pd.to_datetime(dates))


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #

def test_cost_model_commission_and_slippage_math():
    cm = CostModel(commission_per_share=0.01, slippage_bps=20.0)
    # 100 shares @ 50: commission = 100 * 0.01 = 1.0
    assert cm.commission(100) == pytest.approx(1.0)
    # slippage = |100 * 50| * 20 bps = 5000 * 0.0020 = 10.0
    assert cm.slippage(100, 50) == pytest.approx(10.0)
    assert cm.cost(100, 50) == pytest.approx(11.0)


def test_cost_model_is_symmetric_in_direction():
    cm = CostModel(commission_per_share=0.01, slippage_bps=20.0)
    assert cm.cost(-100, 50) == cm.cost(100, 50)   # sells cost the same as buys


def test_cost_model_free_is_zero():
    cm = CostModel.free()
    assert cm.cost(1234, 56.7) == 0.0


def test_cost_model_rejects_negative_params():
    with pytest.raises(ValidationError):
        CostModel(commission_per_share=-0.01)
    with pytest.raises(ValidationError):
        CostModel(slippage_bps=-1.0)


def test_load_cost_model_reads_repo_default():
    cm = load_cost_model()
    # Matches configs/backtest.yaml shipped with the repo.
    assert cm.commission_per_share == pytest.approx(0.005)
    assert cm.slippage_bps == pytest.approx(5.0)


def test_load_cost_model_missing_file_is_zero(tmp_path):
    cm = load_cost_model(tmp_path / "does_not_exist.yaml")
    assert cm.cost(100, 100) == 0.0


def test_default_config_path_points_at_repo_configs():
    assert default_config_path().name == "backtest.yaml"
    assert default_config_path().parent.name == "configs"


# --------------------------------------------------------------------------- #
# NAV / accounting — the canonical "buy 1 share at 100, rises to 110" case
# --------------------------------------------------------------------------- #

def test_buy_one_share_and_mark_to_market():
    prices = _wide({"AAA": [100.0, 110.0]}, ["2020-01-31", "2020-02-28"])
    res = run_backtest(
        prices,
        {"2020-01-31": {"AAA": 1.0}},
        initial_capital=100.0,
        cost_model=CostModel.free(),
    )
    # Exactly one share bought at 100, all cash deployed.
    assert res.trades[0].shares == pytest.approx(1.0)
    assert res.trades[0].price == pytest.approx(100.0)
    assert res.cash.iloc[0] == pytest.approx(0.0)
    # NAV: 100 on day one (1 share * 100), 110 once the stock rises to 110.
    assert res.nav.iloc[0] == pytest.approx(100.0)
    assert res.nav.iloc[-1] == pytest.approx(110.0)
    assert res.total_return == pytest.approx(0.10)


def test_costs_are_deducted_from_nav_exactly():
    # Buy 10 shares @ 100 with $0.50/share commission and 10 bps slippage.
    prices = _wide({"AAA": [100.0, 100.0]}, ["2020-01-31", "2020-02-28"])
    cm = CostModel(commission_per_share=0.50, slippage_bps=10.0)
    res = run_backtest(prices, {"2020-01-31": {"AAA": 1.0}},
                       initial_capital=1000.0, cost_model=cm)

    tr = res.trades[0]
    assert tr.shares == pytest.approx(10.0)
    assert tr.commission == pytest.approx(5.0)     # 10 * 0.50
    assert tr.slippage == pytest.approx(1.0)       # 10*100 * 0.0010
    assert tr.cost == pytest.approx(6.0)
    # Cash = 1000 - 1000 (notional) - 6 (cost) = -6; NAV = -6 + 10*100 = 994.
    assert res.cash.iloc[0] == pytest.approx(-6.0)
    assert res.nav.iloc[0] == pytest.approx(994.0)


def test_partial_allocation_leaves_cash():
    prices = _wide({"AAA": [100.0, 100.0]}, ["2020-01-31", "2020-02-28"])
    res = run_backtest(prices, {"2020-01-31": {"AAA": 0.6}},
                       initial_capital=1000.0, cost_model=CostModel.free())
    # 0.6 * 1000 / 100 = 6 shares (600 invested), 400 left in cash.
    assert res.trades[0].shares == pytest.approx(6.0)
    assert res.cash.iloc[0] == pytest.approx(400.0)
    assert res.nav.iloc[0] == pytest.approx(1000.0)


def test_two_asset_book_marks_each_leg():
    prices = _wide(
        {"AAA": [100.0, 120.0], "BBB": [50.0, 40.0]},
        ["2020-01-31", "2020-02-28"],
    )
    res = run_backtest(prices, {"2020-01-31": {"AAA": 0.5, "BBB": 0.5}},
                       initial_capital=1000.0, cost_model=CostModel.free())
    # AAA: 500/100 = 5 sh; BBB: 500/50 = 10 sh; cash fully deployed.
    assert res.cash.iloc[0] == pytest.approx(0.0)
    assert res.nav.iloc[0] == pytest.approx(1000.0)
    # Day 2: 5*120 + 10*40 = 600 + 400 = 1000.
    assert res.nav.iloc[-1] == pytest.approx(1000.0)


def test_integer_shares_truncates_and_holds_residual_cash():
    prices = _wide({"AAA": [101.0, 101.0]}, ["2020-01-31", "2020-02-28"])
    res = run_backtest(prices, {"2020-01-31": {"AAA": 1.0}},
                       initial_capital=1000.0, cost_model=CostModel.free(),
                       allow_fractional=False)
    # 1000 / 101 = 9.90 -> 9 whole shares (909), 91 residual cash.
    assert res.trades[0].shares == pytest.approx(9.0)
    assert res.cash.iloc[0] == pytest.approx(91.0)
    assert res.nav.iloc[0] == pytest.approx(1000.0)


# --------------------------------------------------------------------------- #
# Rebalancing & turnover
# --------------------------------------------------------------------------- #

def test_full_switch_turnover_is_two():
    prices = _wide(
        {"AAA": [100.0, 100.0, 100.0], "BBB": [100.0, 100.0, 100.0]},
        ["2020-01-31", "2020-02-28", "2020-03-31"],
    )
    res = run_backtest(
        prices,
        {"2020-01-31": {"AAA": 1.0}, "2020-03-31": {"BBB": 1.0}},
        initial_capital=1000.0,
        cost_model=CostModel.free(),
    )
    ts0 = pd.Timestamp("2020-01-31")
    ts2 = pd.Timestamp("2020-03-31")
    # Initial deployment from all-cash trades 100% of the book.
    assert res.turnover.loc[ts0] == pytest.approx(1.0)
    # Selling all AAA (1000) and buying all BBB (1000) trades 2x the book.
    assert res.turnover.loc[ts2] == pytest.approx(2.0)
    # Ends fully in BBB, still worth 1000.
    assert res.positions_value.iloc[-1] == pytest.approx(1000.0)
    assert res.nav.iloc[-1] == pytest.approx(1000.0)


def test_rebalance_defers_to_next_trading_day():
    # 2020-02-01 is a Saturday; the rebalance must execute on Monday 2020-02-03.
    prices = _wide({"AAA": [100.0, 100.0]}, ["2020-01-31", "2020-02-03"])
    res = run_backtest(prices, {"2020-02-01": {"AAA": 1.0}},
                       initial_capital=1000.0, cost_model=CostModel.free())
    assert len(res.trades) == 1
    assert res.trades[0].date == pd.Timestamp("2020-02-03")
    # Day one is still all cash (no look-ahead to the pending rebalance).
    assert res.nav.iloc[0] == pytest.approx(1000.0)
    assert res.positions_value.iloc[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Halts / missing prices
# --------------------------------------------------------------------------- #

def test_halted_ticker_is_skipped_and_recorded():
    # BBB has no price on the rebalance day -> its leg cannot fill.
    prices = _wide(
        {"AAA": [100.0, 100.0], "BBB": [np.nan, 50.0]},
        ["2020-01-31", "2020-02-28"],
    )
    res = run_backtest(prices, {"2020-01-31": {"AAA": 0.5, "BBB": 0.5}},
                       initial_capital=1000.0, cost_model=CostModel.free())
    # Only AAA traded; BBB recorded as skipped.
    assert [t.ticker for t in res.trades] == ["AAA"]
    assert len(res.skipped) == 1
    skip = res.skipped[0]
    assert skip.ticker == "BBB"
    assert skip.date == pd.Timestamp("2020-01-31")
    # AAA leg still fills against the pre-trade value (0.5 * 1000 / 100 = 5 sh).
    assert res.trades[0].shares == pytest.approx(5.0)
    assert res.cash.iloc[0] == pytest.approx(500.0)


def test_missing_mark_price_holds_last_value():
    # AAA halts on day 2 (NaN) then resumes; day-2 NAV holds the last price.
    prices = _wide({"AAA": [100.0, np.nan, 120.0]},
                   ["2020-01-31", "2020-02-28", "2020-03-31"])
    res = run_backtest(prices, {"2020-01-31": {"AAA": 1.0}},
                       initial_capital=1000.0, cost_model=CostModel.free())
    # 10 shares held throughout. Day 2 marks at the carried-forward 100.
    assert res.nav.iloc[1] == pytest.approx(1000.0)
    # Day 3 re-marks at the recovered 120: 10 * 120 = 1200.
    assert res.nav.iloc[2] == pytest.approx(1200.0)


# --------------------------------------------------------------------------- #
# Rebalance calendar utility
# --------------------------------------------------------------------------- #

def test_rebalance_calendar_month_end_picks_last_trading_day():
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    cal = rebalance_calendar(days, "month_end")
    assert cal == [
        pd.Timestamp("2020-01-31"),   # Friday
        pd.Timestamp("2020-02-28"),   # Friday
        pd.Timestamp("2020-03-31"),   # Tuesday
    ]


def test_rebalance_calendar_quarter_end():
    days = pd.bdate_range("2020-01-01", "2020-06-30")
    cal = rebalance_calendar(days, "quarter_end")
    assert cal == [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30")]


def test_rebalance_calendar_rejects_unknown_freq():
    with pytest.raises(ValueError):
        rebalance_calendar(pd.bdate_range("2020-01-01", "2020-01-10"), "daily")


# --------------------------------------------------------------------------- #
# Result object plumbing
# --------------------------------------------------------------------------- #

def test_result_frames_and_returns():
    prices = _wide({"AAA": [100.0, 110.0, 121.0]},
                   ["2020-01-31", "2020-02-28", "2020-03-31"])
    res = run_backtest(prices, {"2020-01-31": {"AAA": 1.0}},
                       initial_capital=1000.0, cost_model=CostModel.free())
    assert isinstance(res, BacktestResult)
    tf = res.trades_frame()
    assert list(tf.columns) == ["date", "ticker", "shares", "price",
                                "notional", "commission", "slippage", "cost"]
    assert len(tf) == 1
    # Two +10% steps -> daily returns of 0.10 each.
    assert res.returns.tolist() == pytest.approx([0.10, 0.10])
    assert res.total_return == pytest.approx(0.21)


def test_long_frame_input_is_pivoted():
    # get_prices-style long frame should work without manual pivoting.
    long = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-31", "2020-02-28",
                                "2020-01-31", "2020-02-28"]),
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
        "adj_close": [100.0, 110.0, 50.0, 55.0],
    })
    res = run_backtest(long, {"2020-01-31": {"AAA": 0.5, "BBB": 0.5}},
                       initial_capital=1000.0, cost_model=CostModel.free())
    # 5 AAA + 10 BBB; day 2: 5*110 + 10*55 = 550 + 550 = 1100.
    assert res.nav.iloc[0] == pytest.approx(1000.0)
    assert res.nav.iloc[-1] == pytest.approx(1100.0)


def test_empty_panel_returns_empty_result():
    empty = pd.DataFrame({"AAA": []}, index=pd.to_datetime([]))
    res = run_backtest(empty, {}, initial_capital=1000.0)
    assert res.nav.empty
    assert res.total_return == 0.0

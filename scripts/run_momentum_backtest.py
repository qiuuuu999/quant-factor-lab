"""First full backtest: long-only 12-1 momentum, S&P 500, 2015-2025.

Strategy
--------
* Universe: the **point-in-time** S&P 500 membership at each rebalance date
  (survivorship-bias-free), intersected with names we have prices for.
* Signal: 12-1 price momentum (:class:`quantlab.factors.MomentumFactor`).
* Selection: rank into deciles each month; hold the top 20% (deciles 9-10).
* Weighting: equal weight, rebalanced on the last trading day of each month.
* Costs: default config (per-share commission + bps slippage).
* Benchmark: SPY over the same window.

Outputs (under ``reports/momentum_12_1/``): the equity curve vs. SPY (log
scale), the drawdown curve, and the performance summary as PNG + Markdown. The
metrics table is also printed to stdout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab.backtest.costs import load_cost_model
from quantlab.backtest.engine import rebalance_calendar, run_backtest
from quantlab.backtest.metrics import compute_metrics
from quantlab.data.prices import get_prices, universe_symbols
from quantlab.data.universe import load_universe
from quantlab.factors import MomentumFactor, deciles
from quantlab.reports import (
    metrics_markdown,
    metrics_table_png,
    plot_drawdown,
    plot_nav_comparison,
)

START = "2015-01-01"
END = "2025-12-31"
DATA_START = "2014-01-01"          # 12m lookback before START
INITIAL_CAPITAL = 1_000_000.0
TOP_DECILES = (9, 10)              # top 20%
OUTDIR = Path("reports/momentum_12_1")


def build_momentum_weights(long_prices, rebal_dates, universe, price_tickers):
    """Target weights per rebalance: equal-weight the top-decile momentum names."""
    factor = MomentumFactor(lookback=12, skip=1)
    long_prices = long_prices.copy()
    weights: dict[pd.Timestamp, dict[str, float]] = {}
    n_selected = []
    for dt in rebal_dates:
        members = set(universe.members_as_of(dt)) & price_tickers
        if not members:
            continue
        lo = dt - pd.DateOffset(months=16)   # enough history for 12-1 momentum
        sub = long_prices[
            long_prices["ticker"].isin(members)
            & (long_prices["date"] >= lo)
            & (long_prices["date"] <= dt)
        ]
        mom = factor.compute(sub, sorted(members), dt)   # PIT-enforced
        buckets = deciles(mom, 10)
        selected = buckets[buckets.isin(TOP_DECILES)].index.tolist()
        if not selected:
            continue
        w = 1.0 / len(selected)
        weights[pd.Timestamp(dt)] = {t: w for t in selected}
        n_selected.append(len(selected))
    avg = sum(n_selected) / len(n_selected) if n_selected else 0
    print(f"Built weights for {len(weights)} rebalances "
          f"(avg {avg:.0f} names/rebalance)")
    return weights


def main() -> None:
    print("Loading universe and prices ...", flush=True)
    universe = load_universe()
    union = universe_symbols(START, END)

    # Daily long frame (for the momentum signal) and wide adj-close panel (for
    # the engine), both labelled with the requested historical symbols.
    long_prices = get_prices(union, DATA_START, END)
    price_tickers = set(long_prices["ticker"].unique())
    wide = get_prices(union, START, END, field="adj_close")
    print(f"Universe union: {len(union)} symbols; priced: {len(price_tickers)}; "
          f"panel {wide.shape[0]} days x {wide.shape[1]} names", flush=True)

    # Month-end rebalance calendar over the backtest window.
    rebal_dates = [
        d for d in rebalance_calendar(wide.index, "month_end")
        if pd.Timestamp(START) <= d <= pd.Timestamp(END)
    ]
    print(f"Rebalance dates: {len(rebal_dates)} "
          f"({rebal_dates[0].date()} .. {rebal_dates[-1].date()})", flush=True)

    weights = build_momentum_weights(long_prices, rebal_dates, universe, price_tickers)

    print("Running backtest ...", flush=True)
    cost_model = load_cost_model()
    result = run_backtest(
        wide, weights,
        initial_capital=INITIAL_CAPITAL,
        cost_model=cost_model,
        price_field="adj_close",
        start=START, end=END,
    )

    # SPY benchmark, scaled to the same starting capital on the first NAV date.
    spy = get_prices("SPY", START, END, field="adj_close")["SPY"].reindex(
        result.nav.index).ffill()
    spy_nav = INITIAL_CAPITAL * spy / spy.iloc[0]

    metrics = compute_metrics(
        result.nav, benchmark=spy_nav, turnover=result.turnover
    )

    # --- artifacts ------------------------------------------------------- #
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plot_nav_comparison(
        result.nav, spy_nav, OUTDIR / "equity_curve.png",
        title="12-1 Momentum (top 20%) vs. SPY — 2015-2025",
        strategy_label="Momentum 12-1", benchmark_label="SPY", log_scale=True,
    )
    plot_drawdown(result.nav, OUTDIR / "drawdown.png",
                  title="12-1 Momentum — Drawdown", label="Momentum 12-1")
    metrics_table_png(metrics, OUTDIR / "metrics_table.png",
                      title="12-1 Momentum — Performance Summary")
    preamble = (
        f"- **Universe**: point-in-time S&P 500 ({len(price_tickers)} priced names)\n"
        f"- **Signal**: 12-1 momentum, top 20% (deciles 9-10), equal-weight\n"
        f"- **Rebalance**: monthly ({len(weights)} rebalances)\n"
        f"- **Costs**: {cost_model.commission_per_share}/share commission, "
        f"{cost_model.slippage_bps} bps slippage\n"
        f"- **Window**: {result.nav.index[0].date()} .. {result.nav.index[-1].date()}\n"
        f"- **Benchmark**: SPY"
    )
    metrics_markdown(metrics, OUTDIR / "metrics.md",
                     title="12-1 Momentum — Performance Summary", preamble=preamble)
    result.nav.rename("nav").to_frame().assign(spy=spy_nav).to_csv(OUTDIR / "nav.csv")

    print("\n" + "=" * 52)
    print("12-1 MOMENTUM (top 20%) vs SPY — 2015-2025")
    print("=" * 52)
    print(metrics.summary())
    print(f"\nArtifacts written to {OUTDIR}/")


if __name__ == "__main__":
    main()

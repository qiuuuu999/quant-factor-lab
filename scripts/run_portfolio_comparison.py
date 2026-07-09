"""Portfolio-construction comparison: same signal, three ways to size it.

Reuses the exact top-decile 12-1 momentum selection from
``scripts/run_momentum_backtest.py`` (point-in-time S&P 500, top 20% by 12-1
momentum, monthly rebalance) and builds target weights three different ways
at every rebalance, holding the selection itself fixed so any difference in
the resulting backtest is attributable purely to the *weighting scheme*:

1. **Equal weight** — ``1/n`` per selected name (the existing baseline).
2. **Mean-variance** (:class:`quantlab.portfolio.optimizer.MeanVarianceOptimizer`)
   — alpha is that period's winsorized, z-scored momentum score; the
   covariance is :class:`~quantlab.risk.covariance.LedoitWolfCovariance` fit
   on trailing monthly returns of the selected names (``N`` names typically
   exceeds the ``T`` trailing months available, the textbook case Ledoit-Wolf
   exists for — see ``docs/risk.md``). Long-only, 2% single-name cap,
   turnover-penalized against the prior rebalance's weights.
3. **Risk parity** (:class:`quantlab.portfolio.risk_parity.RiskParityOptimizer`)
   — same covariance, same names, no alpha: the risk-only control group.

Names without enough trailing return history to estimate a covariance
(recent IPOs, gaps) are dropped from the mean-variance/risk-parity sleeves for
that rebalance only (equal weight is unaffected, since it needs no
covariance) — a practical necessity documented inline at the filter site.

All three run through identical execution assumptions (decide on month-end
close, fill at next-session open, defer halted opens, default cost model), so
the comparison isolates the construction method. Outputs (under
``reports/portfolio_comparison/``): one equity-curve chart with all three NAVs
+ SPY, and a comparison metrics table (PNG + Markdown) with Sharpe, max
drawdown, and annualized turnover side by side. The table is also printed to
stdout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab.backtest.costs import load_cost_model
from quantlab.backtest.engine import rebalance_calendar, run_backtest
from quantlab.backtest.metrics import compute_metrics
from quantlab.data.prices import get_prices, universe_symbols
from quantlab.data.universe import load_universe
from quantlab.factors import MomentumFactor, deciles, winsorize, zscore
from quantlab.portfolio import MeanVarianceOptimizer, RiskParityOptimizer
from quantlab.reports import (
    metrics_comparison_markdown,
    metrics_comparison_table_png,
    plot_multi_nav,
)
from quantlab.risk.covariance import LedoitWolfCovariance

START = "2015-01-01"
END = "2025-12-31"
DATA_START = "2014-01-01"          # earliest date the local price store covers
INITIAL_CAPITAL = 1_000_000.0
TOP_DECILES = (9, 10)              # top 20%, mirrors run_momentum_backtest.py
MAX_WEIGHT = 0.02                  # 2% single-name cap
RISK_AVERSION = 5.0
TURNOVER_LAMBDA = 0.01
COV_HISTORY_MONTHS = 24
MIN_COV_PERIODS = 6                # minimum trailing months to trust a name's covariance row
OUTDIR = Path("reports/portfolio_comparison")


def _select_names(long_prices, dt, universe, price_tickers, factor):
    """Point-in-time top-20%-by-momentum selection, identical to run_momentum_backtest.py."""
    members = set(universe.members_as_of(dt)) & price_tickers
    if not members:
        return None, None
    lo = dt - pd.DateOffset(months=16)   # enough history for 12-1 momentum
    sub = long_prices[
        long_prices["ticker"].isin(members)
        & (long_prices["date"] >= lo)
        & (long_prices["date"] <= dt)
    ]
    mom = factor.compute(sub, sorted(members), dt)   # PIT-enforced
    buckets = deciles(mom, 10)
    selected = buckets[buckets.isin(TOP_DECILES)].index.tolist()
    return selected, mom


def build_three_weight_schemes(long_prices, monthly_returns, rebal_dates, universe, price_tickers):
    """Equal-weight / mean-variance / risk-parity target weights, one dict each."""
    factor = MomentumFactor(lookback=12, skip=1)
    mv_opt = MeanVarianceOptimizer(
        risk_aversion=RISK_AVERSION, long_only=True,
        max_weight=MAX_WEIGHT, turnover_penalty=TURNOVER_LAMBDA,
    )
    rp_opt = RiskParityOptimizer(max_weight=MAX_WEIGHT)

    eq_weights: dict[pd.Timestamp, dict[str, float]] = {}
    mv_weights: dict[pd.Timestamp, dict[str, float]] = {}
    rp_weights: dict[pd.Timestamp, dict[str, float]] = {}
    prev_mv_weights: pd.Series | None = None

    for dt in rebal_dates:
        selected, mom = _select_names(long_prices, dt, universe, price_tickers, factor)
        if not selected:
            continue
        eq_weights[dt] = {t: 1.0 / len(selected) for t in selected}

        # Winsorize + z-score cross-sectionally over the *full* cross-section
        # (the platform convention -- see docs/factors.md), then subset to
        # the selected names as the optimizer's alpha.
        alpha_all = zscore(winsorize(mom, 0.01, 0.99))
        alpha = alpha_all.reindex(selected).fillna(0.0)

        hist = monthly_returns.loc[monthly_returns.index <= dt].tail(COV_HISTORY_MONTHS)
        # Only names with a fully-populated trailing history enter the
        # covariance estimate; a name with any gap (e.g. recent IPO) is
        # dropped from the risk-based sleeves for this rebalance only.
        cov_tickers = [
            t for t in selected
            if t in hist.columns and hist[t].tail(MIN_COV_PERIODS).notna().all()
            and hist[t].notna().sum() >= MIN_COV_PERIODS
        ]
        if len(cov_tickers) < 2:
            mv_weights[dt] = dict(eq_weights[dt])
            rp_weights[dt] = dict(eq_weights[dt])
            continue

        cov_hist = hist[cov_tickers].dropna(how="any")
        cov = LedoitWolfCovariance(min_periods=MIN_COV_PERIODS).estimate(cov_hist)
        a = alpha.reindex(cov_tickers).fillna(0.0)

        prev = None
        if prev_mv_weights is not None:
            prev = prev_mv_weights.reindex(cov_tickers).fillna(0.0)
        mv_result = mv_opt.optimize(a, cov, previous_weights=prev)
        mv_w = mv_result.weights.clip(lower=0.0)
        mv_w = mv_w[mv_w > 1e-6]
        mv_weights[dt] = mv_w.to_dict()
        prev_mv_weights = mv_result.weights

        rp_result = rp_opt.optimize(cov)
        rp_w = rp_result.weights[rp_result.weights > 1e-6]
        rp_weights[dt] = rp_w.to_dict()

    return eq_weights, mv_weights, rp_weights


def _run(prices, wide_open, weights, cost_model):
    return run_backtest(
        prices, weights,
        initial_capital=INITIAL_CAPITAL,
        cost_model=cost_model,
        price_field="adj_close",
        execution_prices=wide_open,
        execution_price_field="adj_open",
        execution_lag=1,
        defer_halted=True,
        start=START, end=END,
    )


def main() -> None:
    print("Loading universe and prices ...", flush=True)
    universe = load_universe()
    union = universe_symbols(START, END)

    long_prices = get_prices(union, DATA_START, END)
    price_tickers = set(long_prices["ticker"].unique())
    wide = get_prices(union, START, END, field="adj_close")
    wide_open = get_prices(union, START, END, field="adj_open")
    wide_hist = get_prices(union, DATA_START, END, field="adj_close")
    monthly_returns = wide_hist.resample("ME").last().pct_change()
    print(f"Universe union: {len(union)} symbols; priced: {len(price_tickers)}", flush=True)

    rebal_dates = [
        d for d in rebalance_calendar(wide.index, "month_end")
        if pd.Timestamp(START) <= d <= pd.Timestamp(END)
    ]
    print(f"Rebalance dates: {len(rebal_dates)} "
          f"({rebal_dates[0].date()} .. {rebal_dates[-1].date()})", flush=True)

    print("Building equal-weight / mean-variance / risk-parity weight schemes ...", flush=True)
    eq_w, mv_w, rp_w = build_three_weight_schemes(
        long_prices, monthly_returns, rebal_dates, universe, price_tickers
    )
    print(f"  equal-weight: {len(eq_w)} rebalances built", flush=True)
    print(f"  mean-variance: {len(mv_w)} rebalances built", flush=True)
    print(f"  risk-parity: {len(rp_w)} rebalances built", flush=True)

    cost_model = load_cost_model()
    results = {}
    for label, weights in [("Equal Weight", eq_w), ("Mean-Variance", mv_w), ("Risk Parity", rp_w)]:
        print(f"Running backtest: {label} ...", flush=True)
        results[label] = _run(wide, wide_open, weights, cost_model)

    spy = get_prices("SPY", START, END, field="adj_close")["SPY"]
    navs = {}
    metrics = {}
    for label, res in results.items():
        navs[label] = res.nav
        bench = spy.reindex(res.nav.index).ffill()
        bench_nav = INITIAL_CAPITAL * bench / bench.iloc[0]
        metrics[label] = compute_metrics(res.nav, benchmark=bench_nav, turnover=res.turnover)

    spy_full = spy.reindex(navs["Equal Weight"].index).ffill()
    spy_nav = INITIAL_CAPITAL * spy_full / spy_full.iloc[0]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    plot_multi_nav(
        navs, OUTDIR / "equity_curve.png",
        title="12-1 Momentum: Equal Weight vs. Mean-Variance vs. Risk Parity (2015-2025)",
        benchmark=spy_nav, benchmark_label="SPY", log_scale=True,
    )
    metrics_comparison_table_png(
        metrics, OUTDIR / "comparison_table.png",
        title="Portfolio Construction Comparison — 12-1 Momentum, 2015-2025",
    )
    preamble = (
        f"- **Signal**: 12-1 momentum, top 20% (deciles 9-10), same selection for all three\n"
        f"- **Equal Weight**: `1/n` per selected name\n"
        f"- **Mean-Variance**: alpha = z-scored momentum; covariance = Ledoit-Wolf on trailing "
        f"{COV_HISTORY_MONTHS}m returns; long-only, {MAX_WEIGHT:.0%} cap, "
        f"turnover_penalty={TURNOVER_LAMBDA}, risk_aversion={RISK_AVERSION}\n"
        f"- **Risk Parity**: equal risk contribution on the same Ledoit-Wolf covariance, "
        f"{MAX_WEIGHT:.0%} cap, no alpha\n"
        f"- **Execution**: signal on month-end close, fill at next-session open (t+1), "
        f"halted opens rolled forward\n"
        f"- **Costs**: {cost_model.commission_per_share}/share commission, "
        f"{cost_model.slippage_bps} bps slippage\n"
        f"- **Window**: {START} .. {END}\n"
        f"- **Benchmark**: SPY"
    )
    metrics_comparison_markdown(
        metrics, OUTDIR / "summary.md",
        title="Portfolio Construction Comparison — 12-1 Momentum, 2015-2025",
        preamble=preamble,
    )
    pd.DataFrame(navs).assign(spy=spy_nav).to_csv(OUTDIR / "nav.csv")

    print("\n" + "=" * 78)
    print("PORTFOLIO CONSTRUCTION COMPARISON — 12-1 MOMENTUM, 2015-2025")
    print("=" * 78)
    rows = ["Total Return", "CAGR", "Annual Volatility", "Sharpe Ratio",
            "Max Drawdown", "Calmar Ratio", "Annual Turnover"]
    comp = pd.DataFrame({label: m.to_series() for label, m in metrics.items()}).loc[rows]
    print(comp.to_string())
    print(f"\nArtifacts written to {OUTDIR}/")


if __name__ == "__main__":
    main()

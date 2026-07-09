"""Regime-adaptive dynamic factor weighting vs. three baselines, 2017-2025.

Four strategies, one tearsheet:

1. **Regime-Adaptive** (:mod:`quantlab.portfolio.regime_adaptive`) — at every
   rebalance, combine the four price factors into one composite score using
   weights derived from each factor's own regime-conditioned IC track record
   *up to that point in time* (current regime from
   :func:`~quantlab.portfolio.regime_adaptive.pit_regime_by_date`, an
   expanding-window, point-in-time classification of SPY; factor weights from
   :func:`~quantlab.portfolio.regime_adaptive.regime_conditioned_factor_weights`,
   an expanding-window mean IC conditioned on that regime, using only IC
   observations dated strictly before the rebalance). Top 20% by composite
   score, equal-weighted.
2. **Static Multi-Factor** (control group) — the same four factors, same
   composite-score-then-top-20% selection, but combined with a fixed
   1/4-1/4-1/4-1/4 weight every period: no regime conditioning, no historical
   IC at all. Isolates what regime-conditioning specifically adds (or costs)
   over simply diversifying across factors.
3. **Pure Momentum** — the platform's original 12-1 momentum strategy
   (identical selection to ``scripts/run_momentum_backtest.py``), as the
   single-factor reference point.
4. **SPY** — buy-and-hold benchmark.

Point-in-time warm-up
----------------------
The factor-regime IC history needs time to accumulate before it is trustworthy
(``min_obs`` regime-matched prior observations per factor). IC and regime
series are built starting ``WARMUP_START`` (2015-01), but capital is only
deployed — i.e. only rebalances from ``START`` (2017-01) onward produce
tradable weights — for all four strategies, so the comparison is apples to
apples over the same 2017-2025 window while regime-adaptive's factor weights
already have two years of (real, not synthetic) regime history behind them by
the first live rebalance.

Outputs (under ``reports/regime_adaptive/``): one equity-curve chart with all
four NAVs, a comparison metrics table (PNG + Markdown), the regime-adaptive
sleeve's factor-weight history (CSV, for inspecting how the mix shifted), and
a written analysis in ``docs/regime_adaptive.md``. The comparison table is
also printed to stdout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab.backtest.costs import load_cost_model
from quantlab.backtest.engine import rebalance_calendar, run_backtest
from quantlab.backtest.metrics import compute_metrics
from quantlab.data.prices import get_prices, universe_symbols
from quantlab.data.universe import load_universe
from quantlab.factors import (
    AmihudIlliquidityFactor,
    LowVolatilityFactor,
    MomentumFactor,
    ShortTermReversalFactor,
    deciles,
    winsorize,
    zscore,
)
from quantlab.factors.evaluation import build_factor_panel, information_coefficient
from quantlab.portfolio.regime_adaptive import (
    build_regime_adaptive_weights,
    build_static_multifactor_weights,
    pit_regime_by_date,
)
from quantlab.reports import (
    metrics_comparison_markdown,
    metrics_comparison_table_png,
    plot_multi_nav,
)

WARMUP_START = "2015-01-01"        # IC/regime history begins accumulating here
START = "2017-01-01"               # capital is deployed (all four strategies) from here
END = "2025-12-31"
DATA_START = "2013-09-01"          # >16m history before WARMUP_START for factor formation
INITIAL_CAPITAL = 1_000_000.0
TOP_DECILES = (9, 10)              # top 20%
N_BUCKETS = 10
NEGATIVE_HANDLING = "invert"       # regime_conditioned_factor_weights default
MIN_OBS = 6                        # min regime-matched prior IC observations to trust a factor
OUTDIR = Path("reports/regime_adaptive")

FACTORS = [
    MomentumFactor(lookback=12, skip=1),
    LowVolatilityFactor(window=252),
    ShortTermReversalFactor(lookback_months=1),
    AmihudIlliquidityFactor(window=21),
]


def build_momentum_weights(long_prices, rebal_dates, universe, price_tickers):
    """Target weights per rebalance: equal-weight the top-decile momentum names.

    Identical selection rule to ``scripts/run_momentum_backtest.py``.
    """
    factor = MomentumFactor(lookback=12, skip=1)
    weights: dict[pd.Timestamp, dict[str, float]] = {}
    for dt in rebal_dates:
        members = set(universe.members_as_of(dt)) & price_tickers
        if not members:
            continue
        lo = dt - pd.DateOffset(months=16)
        sub = long_prices[
            long_prices["ticker"].isin(members)
            & (long_prices["date"] >= lo)
            & (long_prices["date"] <= dt)
        ]
        mom = factor.compute(sub, sorted(members), dt)
        buckets = deciles(mom, 10)
        selected = buckets[buckets.isin(TOP_DECILES)].index.tolist()
        if not selected:
            continue
        w = 1.0 / len(selected)
        weights[pd.Timestamp(dt)] = {t: w for t in selected}
    return weights


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
    union = universe_symbols(WARMUP_START, END)
    long_prices = get_prices(union, DATA_START, END)
    price_tickers = set(long_prices["ticker"].unique())
    wide = get_prices(union, START, END, field="adj_close")
    wide_open = get_prices(union, START, END, field="adj_open")
    print(f"Universe union: {len(union)} symbols; priced: {len(price_tickers)}", flush=True)

    all_rebal_dates = [
        d for d in rebalance_calendar(
            get_prices(union, WARMUP_START, END, field="adj_close").index, "month_end"
        )
        if pd.Timestamp(WARMUP_START) <= d <= pd.Timestamp(END)
    ]
    backtest_rebal_dates = [d for d in all_rebal_dates if d >= pd.Timestamp(START)]
    print(f"Rebalance dates: {len(all_rebal_dates)} total from {WARMUP_START} "
          f"(warm-up), {len(backtest_rebal_dates)} live from {START}", flush=True)

    def members_by_date(dt: pd.Timestamp):
        return set(universe.members_as_of(dt)) & price_tickers

    print("Building factor panels and IC history (from warm-up start) ...", flush=True)
    panels = {
        f.name: build_factor_panel(f, long_prices, all_rebal_dates, members_by_date, history_months=16)
        for f in FACTORS
    }
    ic_by_factor = {name: information_coefficient(panel) for name, panel in panels.items()}

    # Winsorized, z-scored cross-sectional exposures per date (mirrors
    # quantlab.risk.factor_risk.build_exposure_panel's construction, reusing
    # the panels already built above instead of recomputing them).
    exposures_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for dt in all_rebal_dates:
        if not all(dt in p.dates for p in panels.values()):
            continue
        cols = {name: zscore(winsorize(p.factor_values.loc[dt], 0.01, 0.99)) for name, p in panels.items()}
        exposures_by_date[dt] = pd.DataFrame(cols)

    print("Classifying SPY market regime (point-in-time, expanding median) ...", flush=True)
    spy_hist = get_prices("SPY", DATA_START, END, field="adj_close")["SPY"]
    regime_by_date = pit_regime_by_date(spy_hist, all_rebal_dates)

    print("Building regime-adaptive weights ...", flush=True)
    ra_weights, diagnostics = build_regime_adaptive_weights(
        exposures_by_date, ic_by_factor, regime_by_date, backtest_rebal_dates,
        negative_handling=NEGATIVE_HANDLING, min_obs=MIN_OBS,
        n_buckets=N_BUCKETS, top_buckets=TOP_DECILES,
    )
    print(f"  {len(ra_weights)} live rebalances built", flush=True)

    print("Building static multi-factor weights ...", flush=True)
    static_weights = build_static_multifactor_weights(
        exposures_by_date, backtest_rebal_dates, n_buckets=N_BUCKETS, top_buckets=TOP_DECILES,
    )
    print(f"  {len(static_weights)} live rebalances built", flush=True)

    print("Building pure-momentum weights ...", flush=True)
    momentum_weights = build_momentum_weights(long_prices, backtest_rebal_dates, universe, price_tickers)
    print(f"  {len(momentum_weights)} live rebalances built", flush=True)

    cost_model = load_cost_model()
    results = {}
    for label, weights in [
        ("Regime-Adaptive", ra_weights),
        ("Static Multi-Factor", static_weights),
        ("Pure Momentum", momentum_weights),
    ]:
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

    spy_full = spy.reindex(navs["Pure Momentum"].index).ffill()
    spy_nav = INITIAL_CAPITAL * spy_full / spy_full.iloc[0]
    metrics["SPY"] = compute_metrics(spy_nav)
    navs["SPY"] = spy_nav

    OUTDIR.mkdir(parents=True, exist_ok=True)
    plot_multi_nav(
        navs, OUTDIR / "equity_curve.png",
        title="Regime-Adaptive vs. Static Multi-Factor vs. Pure Momentum vs. SPY (2017-2025)",
        log_scale=True,
    )
    metrics_comparison_table_png(
        metrics, OUTDIR / "comparison_table.png",
        title="Regime-Adaptive Strategy Comparison, 2017-2025",
    )
    preamble = (
        f"- **Factors**: 12-1 momentum, low volatility, 1-month reversal, Amihud illiquidity\n"
        f"- **Regime-Adaptive**: composite score = regime-conditioned-IC-weighted combination "
        f"of the four z-scored factors (negative_handling={NEGATIVE_HANDLING!r}, "
        f"min_obs={MIN_OBS}); top 20% by composite score, equal-weighted\n"
        f"- **Static Multi-Factor**: same selection rule, fixed equal (1/4 each) factor "
        f"combination -- no regime conditioning\n"
        f"- **Pure Momentum**: 12-1 momentum only, top 20%, equal-weighted (identical to "
        f"`reports/momentum_12_1/`)\n"
        f"- **Regime/IC warm-up**: {WARMUP_START} .. {START} (not traded; builds the "
        f"regime-conditioned IC history the live period conditions on)\n"
        f"- **Live window**: {START} .. {END}\n"
        f"- **Execution**: signal on month-end close, fill at next-session open (t+1), "
        f"halted opens rolled forward\n"
        f"- **Costs**: {cost_model.commission_per_share}/share commission, "
        f"{cost_model.slippage_bps} bps slippage\n"
        f"- **Benchmark**: SPY"
    )
    metrics_comparison_markdown(
        metrics, OUTDIR / "summary.md",
        title="Regime-Adaptive Strategy Comparison, 2017-2025",
        preamble=preamble,
    )
    pd.DataFrame(navs).to_csv(OUTDIR / "nav.csv")
    diagnostics.to_csv(OUTDIR / "regime_adaptive_factor_weights.csv")

    print("\n" + "=" * 88)
    print("REGIME-ADAPTIVE vs STATIC MULTI-FACTOR vs PURE MOMENTUM vs SPY — 2017-2025")
    print("=" * 88)
    rows = ["Total Return", "CAGR", "Annual Volatility", "Sharpe Ratio",
            "Max Drawdown", "Calmar Ratio", "Annual Turnover"]
    comp = pd.DataFrame({label: m.to_series() for label, m in metrics.items()}).loc[rows]
    comp = comp[["Regime-Adaptive", "Static Multi-Factor", "Pure Momentum", "SPY"]]
    print(comp.to_string())
    print(f"\nArtifacts written to {OUTDIR}/")


if __name__ == "__main__":
    main()

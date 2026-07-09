"""Milestone 4: evaluate all four price factors over 2015-2025.

Runs the :mod:`quantlab.factors.evaluation` toolkit on the point-in-time S&P 500
for each of the four pure-price factors —

* 12-1 momentum         (:class:`MomentumFactor`)
* low volatility        (:class:`LowVolatilityFactor`)
* short-term reversal   (:class:`ShortTermReversalFactor`)
* Amihud illiquidity    (:class:`AmihudIlliquidityFactor`)

and writes, under ``reports/factor_eval/``:

* ``<factor>_ic.png``      — IC time series (per-period, rolling, full-sample mean)
* ``<factor>_deciles.png`` — annualised return by factor quantile
* ``factor_correlation.png`` — average cross-sectional rank-correlation heatmap
* ``summary.md`` / ``summary.csv`` — the headline table (mean IC, ICIR, IC t-stat,
  hit rate, decile L/S annualised return, monotonicity, rank autocorrelation)

The headline table is also printed to stdout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab.backtest.engine import rebalance_calendar
from quantlab.data.prices import get_prices, universe_symbols
from quantlab.data.universe import load_universe
from quantlab.factors import (
    AmihudIlliquidityFactor,
    LowVolatilityFactor,
    MomentumFactor,
    ShortTermReversalFactor,
)
from quantlab.factors.evaluation import (
    ICResult,
    build_factor_panel,
    decile_returns,
    factor_correlation,
    information_coefficient,
    rank_autocorrelation,
)
from quantlab.reports import (
    plot_decile_returns,
    plot_factor_correlation_heatmap,
    plot_ic_timeseries,
)

START = "2015-01-01"
END = "2025-12-31"
DATA_START = "2013-09-01"          # >16m history before START for the panel
N_BUCKETS = 10
OUTDIR = Path("reports/factor_eval")

FACTORS = [
    MomentumFactor(lookback=12, skip=1),
    LowVolatilityFactor(window=252),
    ShortTermReversalFactor(lookback_months=1),
    AmihudIlliquidityFactor(window=21),
]


def main() -> None:
    print("Loading universe and prices ...", flush=True)
    universe = load_universe()
    union = universe_symbols(START, END)
    long_prices = get_prices(union, DATA_START, END)
    price_tickers = set(long_prices["ticker"].unique())
    print(f"Universe union: {len(union)} symbols; priced: {len(price_tickers)}",
          flush=True)

    # Wide panel only to source a month-end rebalance calendar over the window.
    wide = get_prices(union, START, END, field="adj_close")
    rebal_dates = [
        d for d in rebalance_calendar(wide.index, "month_end")
        if pd.Timestamp(START) <= d <= pd.Timestamp(END)
    ]
    print(f"Rebalance dates: {len(rebal_dates)} "
          f"({rebal_dates[0].date()} .. {rebal_dates[-1].date()})", flush=True)

    def members_by_date(dt: pd.Timestamp):
        return set(universe.members_as_of(dt)) & price_tickers

    OUTDIR.mkdir(parents=True, exist_ok=True)
    panels = []
    rows = []
    for factor in FACTORS:
        print(f"Evaluating {factor.name} ...", flush=True)
        panel = build_factor_panel(
            factor, long_prices, rebal_dates, members_by_date,
            history_months=16,
        )
        panels.append(panel)

        ic = information_coefficient(panel)
        ic_res = ICResult.from_series(ic, factor.name)
        dec = decile_returns(panel, n=N_BUCKETS)
        autocorr = rank_autocorrelation(panel)

        # Per-factor figures.
        plot_ic_timeseries(
            ic, OUTDIR / f"{factor.name}_ic.png",
            title=f"{factor.name} — Rank IC (factor vs. next-month return)",
        )
        plot_decile_returns(
            dec.annualized, OUTDIR / f"{factor.name}_deciles.png",
            title=f"{factor.name} — Annualised Return by Decile (2015-2025)",
        )

        rows.append({
            "factor": factor.name,
            "mean_ic": ic_res.mean,
            "icir": ic_res.icir,
            "ic_t_stat": ic_res.t_stat,
            "ic_hit_rate": ic_res.hit_rate,
            "n_periods": ic_res.n,
            "ls_ann_return": dec.long_short_annualized,
            "top_decile_ann": dec.annualized.iloc[-1],
            "bottom_decile_ann": dec.annualized.iloc[0],
            "monotonicity": dec.monotonicity,
            "rank_autocorr": float(autocorr.mean()),
        })
        print("  " + ic_res.summary())
        print("  " + dec.summary())

    # Cross-factor correlation heatmap.
    corr = factor_correlation(panels)
    plot_factor_correlation_heatmap(
        corr, OUTDIR / "factor_correlation.png",
        title="Factor Rank-Correlation (avg cross-sectional Spearman ρ, 2015-2025)",
    )

    # Headline table -> CSV + Markdown + stdout.
    table = pd.DataFrame(rows).set_index("factor")
    table.to_csv(OUTDIR / "summary.csv")
    _write_markdown(table, corr, OUTDIR / "summary.md")

    print("\n" + "=" * 78)
    print("FACTOR EVALUATION SUMMARY — S&P 500 (PIT), 2015-2025")
    print("=" * 78)
    print(_headline(table))
    print(f"\nArtifacts written to {OUTDIR}/")


def _headline(table: pd.DataFrame) -> str:
    """The three headline numbers the milestone asks for, per factor."""
    show = pd.DataFrame({
        "mean IC": table["mean_ic"].map(lambda v: f"{v:+.4f}"),
        "ICIR": table["icir"].map(lambda v: f"{v:+.3f}"),
        "IC t-stat": table["ic_t_stat"].map(lambda v: f"{v:+.2f}"),
        "decile L/S ann.": table["ls_ann_return"].map(lambda v: f"{v:+.2%}"),
        "monotonicity": table["monotonicity"].map(lambda v: f"{v:+.2f}"),
    })
    return show.to_string()


def _write_markdown(table: pd.DataFrame, corr: pd.DataFrame, path: Path) -> None:
    def pct(v):
        return f"{v:+.2%}"

    lines = [
        "# Factor Evaluation — S&P 500 (point-in-time), 2015-2025",
        "",
        "Signal-quality diagnostics for the four pure-price factors. Factor values "
        "are formed point-in-time (look-ahead guard on); forward returns look "
        "strictly forward (formation close to next formation close). Rank-based, so "
        "invariant to winsorization/standardisation. This measures *signal quality*, "
        "not tradable P&L — the backtest engine (t+1 open fills, costs) is the "
        "tradable measure.",
        "",
        "## Headline",
        "",
        "| Factor | Mean IC | ICIR | IC t-stat | Hit rate | Decile L/S (ann.) "
        "| Top decile | Bottom decile | Monotonicity | Rank autocorr |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f, r in table.iterrows():
        lines.append(
            f"| {f} | {r['mean_ic']:+.4f} | {r['icir']:+.3f} | "
            f"{r['ic_t_stat']:+.2f} | {r['ic_hit_rate']:.0%} | "
            f"{pct(r['ls_ann_return'])} | {pct(r['top_decile_ann'])} | "
            f"{pct(r['bottom_decile_ann'])} | {r['monotonicity']:+.2f} | "
            f"{r['rank_autocorr']:+.2f} |"
        )

    lines += ["", "## Cross-factor rank correlation", "",
              "Average cross-sectional Spearman ρ between factor values.", ""]
    lines.append("| | " + " | ".join(corr.columns) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in corr.columns) + " |")
    for name, row in corr.iterrows():
        lines.append(f"| {name} | " +
                     " | ".join(f"{v:+.2f}" for v in row.values) + " |")

    lines += [
        "",
        "## Figures",
        "",
        "- `<factor>_ic.png` — per-period rank IC with rolling and full-sample mean",
        "- `<factor>_deciles.png` — annualised return by factor decile (monotonicity)",
        "- `factor_correlation.png` — cross-factor rank-correlation heatmap",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

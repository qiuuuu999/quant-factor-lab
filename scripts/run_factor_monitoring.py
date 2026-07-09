"""Factor health monitoring: IC decay detection + regime fit, 2015-2025.

Runs :mod:`quantlab.monitor` on the four price factors already scored in
``scripts/run_factor_evaluation.py`` —

* 12-1 momentum         (:class:`MomentumFactor`)
* low volatility        (:class:`LowVolatilityFactor`)
* short-term reversal   (:class:`ShortTermReversalFactor`)
* Amihud illiquidity    (:class:`AmihudIlliquidityFactor`)

For each factor:

1. Builds the point-in-time IC time series
   (:func:`quantlab.factors.evaluation.information_coefficient`).
2. Runs a CUSUM change-point test on it
   (:func:`quantlab.monitor.decay.factor_health_report`) to flag IC decay and
   its estimated onset date.
3. Classifies SPY into one of four market regimes
   (:func:`quantlab.monitor.regime.classify_regime`) and cross-tabulates each
   factor's IC by the regime active at each rebalance
   (:func:`quantlab.monitor.regime.factor_regime_matrix`).

Writes, under ``reports/monitor/``:

* ``<factor>_health.png`` — per-period IC, rolling mean, CUSUM alert marker
* ``regime_heatmap.png`` — factor x regime mean-IC heatmap
* ``summary.md`` — health status and best/worst regime per factor

The per-factor health summary and best/worst regime are also printed to
stdout.
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
from quantlab.factors.evaluation import build_factor_panel, information_coefficient
from quantlab.monitor.decay import factor_health_report
from quantlab.monitor.regime import classify_regime, factor_regime_matrix, regime_as_of
from quantlab.reports import plot_factor_health, plot_regime_heatmap

START = "2015-01-01"
END = "2025-12-31"
DATA_START = "2013-09-01"          # >16m history before START for the panel
CUSUM_WINDOW = 36                  # rolling-IC window, months
OUTDIR = Path("reports/monitor")

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

    wide = get_prices(union, START, END, field="adj_close")
    rebal_dates = [
        d for d in rebalance_calendar(wide.index, "month_end")
        if pd.Timestamp(START) <= d <= pd.Timestamp(END)
    ]
    print(f"Rebalance dates: {len(rebal_dates)} "
          f"({rebal_dates[0].date()} .. {rebal_dates[-1].date()})", flush=True)

    def members_by_date(dt: pd.Timestamp):
        return set(universe.members_as_of(dt)) & price_tickers

    print("Classifying SPY market regime ...", flush=True)
    spy = get_prices("SPY", DATA_START, END, field="adj_close")["SPY"]
    regime = classify_regime(spy)
    regime_at_rebal = regime_as_of(regime, rebal_dates)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ic_by_factor: dict[str, pd.Series] = {}
    reports = {}
    for factor in FACTORS:
        print(f"Monitoring {factor.name} ...", flush=True)
        panel = build_factor_panel(
            factor, long_prices, rebal_dates, members_by_date, history_months=16,
        )
        ic = information_coefficient(panel)
        ic_by_factor[factor.name] = ic

        report = factor_health_report(factor.name, ic, window=CUSUM_WINDOW)
        reports[factor.name] = report
        plot_factor_health(report, OUTDIR / f"{factor.name}_health.png")
        print("  " + report.summary())
        print("  " + report.cusum.summary())

    print("Building factor-regime fit matrix ...", flush=True)
    matrix = factor_regime_matrix(ic_by_factor, regime_at_rebal)
    plot_regime_heatmap(matrix, OUTDIR / "regime_heatmap.png")

    _write_markdown(reports, matrix, OUTDIR / "summary.md")

    print("\n" + "=" * 78)
    print("FACTOR HEALTH MONITORING — S&P 500 (PIT), 2015-2025")
    print("=" * 78)
    for name, report in reports.items():
        print(report.summary())
        best, worst = matrix.best_regime[name], matrix.worst_regime[name]
        print(f"{'':>16}  best regime: {best} ({matrix.mean_ic.loc[name, best]:+.4f})  "
              f"worst regime: {worst} ({matrix.mean_ic.loc[name, worst]:+.4f})")
    print(f"\nArtifacts written to {OUTDIR}/")


def _write_markdown(reports: dict, matrix, path: Path) -> None:
    lines = [
        "# Factor Health Monitoring — S&P 500 (point-in-time), 2015-2025",
        "",
        "CUSUM change-point test on each factor's per-period rank IC "
        f"(rolling window {CUSUM_WINDOW} months); SPY rolling volatility / "
        "200-day trend regime classification. See `docs/monitoring.md` for "
        "methodology.",
        "",
        "## Factor health",
        "",
        "| Factor | Status | Current rolling IC | Historical mean IC | "
        "Alert date | CUSUM stat | Critical value |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, report in reports.items():
        status = "DECAY ALERT" if report.decay_alert else "OK"
        alert = report.alert_date.date() if report.alert_date is not None else "—"
        lines.append(
            f"| {name} | {status} | {report.current_rolling_ic:+.4f} | "
            f"{report.historical_mean_ic:+.4f} | {alert} | "
            f"{report.cusum.statistic:.2f} | {report.cusum.critical_value:.2f} |"
        )

    lines += ["", "## Factor – regime fit (mean IC)", "",
              "| Factor | " + " | ".join(matrix.mean_ic.columns) + " | Best | Worst |",
              "| --- | " + " | ".join("---" for _ in matrix.mean_ic.columns) + " | --- | --- |"]
    for name, row in matrix.mean_ic.iterrows():
        cells = " | ".join(f"{v:+.4f}" for v in row.values)
        lines.append(
            f"| {name} | {cells} | {matrix.best_regime[name]} | {matrix.worst_regime[name]} |"
        )

    lines += ["", "## Figures", "",
              "- `<factor>_health.png` — per-period IC, rolling mean, CUSUM alert marker",
              "- `regime_heatmap.png` — factor x regime mean-IC heatmap",
              ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

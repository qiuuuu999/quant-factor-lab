"""Risk attribution demo: the 12-1 momentum strategy's monthly holdings, 2015-2025.

Reuses the exact top-decile momentum selection from
``scripts/run_momentum_backtest.py`` (point-in-time S&P 500, equal-weight top
20% by 12-1 momentum, rebalanced monthly) as the portfolio under study, then:

1. Builds winsorized, z-scored exposures to all four library factors --
   12-1 momentum, low volatility, 1-month reversal, Amihud illiquidity -- for
   every name in the universe at every rebalance
   (:func:`quantlab.risk.factor_risk.build_exposure_panel`).
2. Fits a :class:`~quantlab.risk.factor_risk.FactorRiskModel`: a per-period
   cross-sectional regression of forward returns on those exposures gives a
   factor-return time series, whose sample covariance is the factor
   covariance (only 4 factors and ~130 monthly observations here, so ``T >>
   K`` and the plain sample estimator is already well-conditioned --
   ``LedoitWolfCovariance`` is the right tool when the cross-section is large
   relative to history, e.g. an asset-level covariance across hundreds of
   names; see ``docs/risk.md`` and ``tests/test_risk.py`` for that regime).
3. Decomposes the strategy's variance, at its most recent rebalance, into
   factor vs. specific components, and profiles its net factor exposure.

Outputs (under ``reports/risk_attribution/``): the exposure-profile bar chart,
the risk-decomposition bar chart, and a Markdown summary. The exposure profile
and decomposition are also printed to stdout.
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
    deciles,
)
from quantlab.reports import plot_exposure_profile, plot_risk_decomposition
from quantlab.risk.attribution import ExposureProfile, portfolio_factor_exposure
from quantlab.risk.factor_risk import FactorRiskModel, build_exposure_panel

START = "2015-01-01"
END = "2025-12-31"
DATA_START = "2013-09-01"          # >16m history before START for the panel
TOP_DECILES = (9, 10)              # top 20%, mirrors run_momentum_backtest.py
OUTDIR = Path("reports/risk_attribution")

FACTORS = [
    MomentumFactor(lookback=12, skip=1),
    LowVolatilityFactor(window=252),
    ShortTermReversalFactor(lookback_months=1),
    AmihudIlliquidityFactor(window=21),
]


def build_momentum_weights(long_prices, rebal_dates, universe, price_tickers):
    """Target weights per rebalance: equal-weight the top-decile momentum names.

    Identical selection rule to ``scripts/run_momentum_backtest.py`` -- this
    script studies that strategy's risk, not a different one.
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


def main() -> None:
    print("Loading universe and prices ...", flush=True)
    universe = load_universe()
    union = universe_symbols(START, END)
    long_prices = get_prices(union, DATA_START, END)
    price_tickers = set(long_prices["ticker"].unique())
    wide = get_prices(union, START, END, field="adj_close")
    print(f"Universe union: {len(union)} symbols; priced: {len(price_tickers)}",
          flush=True)

    rebal_dates = [
        d for d in rebalance_calendar(wide.index, "month_end")
        if pd.Timestamp(START) <= d <= pd.Timestamp(END)
    ]
    print(f"Rebalance dates: {len(rebal_dates)} "
          f"({rebal_dates[0].date()} .. {rebal_dates[-1].date()})", flush=True)

    def members_by_date(dt: pd.Timestamp):
        return set(universe.members_as_of(dt)) & price_tickers

    print("Building momentum-strategy weights ...", flush=True)
    weights_by_date = build_momentum_weights(long_prices, rebal_dates, universe, price_tickers)

    print("Building factor exposure panel (4 factors) ...", flush=True)
    exposures_by_date, forward_returns = build_exposure_panel(
        FACTORS, long_prices, rebal_dates, members_by_date, history_months=16,
    )

    print("Fitting factor risk model (cross-sectional regression per period) ...",
          flush=True)
    model = FactorRiskModel().fit(exposures_by_date, forward_returns)
    factor_cov = model.factor_covariance()          # T=~130 >> K=4: sample is fine
    specific_var = model.specific_variance()

    # Full-history time series of the strategy's own factor exposure (context).
    exposure_ts = pd.DataFrame({
        dt: portfolio_factor_exposure(pd.Series(w), exposures_by_date[dt])
        for dt, w in weights_by_date.items() if dt in exposures_by_date
    }).T.sort_index()

    # Snapshot at the most recent rebalance for the headline profile + decomposition.
    as_of = max(dt for dt in weights_by_date if dt in exposures_by_date)
    weights = pd.Series(weights_by_date[as_of])
    exposures = exposures_by_date[as_of]

    profile = ExposureProfile.build(weights, exposures, as_of)
    decomposition = model.decompose(weights, exposures)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    plot_exposure_profile(
        profile, OUTDIR / "exposure_profile.png",
        title="12-1 Momentum Strategy — Factor Exposure",
    )
    plot_risk_decomposition(
        decomposition, OUTDIR / "risk_decomposition.png",
        title="12-1 Momentum Strategy — Risk Decomposition",
    )
    exposure_ts.to_csv(OUTDIR / "exposure_timeseries.csv")
    _write_markdown(profile, decomposition, exposure_ts, factor_cov, OUTDIR / "summary.md")

    print("\n" + "=" * 60)
    print("RISK ATTRIBUTION — 12-1 MOMENTUM STRATEGY, 2015-2025")
    print("=" * 60)
    print(profile.summary())
    print()
    print(decomposition.summary())
    print(f"\nArtifacts written to {OUTDIR}/")


def _write_markdown(profile, decomposition, exposure_ts, factor_cov, path: Path) -> None:
    avg_exposure = exposure_ts.mean()
    lines = [
        "# Risk Attribution — 12-1 Momentum Strategy, 2015-2025",
        "",
        "- **Portfolio**: point-in-time S&P 500, equal-weight top 20% by 12-1 "
        "momentum, monthly rebalance (identical selection to `reports/momentum_12_1/`)",
        "- **Risk model**: 4-factor cross-sectional regression (momentum, low-vol, "
        "reversal, liquidity exposures; sample covariance of the fitted factor returns)",
        f"- **Snapshot date**: {profile.as_of.date()} ({profile.n_names} names held)",
        "",
        "## Factor exposure profile (snapshot)",
        "",
        "| Factor | Exposure (σ) |",
        "| --- | --- |",
    ]
    for name, value in profile.exposure.items():
        lines.append(f"| {name} | {value:+.2f} |")

    lines += [
        "",
        "## Factor exposure profile (average over 2015-2025 rebalances)",
        "",
        "| Factor | Avg. exposure (σ) |",
        "| --- | --- |",
    ]
    for name, value in avg_exposure.items():
        lines.append(f"| {name} | {value:+.2f} |")

    lines += [
        "",
        "## Risk decomposition (snapshot)",
        "",
        f"- Total variance: {decomposition.total_variance:.6f}",
        f"- Factor variance: {decomposition.factor_variance:.6f} "
        f"({decomposition.factor_variance_pct:.1%})",
        f"- Specific variance: {decomposition.specific_variance:.6f} "
        f"({decomposition.specific_variance_pct:.1%})",
        "",
        "| Factor | Contribution | Share of factor variance |",
        "| --- | --- | --- |",
    ]
    for name, value in decomposition.factor_contributions.items():
        share = value / decomposition.factor_variance if decomposition.factor_variance else float("nan")
        lines.append(f"| {name} | {value:+.6f} | {share:.1%} |")

    lines += [
        "",
        "## Factor covariance matrix (monthly)",
        "",
        "| | " + " | ".join(factor_cov.columns) + " |",
        "| --- | " + " | ".join("---" for _ in factor_cov.columns) + " |",
    ]
    for name, row in factor_cov.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"{v:.5f}" for v in row.values) + " |")

    lines += ["", "## Figures", "",
              "- `exposure_profile.png` — snapshot factor exposure (std. dev.)",
              "- `risk_decomposition.png` — factor vs. specific variance split",
              ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

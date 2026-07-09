"""Tests for quantlab.portfolio.regime_adaptive.

Hermetic. Two families of test matter most here:

1. **Correctness of the switching logic** — synthetic factors with a known,
   engineered regime-dependent IC should produce composite weights that
   favor the right factor in the right regime.
2. **Look-ahead protection** — the whole point of this module is that a
   rebalance decision at date ``t`` must be reproducible from data available
   at ``t`` alone. Both the expanding-median regime classification and the
   regime-conditioned IC weighting are checked directly: mutating data dated
   *after* the decision date must not change the decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.monitor.regime import classify_regime
from quantlab.portfolio.regime_adaptive import (
    build_regime_adaptive_weights,
    build_static_multifactor_weights,
    composite_score,
    pit_regime_by_date,
    regime_conditioned_factor_weights,
)

# --------------------------------------------------------------------------- #
# Look-ahead protection: expanding regime classification
# --------------------------------------------------------------------------- #

def _segment(n: int, drift: float, vol: float, start_price: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    return start_price * np.cumprod(1.0 + rets)


def _synthetic_benchmark(n_per_segment: int = 300) -> pd.Series:
    """Four consecutive segments: calm-up, calm-down, turbulent-up, turbulent-down."""
    segments = [
        (0.0015, 0.001),
        (-0.0015, 0.001),
        (0.0015, 0.02),
        (-0.0015, 0.02),
    ]
    prices, price = [], 100.0
    for i, (drift, vol) in enumerate(segments):
        seg = _segment(n_per_segment, drift, vol, price, seed=i)
        prices.append(seg)
        price = seg[-1]
    values = np.concatenate(prices)
    dates = pd.bdate_range("2010-01-01", periods=len(values))
    return pd.Series(values, index=dates)


def test_expanding_classification_unaffected_by_future_data():
    full = _synthetic_benchmark()
    cutoff = full.index[700]
    truncated = full.loc[:cutoff]

    full_labels = classify_regime(full, expanding=True)
    truncated_labels = classify_regime(truncated, expanding=True)

    common = truncated_labels.index
    # Labels through the cutoff must be identical whether or not the series
    # is later extended -- the defining look-ahead-safety property.
    pd.testing.assert_series_equal(
        full_labels.reindex(common), truncated_labels.reindex(common)
    )


def test_expanding_classification_differs_from_full_sample_median():
    # The whole point of `expanding=True`: early labels can differ from the
    # full-sample-median classification once a later, very different regime
    # is folded into the full-sample threshold.
    full = _synthetic_benchmark()
    retrospective = classify_regime(full, expanding=False)
    pit = classify_regime(full, expanding=True)
    # Not asserting exact disagreement counts (that's a property of the
    # synthetic data, not the code), just that the two modes are not
    # trivially identical -- i.e. `expanding` actually changes behaviour.
    assert not retrospective.equals(pit)


def test_pit_regime_by_date_matches_expanding_classification_forward_filled():
    bench = _synthetic_benchmark()
    dates = list(bench.index[::50])
    mapped = pit_regime_by_date(bench, dates)
    daily = classify_regime(bench, expanding=True)
    expected = daily.reindex(daily.index.union(dates)).sort_index().ffill().reindex(dates).dropna()
    assert mapped.index.equals(expected.index)
    assert (mapped.to_numpy() == expected.to_numpy()).all()


# --------------------------------------------------------------------------- #
# Look-ahead protection: regime-conditioned factor weights
# --------------------------------------------------------------------------- #

def _ic_series(values: list[float], dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(values, index=dates)


def test_regime_conditioned_weights_ignore_future_ic_observations():
    dates = pd.date_range("2015-01-31", periods=20, freq="ME")
    regime_by_date = pd.Series(["low_vol_up"] * 20, index=dates)
    ic = _ic_series([0.05] * 10 + [0.05] * 10, dates)   # placeholder, mutated below
    as_of = dates[10]

    ic_by_factor = {"A": ic.copy()}
    weights_before = regime_conditioned_factor_weights(
        ic_by_factor, regime_by_date, as_of, "low_vol_up", min_obs=3
    )

    # Corrupt every observation strictly after `as_of` with extreme, opposite-sign
    # values -- if the function is look-ahead-safe this must not move the result.
    ic_by_factor["A"].loc[ic_by_factor["A"].index >= as_of] = -99.0
    weights_after = regime_conditioned_factor_weights(
        ic_by_factor, regime_by_date, as_of, "low_vol_up", min_obs=3
    )

    pd.testing.assert_series_equal(weights_before, weights_after)


def test_regime_conditioned_weights_exclude_current_period_own_ic():
    # ic.loc[as_of] itself must never be used (its forward return isn't
    # realized yet at decision time) -- setting it to an extreme value must
    # not change the result versus leaving it NaN.
    dates = pd.date_range("2015-01-31", periods=10, freq="ME")
    regime_by_date = pd.Series(["low_vol_up"] * 10, index=dates)
    as_of = dates[8]

    base = _ic_series([0.02] * 10, dates)
    ic_by_factor_a = {"A": base.copy()}
    ic_by_factor_b = {"A": base.copy()}
    ic_by_factor_b["A"].loc[as_of] = 999.0

    wa = regime_conditioned_factor_weights(ic_by_factor_a, regime_by_date, as_of, "low_vol_up", min_obs=3)
    wb = regime_conditioned_factor_weights(ic_by_factor_b, regime_by_date, as_of, "low_vol_up", min_obs=3)
    pd.testing.assert_series_equal(wa, wb)


# --------------------------------------------------------------------------- #
# Switching logic correctness
# --------------------------------------------------------------------------- #

def test_weights_favor_the_factor_that_historically_worked_in_current_regime():
    dates = pd.date_range("2015-01-31", periods=20, freq="ME")
    # Alternate regimes so both accumulate history.
    regimes = ["low_vol_up" if i % 2 == 0 else "high_vol_down" for i in range(20)]
    regime_by_date = pd.Series(regimes, index=dates)

    # Factor A: strong positive IC in low_vol_up, ~zero in high_vol_down.
    # Factor B: strong positive IC in high_vol_down, ~zero in low_vol_up.
    ic_a = pd.Series(
        [0.08 if r == "low_vol_up" else 0.00 for r in regimes], index=dates
    )
    ic_b = pd.Series(
        [0.00 if r == "low_vol_up" else 0.08 for r in regimes], index=dates
    )
    ic_by_factor = {"A": ic_a, "B": ic_b}

    as_of = dates[15]   # plenty of prior history accumulated by here
    w_low_vol_up = regime_conditioned_factor_weights(
        ic_by_factor, regime_by_date, as_of, "low_vol_up", min_obs=3
    )
    w_high_vol_down = regime_conditioned_factor_weights(
        ic_by_factor, regime_by_date, as_of, "high_vol_down", min_obs=3
    )

    assert w_low_vol_up["A"] > w_low_vol_up["B"]
    assert w_high_vol_down["B"] > w_high_vol_down["A"]


def test_negative_handling_invert_keeps_signed_weight():
    dates = pd.date_range("2015-01-31", periods=10, freq="ME")
    regime_by_date = pd.Series(["low_vol_up"] * 10, index=dates)
    ic = pd.Series([-0.05] * 10, index=dates)
    as_of = dates[8]

    w = regime_conditioned_factor_weights(
        {"A": ic}, regime_by_date, as_of, "low_vol_up",
        negative_handling="invert", min_obs=3,
    )
    assert w["A"] < 0.0
    assert w["A"] == pytest.approx(-1.0)   # only factor -> normalized to -1 in abs value


def test_negative_handling_zero_drops_negative_factor():
    dates = pd.date_range("2015-01-31", periods=10, freq="ME")
    regime_by_date = pd.Series(["low_vol_up"] * 10, index=dates)
    ic_neg = pd.Series([-0.05] * 10, index=dates)
    ic_pos = pd.Series([0.03] * 10, index=dates)
    as_of = dates[8]

    w = regime_conditioned_factor_weights(
        {"A": ic_neg, "B": ic_pos}, regime_by_date, as_of, "low_vol_up",
        negative_handling="zero", min_obs=3,
    )
    assert w["A"] == 0.0
    assert w["B"] == pytest.approx(1.0)


def test_insufficient_history_falls_back_to_equal_weight():
    dates = pd.date_range("2015-01-31", periods=4, freq="ME")
    regime_by_date = pd.Series(["low_vol_up"] * 4, index=dates)
    ic = pd.Series([0.05, 0.04, 0.03, 0.02], index=dates)
    as_of = dates[2]   # only 2 prior observations, below min_obs=6

    w = regime_conditioned_factor_weights(
        {"A": ic, "B": ic * -1}, regime_by_date, as_of, "low_vol_up", min_obs=6,
    )
    assert w["A"] == pytest.approx(0.5)
    assert w["B"] == pytest.approx(0.5)
    assert w.sum() == pytest.approx(1.0)


def test_invalid_negative_handling_raises():
    dates = pd.date_range("2015-01-31", periods=4, freq="ME")
    regime_by_date = pd.Series(["low_vol_up"] * 4, index=dates)
    ic = pd.Series([0.05] * 4, index=dates)
    with pytest.raises(ValueError):
        regime_conditioned_factor_weights(
            {"A": ic}, regime_by_date, dates[3], "low_vol_up", negative_handling="bogus"
        )


# --------------------------------------------------------------------------- #
# composite_score
# --------------------------------------------------------------------------- #

def test_composite_score_is_weighted_sum_of_exposures():
    exposures = pd.DataFrame(
        {"mom": [1.0, -1.0, 0.5], "vol": [0.0, 2.0, -1.0]}, index=["A", "B", "C"]
    )
    weights = pd.Series({"mom": 0.7, "vol": -0.3})
    score = composite_score(exposures, weights)
    expected = exposures["mom"] * 0.7 + exposures["vol"] * -0.3
    pd.testing.assert_series_equal(score, expected, check_names=False)


def test_composite_score_ignores_unmatched_factor_names():
    exposures = pd.DataFrame({"mom": [1.0, 2.0]}, index=["A", "B"])
    weights = pd.Series({"mom": 1.0, "unrelated": 5.0})
    score = composite_score(exposures, weights)
    pd.testing.assert_series_equal(score, exposures["mom"], check_names=False)


# --------------------------------------------------------------------------- #
# End-to-end weight construction (small synthetic universe)
# --------------------------------------------------------------------------- #

def _synthetic_exposures_by_date(dates, tickers, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for dt in dates:
        out[dt] = pd.DataFrame(
            {"mom": rng.normal(0, 1, len(tickers)), "vol": rng.normal(0, 1, len(tickers))},
            index=tickers,
        )
    return out


def test_build_static_multifactor_weights_selects_top_bucket_equal_weight():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    tickers = [f"T{i}" for i in range(20)]
    exposures_by_date = _synthetic_exposures_by_date(dates, tickers)

    weights = build_static_multifactor_weights(
        exposures_by_date, dates, n_buckets=4, top_buckets=(4,)
    )
    assert set(weights.keys()) == set(dates)
    for dt, w in weights.items():
        assert len(w) == 5   # top quartile of 20 names
        assert sum(w.values()) == pytest.approx(1.0)
        assert all(v == pytest.approx(1.0 / 5) for v in w.values())


def test_build_regime_adaptive_weights_end_to_end_wires_regime_and_ic():
    dates = pd.date_range("2015-01-31", periods=24, freq="ME")
    tickers = [f"T{i}" for i in range(20)]
    exposures_by_date = _synthetic_exposures_by_date(dates, tickers)

    regimes = ["low_vol_up" if i % 2 == 0 else "high_vol_down" for i in range(24)]
    regime_by_date = pd.Series(regimes, index=dates)
    ic_mom = pd.Series([0.06 if r == "low_vol_up" else 0.0 for r in regimes], index=dates)
    ic_vol = pd.Series([0.0 if r == "low_vol_up" else 0.06 for r in regimes], index=dates)
    ic_by_factor = {"mom": ic_mom, "vol": ic_vol}

    weights_by_date, diagnostics = build_regime_adaptive_weights(
        exposures_by_date, ic_by_factor, regime_by_date, dates,
        min_obs=3, n_buckets=4, top_buckets=(4,),
    )

    # Every date has enough history except the first couple of rebalances
    # (min_obs=3 regime-matched observations needed).
    assert len(weights_by_date) > 15
    assert not diagnostics.empty
    assert list(diagnostics.columns) == ["mom", "vol"]

    # In a low_vol_up rebalance late enough to have accumulated history,
    # 'mom' should dominate the combination weight; in high_vol_down, 'vol'
    # should dominate -- confirms the regime label actually drives the mix.
    late_low_vol_up = [d for d, r in zip(dates, regimes) if r == "low_vol_up"][-1]
    late_high_vol_down = [d for d, r in zip(dates, regimes) if r == "high_vol_down"][-1]
    assert diagnostics.loc[late_low_vol_up, "mom"] > diagnostics.loc[late_low_vol_up, "vol"]
    assert diagnostics.loc[late_high_vol_down, "vol"] > diagnostics.loc[late_high_vol_down, "mom"]


def test_build_regime_adaptive_weights_skips_dates_without_known_regime():
    dates = pd.date_range("2015-01-31", periods=5, freq="ME")
    tickers = [f"T{i}" for i in range(20)]
    exposures_by_date = _synthetic_exposures_by_date(dates, tickers)

    # Regime known for only the first 3 dates.
    regime_by_date = pd.Series(["low_vol_up"] * 3, index=dates[:3])
    ic_by_factor = {"mom": pd.Series([0.05] * 5, index=dates)}

    weights_by_date, _ = build_regime_adaptive_weights(
        exposures_by_date, ic_by_factor, regime_by_date, dates, min_obs=1,
    )
    assert set(weights_by_date.keys()) <= set(dates[:3])

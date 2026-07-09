"""Tests for the monitor module (quantlab.monitor).

Hermetic: the CUSUM change-point test is checked against a synthetic IC series
with a known, engineered break (positive IC regime -> negative IC regime), and
regime classification is checked against synthetic price paths with known,
engineered volatility/trend segments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.monitor.decay import (
    cusum_test,
    factor_health_report,
    rolling_ic,
)
from quantlab.monitor.regime import (
    REGIMES,
    classify_regime,
    factor_regime_matrix,
    regime_as_of,
)

# --------------------------------------------------------------------------- #
# CUSUM change-point detection
# --------------------------------------------------------------------------- #

def _synthetic_decaying_ic(n_before: int = 60, n_after: int = 60, seed: int = 0) -> pd.Series:
    """IC series that is healthy for ``n_before`` periods then decays."""
    rng = np.random.default_rng(seed)
    before = rng.normal(0.05, 0.02, n_before)
    after = rng.normal(-0.03, 0.02, n_after)
    values = np.concatenate([before, after])
    dates = pd.date_range("2015-01-31", periods=len(values), freq="ME")
    return pd.Series(values, index=dates, name="ic")


def test_cusum_detects_break_near_true_change_point():
    n_before = 60
    ic = _synthetic_decaying_ic(n_before=n_before, n_after=60)
    result = cusum_test(ic, confidence=0.95)

    assert result.triggered
    assert result.is_decay
    # Estimated break should land within a handful of periods of the true one.
    true_break = ic.index[n_before - 1]
    assert abs((result.change_point - true_break).days) <= 6 * 31


def test_cusum_no_break_on_stationary_series():
    rng = np.random.default_rng(1)
    values = rng.normal(0.03, 0.02, 120)
    dates = pd.date_range("2010-01-31", periods=len(values), freq="ME")
    ic = pd.Series(values, index=dates)

    result = cusum_test(ic, confidence=0.95)
    assert not result.triggered
    assert not result.is_decay


def test_cusum_improving_series_is_not_flagged_as_decay():
    # Break exists (negative -> positive) but is an *improvement*, not decay.
    rng = np.random.default_rng(2)
    before = rng.normal(-0.04, 0.02, 60)
    after = rng.normal(0.05, 0.02, 60)
    values = np.concatenate([before, after])
    dates = pd.date_range("2015-01-31", periods=len(values), freq="ME")
    ic = pd.Series(values, index=dates)

    result = cusum_test(ic, confidence=0.95)
    assert result.triggered
    assert not result.is_decay
    assert result.mean_after > result.mean_before


def test_cusum_handles_insufficient_data():
    ic = pd.Series([0.01], index=pd.date_range("2020-01-31", periods=1, freq="ME"))
    result = cusum_test(ic)
    assert not result.triggered
    assert result.change_point is None


def test_rolling_ic_matches_manual_rolling_mean():
    values = np.linspace(0.0, 0.1, 50)
    dates = pd.date_range("2018-01-31", periods=50, freq="ME")
    ic = pd.Series(values, index=dates)

    roll = rolling_ic(ic, window=12, min_periods=12)
    expected = ic.rolling(12, min_periods=12).mean()
    pd.testing.assert_series_equal(roll, expected, check_names=False)


def test_factor_health_report_flags_decay_and_reports_current_vs_historical():
    ic = _synthetic_decaying_ic(n_before=60, n_after=60)
    report = factor_health_report("mom", ic, window=24)

    assert report.decay_alert
    assert report.alert_date is not None
    # Current (recent) rolling IC should sit well below the full-history mean
    # once the decay has taken hold.
    assert report.current_rolling_ic < report.historical_mean_ic


# --------------------------------------------------------------------------- #
# Regime classification
# --------------------------------------------------------------------------- #

def _segment(n: int, drift: float, vol: float, start_price: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    return start_price * np.cumprod(1.0 + rets)


def _synthetic_regime_prices() -> tuple[pd.Series, dict[str, tuple[int, int]]]:
    """Four consecutive 400-day segments, one per regime, low->low->high->high vol."""
    n = 400
    segments = [
        ("low_vol_up", 0.0015, 0.001),
        ("low_vol_down", -0.0015, 0.001),
        ("high_vol_up", 0.0015, 0.02),
        ("high_vol_down", -0.0015, 0.02),
    ]
    prices = []
    bounds: dict[str, tuple[int, int]] = {}
    price = 100.0
    cursor = 0
    for i, (label, drift, vol) in enumerate(segments):
        seg = _segment(n, drift, vol, price, seed=i)
        prices.append(seg)
        bounds[label] = (cursor, cursor + n)
        cursor += n
        price = seg[-1]

    values = np.concatenate(prices)
    dates = pd.bdate_range("2010-01-01", periods=len(values))
    return pd.Series(values, index=dates), bounds


def test_classify_regime_recovers_engineered_segments():
    prices, bounds = _synthetic_regime_prices()
    regime = classify_regime(prices, vol_window=21, trend_window=200)

    # Check the tail of each segment (well past the 200-day trend warm-up so
    # the moving average has caught up with that segment's own drift).
    for label, (lo, hi) in bounds.items():
        window_dates = prices.index[hi - 50 : hi]
        labels = regime.reindex(window_dates).dropna()
        assert len(labels) > 0
        match_rate = (labels == label).mean()
        assert match_rate >= 0.9, f"{label}: only {match_rate:.0%} matched"


def test_classify_regime_drops_warmup_period():
    prices, _ = _synthetic_regime_prices()
    regime = classify_regime(prices, vol_window=21, trend_window=200)
    assert regime.index.min() > prices.index[198]


def test_regime_as_of_forward_fills_to_nearest_prior_day():
    dates = pd.bdate_range("2020-01-01", periods=10)
    regime = pd.Series(["low_vol_up"] * 5 + ["high_vol_down"] * 5, index=dates)

    query_dates = [dates[2], dates[7], dates[-1] + pd.Timedelta(days=3)]
    mapped = regime_as_of(regime, query_dates)

    assert mapped.loc[dates[2]] == "low_vol_up"
    assert mapped.loc[dates[7]] == "high_vol_down"
    assert mapped.iloc[-1] == "high_vol_down"  # ffilled past the last known day


def test_factor_regime_matrix_reports_best_and_worst_regime():
    dates = pd.date_range("2020-01-31", periods=8, freq="ME")
    regime_by_date = pd.Series(
        ["low_vol_up", "low_vol_up", "low_vol_down", "low_vol_down",
         "high_vol_up", "high_vol_up", "high_vol_down", "high_vol_down"],
        index=dates,
    )
    # Factor A: great in low_vol_up, terrible in high_vol_down.
    ic_a = pd.Series([0.10, 0.08, 0.01, 0.02, -0.01, 0.00, -0.10, -0.08], index=dates)
    ic_by_factor = {"A": ic_a}

    matrix = factor_regime_matrix(ic_by_factor, regime_by_date)

    assert list(matrix.mean_ic.columns) == REGIMES
    assert matrix.best_regime["A"] == "low_vol_up"
    assert matrix.worst_regime["A"] == "high_vol_down"
    assert matrix.counts.loc["A", "low_vol_up"] == 2


def test_factor_regime_matrix_drops_unmatched_dates():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    regime_by_date = pd.Series(["low_vol_up", "low_vol_up"], index=dates[:2])
    ic = pd.Series([0.05, 0.04, 0.03, 0.02], index=dates)

    matrix = factor_regime_matrix({"A": ic}, regime_by_date)
    assert matrix.counts.loc["A", "low_vol_up"] == 2
    assert matrix.counts.loc["A"].sum() == 2

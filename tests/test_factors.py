"""Tests for the factor framework, momentum factor, and preprocessing utils.

Hermetic: momentum correctness is asserted against hand-built price series with
known closed-form answers, and the look-ahead guard is checked directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.factors import (
    Factor,
    LookaheadBiasError,
    MomentumFactor,
    deciles,
    winsorize,
    zscore,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _monthly_long(prices_by_ticker: dict[str, list[float]], end="2020-06-30"):
    """Build a long price frame with month-end dates ending at `end`.

    Each ticker maps to a list of adjusted closes (oldest first).
    """
    n = max(len(v) for v in prices_by_ticker.values())
    dates = pd.date_range(end=end, periods=n, freq="ME")
    rows = []
    for ticker, series in prices_by_ticker.items():
        pad = n - len(series)
        for i, px in enumerate(series):
            rows.append({"date": dates[pad + i], "ticker": ticker,
                         "adj_close": px, "close": px, "volume": 1_000})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Momentum correctness (closed-form)
# --------------------------------------------------------------------------- #

def test_momentum_one_percent_per_month():
    # Price rises exactly 1% each month. 12-1 momentum spans months t-12..t-1,
    # i.e. 11 monthly steps of +1% => (1.01 ** 11) - 1.
    n = 14
    series = [100.0 * (1.01 ** i) for i in range(n)]
    prices = _monthly_long({"UP": series}, end="2020-06-30")

    mom = MomentumFactor().compute(prices, ["UP"], "2020-06-30")
    expected = 1.01 ** 11 - 1.0
    assert mom["UP"] == pytest.approx(expected, rel=1e-9)


def test_momentum_excludes_most_recent_month():
    # A huge spike in the SKIPPED most-recent month must not affect 12-1 momentum.
    n = 14
    series = [100.0 * (1.01 ** i) for i in range(n)]
    spiked = series.copy()
    spiked[-1] = series[-1] * 1.5          # +50% in the excluded last month
    base = MomentumFactor().compute(_monthly_long({"X": series}), ["X"], "2020-06-30")
    with_spike = MomentumFactor().compute(_monthly_long({"X": spiked}), ["X"], "2020-06-30")
    assert base["X"] == pytest.approx(with_spike["X"], rel=1e-12)


def test_momentum_insufficient_history_is_nan():
    series = [100.0 * (1.01 ** i) for i in range(10)]   # < lookback + 1 = 13 months
    mom = MomentumFactor().compute(_monthly_long({"NEW": series}), ["NEW"], "2020-06-30")
    assert np.isnan(mom["NEW"])


def test_momentum_missing_ticker_is_nan_not_error():
    series = [100.0 * (1.01 ** i) for i in range(14)]
    prices = _monthly_long({"UP": series})
    mom = MomentumFactor().compute(prices, ["UP", "ABSENT"], "2020-06-30")
    assert set(mom.index) == {"UP", "ABSENT"}       # reindexed to full universe
    assert np.isnan(mom["ABSENT"])


def test_momentum_ranks_faster_growth_higher():
    fast = [100.0 * (1.02 ** i) for i in range(14)]
    slow = [100.0 * (1.005 ** i) for i in range(14)]
    prices = _monthly_long({"FAST": fast, "SLOW": slow})
    mom = MomentumFactor().compute(prices, ["FAST", "SLOW"], "2020-06-30")
    assert mom["FAST"] > mom["SLOW"] > 0


# --------------------------------------------------------------------------- #
# Look-ahead guard
# --------------------------------------------------------------------------- #

def test_lookahead_guard_intercepts_future_data():
    series = [100.0 * (1.01 ** i) for i in range(14)]
    # Frame extends to 2020-07-31, but we ask for a factor as of 2020-06-30.
    prices = _monthly_long({"UP": series}, end="2020-07-31")
    with pytest.raises(LookaheadBiasError):
        MomentumFactor().compute(prices, ["UP"], "2020-06-30")


def test_lookahead_guard_can_be_disabled_and_slices():
    # With enforcement off, future rows are still sliced away (defense in depth),
    # so the answer matches a frame that never contained them.
    n = 14
    series = [100.0 * (1.01 ** i) for i in range(n)]
    clean = _monthly_long({"UP": series}, end="2020-06-30")
    dirty = _monthly_long({"UP": series + [999.0]}, end="2020-07-31")

    a = MomentumFactor().compute(clean, ["UP"], "2020-06-30")
    b = MomentumFactor().compute(dirty, ["UP"], "2020-06-30", enforce_pit=False)
    assert a["UP"] == pytest.approx(b["UP"], rel=1e-12)


def test_compute_requires_expected_columns():
    bad = pd.DataFrame({"date": pd.to_datetime(["2020-01-31"]), "px": [1.0]})
    with pytest.raises(ValueError):
        MomentumFactor().compute(bad, ["X"], "2020-06-30")


def test_factor_base_reindexes_and_names_series():
    class ConstFactor(Factor):
        name = "const"

        def _compute(self, prices, universe, as_of_date):
            # Only returns a value for the first ticker.
            return pd.Series({universe[0]: 42.0})

    prices = _monthly_long({"A": [1.0, 2.0], "B": [1.0, 2.0]})
    out = ConstFactor().compute(prices, ["A", "B", "C"], "2020-06-30")
    assert out.name == "const"
    assert list(out.index) == ["A", "B", "C"]
    assert out["A"] == 42.0 and np.isnan(out["B"]) and np.isnan(out["C"])


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

def test_winsorize_clips_tails():
    s = pd.Series([-100.0] + list(range(1, 99)) + [100.0], dtype=float)
    w = winsorize(s, 0.01, 0.99)
    assert w.max() < 100.0 and w.min() > -100.0
    # Values inside the range are unchanged.
    assert w.iloc[50] == s.iloc[50]


def test_winsorize_preserves_nan():
    s = pd.Series([1.0, np.nan, 2.0, 3.0, 100.0])
    w = winsorize(s, 0.0, 1.0)
    assert np.isnan(w.iloc[1])


def test_zscore_mean_zero_unit_std():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore(s)
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std(ddof=1) == pytest.approx(1.0, abs=1e-12)


def test_zscore_handles_zero_variance():
    s = pd.Series([5.0, 5.0, 5.0])
    z = zscore(s)
    assert (z == 0.0).all()


def test_deciles_bucket_counts_and_order():
    s = pd.Series(np.arange(100, dtype=float))
    d = deciles(s, 10)
    assert set(d.unique()) == set(range(1, 11))
    assert (d.value_counts() == 10).all()          # equal-count buckets
    # Highest raw values land in the top bucket.
    assert d.iloc[-1] == 10 and d.iloc[0] == 1


def test_deciles_preserves_nan():
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    d = deciles(s, 2)
    assert np.isnan(d.iloc[2])
    assert set(d.dropna().unique()) <= {1.0, 2.0}

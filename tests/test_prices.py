"""Tests for the price data pipeline (quantlab.data.prices).

Fully hermetic: the yfinance network call is replaced by fake fetchers that
return yfinance-shaped frames, so download/retry/validation/storage logic is
exercised deterministically without touching the network.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from quantlab.data import prices as P
from quantlab.data.prices import (
    QualityReport,
    TickerQuality,
    download_prices,
    get_prices,
    universe_symbols,
    validate_ticker,
)


# --------------------------------------------------------------------------- #
# Helpers: build yfinance-shaped frames
# --------------------------------------------------------------------------- #

def _ticker_frame(n=12, start="2021-01-04", base=100.0):
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(np.linspace(base, base + n - 1, n), index=idx)
    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 1,
            "Low": close - 2,
            "Close": close,
            "Adj Close": close * 0.99,
            "Volume": pd.Series(np.full(n, 1_000_000), index=idx),
        },
        index=idx,
    )


def _multi_raw(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-ticker frames into a (ticker, field) MultiIndex frame."""
    return pd.concat(frames.values(), axis=1, keys=frames.keys())


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_validate_flags_missing_zero_volume_and_jumps():
    df = _ticker_frame(n=10)
    # Inject issues.
    df.iloc[3, df.columns.get_loc("Adj Close")] = np.nan          # missing close
    df.iloc[5, df.columns.get_loc("Volume")] = 0                  # zero-volume day
    df.iloc[7, df.columns.get_loc("Adj Close")] = df["Adj Close"].iloc[6] * 2  # +100% jump
    df = df.rename(columns=P._FIELD_MAP)

    q = validate_ticker(df, "TST", jump_threshold=0.5)
    assert q.n_missing_close == 1
    assert q.n_zero_volume == 1
    assert q.n_extreme_moves >= 1          # the doubled price (and the recovery)
    assert not q.ok


def test_validate_clean_series_is_ok():
    df = _ticker_frame(n=20).rename(columns=P._FIELD_MAP)
    q = validate_ticker(df, "OK", jump_threshold=0.5)
    assert q.ok
    assert q.n_rows == 20
    assert q.pct_missing == 0.0


# --------------------------------------------------------------------------- #
# Extraction from yfinance layouts
# --------------------------------------------------------------------------- #

def test_extract_multiindex_and_flat():
    raw = _multi_raw({"AAA": _ticker_frame(), "BBB": _ticker_frame(base=50)})
    sub = P._extract_ticker(raw, "AAA")
    assert list(sub.columns) == P._OHLCV
    assert sub.index.name == "date"

    # Flat single-ticker layout.
    flat = _ticker_frame().rename(columns=P._FIELD_MAP)
    flat.columns = [c.title().replace("_", " ") if c != "adj_close" else "Adj Close"
                    for c in flat.columns]
    sub2 = P._extract_ticker(flat, "WHATEVER")
    assert "adj_close" in sub2.columns


def test_extract_missing_ticker_returns_none():
    raw = _multi_raw({"AAA": _ticker_frame()})
    assert P._extract_ticker(raw, "ZZZ") is None


# --------------------------------------------------------------------------- #
# Download orchestration
# --------------------------------------------------------------------------- #

def test_download_writes_good_and_reports_failures(tmp_path):
    good = {"AAA": _ticker_frame(), "BBB": _ticker_frame(base=200)}

    def fake_fetch(batch, start, end, auto_adjust):
        # Return only the tickers we "have"; CCC is delisted -> absent.
        return _multi_raw({t: good[t] for t in batch if t in good})

    report = download_prices(
        ["AAA", "BBB", "CCC"], "2021-01-01", "2021-02-01",
        price_dir=tmp_path, pause=0, fetcher=fake_fetch,
    )
    assert isinstance(report, QualityReport)
    assert report.requested == 3
    assert report.downloaded == 2
    assert report.failed == ["CCC"]
    assert report.missing_rate == pytest.approx(1 / 3)
    assert (tmp_path / "AAA.parquet").exists()
    assert not (tmp_path / "CCC.parquet").exists()


def test_download_retries_then_succeeds(tmp_path):
    calls = {"n": 0}

    def flaky_fetch(batch, start, end, auto_adjust):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("simulated transient failure")
        return _multi_raw({t: _ticker_frame() for t in batch})

    report = download_prices(
        ["AAA"], "2021-01-01", "2021-02-01",
        price_dir=tmp_path, pause=0, max_retries=3, fetcher=flaky_fetch,
    )
    assert calls["n"] == 2
    assert report.downloaded == 1
    assert report.failed == []


def test_download_permanent_failure_marks_batch_missing(tmp_path):
    def always_fail(batch, start, end, auto_adjust):
        raise ConnectionError("down")

    report = download_prices(
        ["AAA", "BBB"], "2021-01-01", "2021-02-01",
        price_dir=tmp_path, pause=0, max_retries=2, fetcher=always_fail,
    )
    assert report.downloaded == 0
    assert set(report.failed) == {"AAA", "BBB"}
    assert report.missing_rate == 1.0


def test_ticker_symbol_normalized_on_download(tmp_path):
    def fake_fetch(batch, start, end, auto_adjust):
        # yfinance is called with the normalized symbol.
        assert "BRK-B" in batch and "BRK.B" not in batch
        return _multi_raw({t: _ticker_frame() for t in batch})

    download_prices(["BRK.B"], "2021-01-01", "2021-02-01",
                    price_dir=tmp_path, pause=0, fetcher=fake_fetch)
    assert (tmp_path / "BRK-B.parquet").exists()


# --------------------------------------------------------------------------- #
# Read API
# --------------------------------------------------------------------------- #

@pytest.fixture
def populated_store(tmp_path):
    def fake_fetch(batch, start, end, auto_adjust):
        return _multi_raw({t: _ticker_frame(n=15, base=100 + 10 * i)
                           for i, t in enumerate(batch)})

    download_prices(["AAA", "BBB"], "2021-01-01", "2021-03-01",
                    price_dir=tmp_path, pause=0, fetcher=fake_fetch)
    return tmp_path


def test_get_prices_long(populated_store):
    df = get_prices(["AAA", "BBB"], price_dir=populated_store)
    assert set(df["ticker"]) == {"AAA", "BBB"}
    assert list(df.columns) == ["date", "ticker", *P._OHLCV]
    assert df.sort_values(["ticker", "date"]).equals(df)  # already sorted


def test_get_prices_wide_field(populated_store):
    wide = get_prices(["AAA", "BBB"], field="adj_close", price_dir=populated_store)
    assert list(wide.columns) == ["AAA", "BBB"]
    assert wide.index.name == "date"
    assert wide.notna().all().all()


def test_get_prices_date_filter(populated_store):
    df = get_prices("AAA", start="2021-01-11", end="2021-01-15",
                    price_dir=populated_store)
    assert df["date"].min() >= pd.Timestamp("2021-01-11")
    assert df["date"].max() <= pd.Timestamp("2021-01-15")


def test_get_prices_missing_ticker_skipped(populated_store):
    df = get_prices(["AAA", "NOPE"], price_dir=populated_store)
    assert set(df["ticker"]) == {"AAA"}


def test_get_prices_empty_when_nothing_found(tmp_path):
    df = get_prices(["NOPE"], price_dir=tmp_path)
    assert df.empty
    assert list(df.columns) == ["date", "ticker", *P._OHLCV]


def test_get_prices_bad_field_raises(populated_store):
    with pytest.raises(ValueError):
        get_prices(["AAA"], field="not_a_field", price_dir=populated_store)


# --------------------------------------------------------------------------- #
# Universe integration
# --------------------------------------------------------------------------- #

def test_universe_symbols_union(monkeypatch):
    from quantlab.data.universe import Constituent, MembershipChange, UniverseData

    data = UniverseData(
        constituents=[
            Constituent(ticker="AAA", security="A"),
            Constituent(ticker="BBB", security="B"),
        ],
        changes=[
            MembershipChange(date=date(2021, 5, 1), added_ticker="BBB",
                             removed_ticker="OLD"),      # in window
            MembershipChange(date=date(2019, 1, 1), added_ticker="ANCIENT"),  # out of window
        ],
        fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(P, "load_universe", lambda **k: data)

    syms = universe_symbols("2020-01-01", "2022-12-31")
    assert "AAA" in syms and "BBB" in syms
    assert "OLD" in syms              # removed within window -> included
    assert "ANCIENT" not in syms      # change outside window -> excluded

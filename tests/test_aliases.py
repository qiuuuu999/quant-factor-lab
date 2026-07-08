"""Tests for ticker alias resolution and alias-aware price download/read."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from quantlab.data import prices as P
from quantlab.data.aliases import (
    AliasResolver,
    DelistRecord,
    RenameRecord,
    infer_renames_from_changes,
    load_alias_resolver,
)

UTC = timezone.utc


def _resolver(renames, delisted=None):
    return AliasResolver(
        renames=[RenameRecord(old=o, new=n) for o, n in renames],
        delisted=[DelistRecord(ticker=t) for t in (delisted or [])],
    )


# --------------------------------------------------------------------------- #
# Resolver semantics
# --------------------------------------------------------------------------- #

def test_to_current_single_hop():
    r = _resolver([("FB", "META")])
    assert r.to_current("FB") == "META"
    assert r.to_current("META") == "META"      # already current
    assert r.to_current("AAPL") == "AAPL"      # unknown passes through


def test_to_current_follows_chain():
    r = _resolver([("VIAB", "VIAC"), ("VIAC", "PARA")])
    assert r.to_current("VIAB") == "PARA"
    assert r.to_current("VIAC") == "PARA"


def test_to_current_is_cycle_safe():
    # Pathological cycle must not hang.
    r = _resolver([("A", "B"), ("B", "A")])
    assert r.to_current("A") in {"A", "B"}


def test_aliases_of_reverse_lookup():
    r = _resolver([("VIAB", "VIAC"), ("VIAC", "PARA")])
    assert r.aliases_of("PARA") == ["VIAB", "VIAC"]


def test_case_and_dot_normalization():
    r = _resolver([("FB", "META")])
    assert r.to_current("fb") == "META"
    assert r.to_current("BRK.B") == "BRK-B"    # dot -> dash, no rename


def test_is_delisted():
    r = _resolver([("FB", "META")], delisted=["TWTR", "XLNX"])
    assert r.is_delisted("TWTR") and r.is_delisted("xlnx")
    assert not r.is_delisted("META")


# --------------------------------------------------------------------------- #
# Inference from the change log
# --------------------------------------------------------------------------- #

def test_infer_renames_from_changes():
    from quantlab.data.universe import MembershipChange, UniverseData

    data = UniverseData(
        constituents=[],
        changes=[
            # Same company, add-under-new + remove-under-old on one row => rename.
            MembershipChange(date=date(2023, 7, 10), added_ticker="EG",
                             added_security="Everest Group",
                             removed_ticker="RE", removed_security="Everest Re Group"),
            # Different companies swapping in/out => NOT a rename.
            MembershipChange(date=date(2021, 1, 1), added_ticker="NEW",
                             added_security="New Corp",
                             removed_ticker="OLD", removed_security="Old Industries"),
        ],
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    inferred = infer_renames_from_changes(data)
    pairs = {(r.old, r.new) for r in inferred}
    assert ("RE", "EG") in pairs
    assert ("OLD", "NEW") not in pairs


def test_load_alias_resolver_from_yaml(tmp_path):
    cfg = tmp_path / "renames.yaml"
    cfg.write_text(
        "renames:\n"
        "  - {old: FOO, new: BAR, date: 2020-01-01}\n"
        "delisted:\n"
        "  - {ticker: DEAD, date: 2019-01-01}\n"
    )
    r = load_alias_resolver(cfg, use_cache=False)
    assert r.to_current("FOO") == "BAR"
    assert r.is_delisted("DEAD")


def test_repo_config_has_key_renames():
    # The shipped config must at least cover the flagship cases.
    r = load_alias_resolver(use_cache=False)
    assert r.to_current("FB") == "META"
    assert r.to_current("ANTM") == "ELV"
    assert r.to_current("CTL") == "LUMN"


# --------------------------------------------------------------------------- #
# Alias-aware download + read
# --------------------------------------------------------------------------- #

def _fake_frame(n=12, start="2021-01-04", base=100.0):
    import numpy as np
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(np.linspace(base, base + n - 1, n), index=idx)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close,
         "Adj Close": close, "Volume": pd.Series(np.full(n, 1e6), index=idx)},
        index=idx,
    )


def test_download_fetches_current_symbol_and_stores_canonical(tmp_path):
    r = _resolver([("FB", "META")])
    fetched = {}

    def fake_fetch(batch, start, end, auto_adjust):
        fetched["batch"] = list(batch)
        # We only "have" data under the current symbol META.
        return pd.concat({t: _fake_frame() for t in batch if t == "META"},
                         axis=1, keys=[t for t in batch if t == "META"])

    report = P.download_prices(["FB"], "2021-01-01", "2021-02-01",
                               price_dir=tmp_path, pause=0, fetcher=fake_fetch,
                               resolver=r)
    # yfinance was queried with META (current), not FB.
    assert fetched["batch"] == ["META"]
    # Stored under the canonical symbol.
    assert (tmp_path / "META.parquet").exists()
    assert not (tmp_path / "FB.parquet").exists()
    # FB counts as covered because META returned data.
    assert report.downloaded == 1
    assert report.failed == []


def test_get_prices_resolves_both_aliases(tmp_path):
    r = _resolver([("FB", "META")])

    def fake_fetch(batch, start, end, auto_adjust):
        return pd.concat({t: _fake_frame() for t in batch}, axis=1, keys=list(batch))

    P.download_prices(["META"], "2021-01-01", "2021-02-01",
                      price_dir=tmp_path, pause=0, fetcher=fake_fetch, resolver=r)

    via_new = P.get_prices("META", price_dir=tmp_path, resolver=r)
    via_old = P.get_prices("FB", price_dir=tmp_path, resolver=r)

    # Same underlying series, relabelled to the requested symbol.
    assert set(via_new["ticker"]) == {"META"}
    assert set(via_old["ticker"]) == {"FB"}
    pd.testing.assert_series_equal(
        via_new["adj_close"].reset_index(drop=True),
        via_old["adj_close"].reset_index(drop=True),
    )


def test_expected_missing_labels_delisted(tmp_path):
    r = _resolver([], delisted=["TWTR"])

    def fake_fetch(batch, start, end, auto_adjust):
        # Neither symbol has data.
        return pd.DataFrame()

    report = P.download_prices(["TWTR", "GOODCO"], "2021-01-01", "2021-02-01",
                               price_dir=tmp_path, pause=0, fetcher=fake_fetch,
                               resolver=r)
    assert set(report.failed) == {"TWTR", "GOODCO"}
    assert report.expected_missing == ["TWTR"]           # known delisting
    assert report.unexpected_missing == ["GOODCO"]       # needs investigation

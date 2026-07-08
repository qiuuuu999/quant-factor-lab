"""Tests for the point-in-time S&P 500 universe module.

The default suite is fully hermetic (no network): point-in-time reconstruction
is verified against a hand-built synthetic snapshot, and parsing is verified
against an inline HTML fixture (including the ``rowspan`` date case). A live
network test that scrapes the real Wikipedia page is gated behind the
``QUANTLAB_LIVE_TESTS=1`` environment variable so CI/offline runs stay green.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

import pytest

from quantlab.data.universe import (
    Constituent,
    MembershipChange,
    UniverseData,
    load_universe,
    parse_html,
)

UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def synthetic() -> UniverseData:
    """A controlled snapshot exercising every reconstruction edge case.

    Scraped (``fetched_at``) on 2026-01-01. Current members:
    AAA, NEWCO, META, ZZZ, KEEP.
    """
    constituents = [
        Constituent(ticker="AAA", security="Alpha Inc.", date_added=date(2000, 1, 1)),
        Constituent(ticker="NEWCO", security="New Co.", date_added=date(2020, 12, 21)),
        Constituent(ticker="META", security="Meta Platforms", date_added=date(2013, 12, 23)),
        Constituent(ticker="ZZZ", security="Zeta Corp.", date_added=date(2015, 6, 30)),
        Constituent(ticker="KEEP", security="Keep Corp.", date_added=date(2001, 1, 1)),
    ]
    changes = [
        # NEWCO joined and OLDCO left on the same day (2020).
        MembershipChange(date=date(2020, 12, 21), added_ticker="NEWCO",
                         added_security="New Co.", removed_ticker="OLDCO",
                         removed_security="Old Co.", reason="rebalance"),
        # GONE was dropped in 2016 (so it WAS a member in 2015).
        MembershipChange(date=date(2016, 1, 1), removed_ticker="GONE",
                         removed_security="Gone Inc.", reason="acquired"),
        # ZZZ joined exactly on the boundary date used by the tests.
        MembershipChange(date=date(2015, 6, 30), added_ticker="ZZZ",
                         added_security="Zeta Corp.", reason="addition"),
        # META joined in 2013 under its OLD symbol "FB" (as the change log
        # records it). Reconstruction must resolve "FB" -> "META" to undo this.
        MembershipChange(date=date(2013, 12, 23), added_ticker="FB",
                         added_security="Facebook, Inc.", reason="addition"),
        # A future-dated (announced but not-yet-effective) change: must be
        # ignored because the current snapshot does not reflect it yet.
        MembershipChange(date=date(2026, 6, 30), added_ticker="FUTCO",
                         added_security="Future Co.", reason="pending"),
    ]
    return UniverseData(
        constituents=constituents,
        changes=changes,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


# --------------------------------------------------------------------------- #
# Point-in-time reconstruction — the core survivorship-bias tests
# --------------------------------------------------------------------------- #

def test_historical_universe_excludes_later_additions(synthetic):
    members = synthetic.members_as_of("2015-06-30")
    # NEWCO joined only in 2020 -> must NOT appear in a 2015 universe.
    assert "NEWCO" not in members


def test_historical_universe_includes_later_removals(synthetic):
    members = synthetic.members_as_of("2015-06-30")
    # OLDCO (removed 2020) and GONE (removed 2016) were still members in 2015.
    assert "OLDCO" in members
    assert "GONE" in members


def test_boundary_change_is_reflected_on_effective_date(synthetic):
    # A change effective ON the query date counts as already in effect.
    assert "ZZZ" in synthetic.members_as_of("2015-06-30")
    # ...but not the day before it took effect.
    assert "ZZZ" not in synthetic.members_as_of("2015-06-29")


def test_future_dated_change_is_ignored(synthetic):
    # FUTCO's addition is dated after the scrape; the current snapshot does not
    # include it, so it must not leak into any reconstruction.
    for d in ("2015-06-30", "2020-01-01", "2026-01-01"):
        assert "FUTCO" not in synthetic.members_as_of(d)


def test_reference_date_matches_current_constituents(synthetic):
    # As of the scrape date, reconstruction must equal the raw current members.
    at_ref = set(synthetic.members_as_of("2026-01-01"))
    assert at_ref == {c.ticker for c in synthetic.constituents}


def test_ticker_rename_fb_to_meta(synthetic):
    # Before 2022-06-09 the company traded as FB, not META.
    early = synthetic.members_as_of("2015-06-30")
    assert "FB" in early and "META" not in early
    # After the rename it is META.
    late = synthetic.members_as_of("2023-01-01")
    assert "META" in late and "FB" not in late


def test_rename_addition_undone_before_join_date(synthetic):
    # META joined (as "FB") on 2013-12-23. Before that it was in the index under
    # neither symbol — the log records the add as "FB" but membership holds
    # "META", so reconstruction must resolve the alias to remove it.
    before = synthetic.members_as_of("2013-01-01")
    assert "FB" not in before and "META" not in before
    # Just after joining, present under the period-correct symbol "FB".
    after = synthetic.members_as_of("2014-06-30")
    assert "FB" in after and "META" not in after


def test_normalize_ticker_option():
    data = UniverseData(
        constituents=[Constituent(ticker="BRK.B", security="Berkshire")],
        changes=[],
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert data.members_as_of("2025-01-01", normalize=True) == ["BRK-B"]
    assert data.members_as_of("2025-01-01", normalize=False) == ["BRK.B"]


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #

def test_parquet_cache_roundtrip(synthetic, tmp_path):
    synthetic.to_parquet(tmp_path)
    assert UniverseData.cache_exists(tmp_path)

    restored = UniverseData.from_parquet(tmp_path)
    assert len(restored.constituents) == len(synthetic.constituents)
    assert len(restored.changes) == len(synthetic.changes)
    assert restored.fetched_at == synthetic.fetched_at
    # Reconstruction must be identical after a round-trip.
    assert restored.members_as_of("2015-06-30") == synthetic.members_as_of("2015-06-30")


def test_staleness_window(synthetic):
    fresh = UniverseData(
        constituents=[], changes=[], fetched_at=datetime.now(UTC),
    )
    assert not fresh.is_stale(timedelta(days=7))

    old = UniverseData(
        constituents=[], changes=[],
        fetched_at=datetime.now(UTC) - timedelta(days=10),
    )
    assert old.is_stale(timedelta(days=7))


def test_load_universe_uses_fresh_cache_without_network(synthetic, tmp_path):
    # Re-stamp the snapshot as freshly fetched so it is inside the staleness
    # window; otherwise load_universe would (correctly) try to refresh.
    fresh = synthetic.model_copy(update={"fetched_at": datetime.now(UTC)})
    fresh.to_parquet(tmp_path)

    def _boom(*a, **k):  # must never be called when cache is fresh
        raise AssertionError("network fetch attempted despite fresh cache")

    import quantlab.data.universe as uni
    orig = uni.fetch_html
    uni.fetch_html = _boom
    try:
        data = load_universe(cache_dir=tmp_path)
    finally:
        uni.fetch_html = orig
    # Reconstruction relative to the (now-current) reference date still works.
    assert "NEWCO" not in data.members_as_of("2015-06-30")
    assert "OLDCO" in data.members_as_of("2015-06-30")


# --------------------------------------------------------------------------- #
# HTML parsing (inline fixture, incl. rowspan on the date column)
# --------------------------------------------------------------------------- #

_FIXTURE_HTML = """
<html><body>
<table id="constituents">
  <tbody>
    <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Sub</th>
        <th>HQ</th><th>Date added</th><th>CIK</th><th>Founded</th></tr>
    <tr><td>AAA</td><td>Alpha Inc.</td><td>Technology</td><td>Software</td>
        <td>New York</td><td>2005-01-01</td><td>0000000001</td><td>1990</td></tr>
    <tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>Insurance</td>
        <td>Omaha</td><td>2010-02-16</td><td>0000000002</td><td>1839</td></tr>
  </tbody>
</table>
<table id="changes">
  <tbody>
    <tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th>
        <th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>
    <tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
    <tr><td rowspan="2">June 20, 2024</td><td>AAA</td><td>Alpha Inc.</td>
        <td>OLD1</td><td>Old One</td><td>reason1[1]</td></tr>
    <tr><td>CCC</td><td>Gamma</td><td>OLD2</td><td>Old Two</td><td>reason2</td></tr>
    <tr><td>March 3, 2015</td><td></td><td></td><td>DROP</td><td>Dropco</td>
        <td>reason3</td></tr>
  </tbody>
</table>
</body></html>
"""


def test_parse_constituents():
    data = parse_html(_FIXTURE_HTML, fetched_at=datetime(2026, 1, 1, tzinfo=UTC))
    by_ticker = {c.ticker: c for c in data.constituents}
    assert set(by_ticker) == {"AAA", "BRK.B"}
    assert by_ticker["AAA"].date_added == date(2005, 1, 1)
    assert by_ticker["AAA"].sector == "Technology"


def test_parse_changes_with_rowspan_date():
    data = parse_html(_FIXTURE_HTML, fetched_at=datetime(2026, 1, 1, tzinfo=UTC))
    changes = sorted(data.changes, key=lambda c: (c.date, c.added_ticker or ""))
    assert len(changes) == 3

    # The two 2024 rows share one rowspanned date cell.
    y2024 = [c for c in changes if c.date == date(2024, 6, 20)]
    assert {c.added_ticker for c in y2024} == {"AAA", "CCC"}
    assert {c.removed_ticker for c in y2024} == {"OLD1", "OLD2"}

    drop = [c for c in changes if c.date == date(2015, 3, 3)][0]
    assert drop.removed_ticker == "DROP"
    assert drop.added_ticker is None
    assert drop.reason == "reason3"


def test_parse_strips_footnote_markers():
    data = parse_html(_FIXTURE_HTML, fetched_at=datetime(2026, 1, 1, tzinfo=UTC))
    reasons = {c.reason for c in data.changes}
    assert "reason1" in reasons          # the "[1]" citation marker was stripped
    assert "reason1[1]" not in reasons


# --------------------------------------------------------------------------- #
# Live network test (opt-in via QUANTLAB_LIVE_TESTS=1)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    os.environ.get("QUANTLAB_LIVE_TESTS") != "1",
    reason="set QUANTLAB_LIVE_TESTS=1 to run the live Wikipedia scrape test",
)
def test_live_2015_universe(tmp_path):
    data = load_universe(force_refresh=True, cache_dir=tmp_path)

    members = set(data.members_as_of("2015-06-30"))
    # Sanity: the S&P 500 has ~500 members.
    assert 450 <= len(members) <= 520
    # Tesla joined the index on 2020-12-21 — it must be absent in 2015.
    assert "TSLA" not in members
    # Apple has been a member continuously since well before 2015.
    assert "AAPL" in members

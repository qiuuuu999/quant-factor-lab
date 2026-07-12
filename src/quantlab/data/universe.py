"""Point-in-time S&P 500 universe — survivorship-bias-free constituent lookup.

The single most common source of survivorship bias in equity research is using
*today's* index membership to define the tradable universe for *past* dates.
Stocks that were dropped from the index (often because they performed badly or
were acquired) silently disappear from history, inflating backtested returns.

This module reconstructs the universe *as it actually was* on any historical
date by combining two tables scraped from the Wikipedia article
"List of S&P 500 companies":

* the **current constituents** table (membership as of the scrape), and
* the **selected changes** table (every addition / removal with its effective
  date).

Starting from the current membership we walk the change log *backwards* in time,
undoing each addition/removal that happened after the query date, to recover the
membership set that was live on that date.  Known ticker renames (e.g.
``FB`` -> ``META``) are translated back to the symbol that was in use at the
query date.

Data is cached to Parquet under the repo ``data/`` directory (git-ignored) with
a configurable staleness window.

Typical use::

    from quantlab.data.universe import get_universe
    tickers = get_universe("2015-06-30")   # what was *really* in the index then
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, field_validator

__all__ = [
    "Constituent",
    "MembershipChange",
    "UniverseData",
    "get_universe",
    "load_universe",
    "refresh_universe",
    "TICKER_RENAMES",
    "WIKI_URL",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

#: Wikipedia rejects the default python-requests User-Agent; mimic a browser.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

#: Default cache freshness window. Index membership changes at most a few times
#: per quarter, so a week is plenty while still catching rebalances.
DEFAULT_MAX_AGE = timedelta(days=7)

#: Known ticker-symbol changes for companies that stayed in the index under a
#: new symbol. Maps the *current* symbol to a list of ``(old_symbol,
#: effective_date)`` pairs: before ``effective_date`` the company traded under
#: ``old_symbol``. The change log on Wikipedia does not record pure symbol
#: renames, so we translate them here to return the historically-correct ticker.
TICKER_RENAMES: dict[str, list[tuple[str, date]]] = {
    "META": [("FB", date(2022, 6, 9))],       # Facebook -> Meta Platforms
    "PARA": [("VIAC", date(2022, 2, 16))],    # ViacomCBS -> Paramount Global
    "WBD": [("DISCA", date(2022, 4, 11))],    # Discovery -> Warner Bros. Discovery
}

#: Reverse of ``TICKER_RENAMES``: old symbol -> current symbol. Used to resolve a
#: change-log entry (recorded under the symbol of the day, e.g. ``FB``) onto the
#: symbol carried in the current membership set (e.g. ``META``) so that undoing
#: an addition/removal matches the right security.
_OLD_TO_CURRENT: dict[str, str] = {
    old: current for current, history in TICKER_RENAMES.items() for old, _ in history
}
# Note: the rename map is intentionally conservative and easily extended. Only
# symbols that survive to the current constituents list benefit from it; renames
# of companies later dropped are handled best-effort via the change log.

_ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_REF_RE = re.compile(r"\[[^\]]*\]")  # strip footnote/citation markers like [1], [a]

# Cache file names (module-level so pydantic does not treat them as fields).
_CONSTITUENTS_FILE = "constituents.parquet"
_CHANGES_FILE = "changes.parquet"
_META_FILE = "meta.json"


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def _repo_root() -> Path:
    """Walk up from this file to the directory containing ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: three levels up (src/quantlab/data/ -> repo root).
    return here.parents[3]


def default_cache_dir() -> Path:
    """Default Parquet cache location: ``<repo>/data/universe`` (git-ignored)."""
    return _repo_root() / "data" / "universe"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _clean(text: str) -> str:
    """Collapse whitespace and strip footnote/citation markers."""
    return _REF_RE.sub("", text).replace("\xa0", " ").strip()


def _to_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def normalize_ticker(ticker: str) -> str:
    """Normalize a Wikipedia symbol to the yfinance/CRSP style (``.`` -> ``-``)."""
    return ticker.replace(".", "-")


# --------------------------------------------------------------------------- #
# Pydantic data models
# --------------------------------------------------------------------------- #

class Constituent(BaseModel):
    """A single current index member."""

    ticker: str
    security: str
    sector: str | None = None
    date_added: date | None = None

    @field_validator("ticker")
    @classmethod
    def _strip_ticker(cls, v: str) -> str:
        return v.strip()


class MembershipChange(BaseModel):
    """One row of the index change log (an addition and/or a removal)."""

    date: date
    added_ticker: str | None = None
    added_security: str | None = None
    removed_ticker: str | None = None
    removed_security: str | None = None
    reason: str | None = None


class UniverseData(BaseModel):
    """The full scraped universe: current members + change log + fetch time."""

    constituents: list[Constituent] = Field(default_factory=list)
    changes: list[MembershipChange] = Field(default_factory=list)
    fetched_at: datetime
    source_url: str = WIKI_URL

    # -- staleness ---------------------------------------------------------- #

    def is_stale(self, max_age: timedelta = DEFAULT_MAX_AGE) -> bool:
        fetched = self.fetched_at
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - fetched > max_age

    # -- point-in-time reconstruction -------------------------------------- #

    def members_as_of(
        self,
        as_of: str | date | datetime,
        *,
        normalize: bool = False,
        apply_renames: bool = True,
    ) -> list[str]:
        """Return the tickers that were *actually* in the index on ``as_of``.

        Reconstructed by undoing every change that took effect after ``as_of``
        (but on/before the scrape date) against the current membership.
        """
        as_of = _to_date(as_of)
        ref_date = self.fetched_at.date()

        members: set[str] = {c.ticker for c in self.constituents}

        # Process the change log newest-first.
        for ch in sorted(self.changes, key=lambda c: c.date, reverse=True):
            if ch.date > ref_date:
                # Announced but not yet reflected in the current snapshot; the
                # current membership does not include it, so nothing to undo.
                continue
            if ch.date <= as_of:
                # This change (and all older ones) is already reflected in the
                # membership as it stood on `as_of`.
                break
            # Change took effect in (as_of, ref_date]: undo it. Resolve the
            # change-log symbol onto its current symbol first, so a rename
            # (e.g. the log says "FB" while membership holds "META") still
            # matches the right security.
            if ch.added_ticker:
                # wasn't a member yet at as_of
                members.discard(_OLD_TO_CURRENT.get(ch.added_ticker, ch.added_ticker))
            if ch.removed_ticker:
                # was still a member at as_of
                members.add(_OLD_TO_CURRENT.get(ch.removed_ticker, ch.removed_ticker))

        if apply_renames:
            members = self._apply_renames(members, as_of)
        if normalize:
            members = {normalize_ticker(t) for t in members}
        return sorted(members)

    @staticmethod
    def _apply_renames(members: set[str], as_of: date) -> set[str]:
        out = set(members)
        for current_symbol, history in TICKER_RENAMES.items():
            if current_symbol not in out:
                continue
            for old_symbol, effective in history:
                if as_of < effective:
                    out.discard(current_symbol)
                    out.add(old_symbol)
                    break
        return out

    # -- Parquet cache ------------------------------------------------------ #

    @classmethod
    def cache_exists(cls, cache_dir: Path) -> bool:
        cache_dir = Path(cache_dir)
        return all(
            (cache_dir / f).exists()
            for f in (_CONSTITUENTS_FILE, _CHANGES_FILE, _META_FILE)
        )

    def to_parquet(self, cache_dir: Path | None = None) -> Path:
        cache_dir = Path(cache_dir or default_cache_dir())
        cache_dir.mkdir(parents=True, exist_ok=True)

        cons_df = pd.DataFrame(
            [
                {
                    "ticker": c.ticker,
                    "security": c.security,
                    "sector": c.sector,
                    "date_added": c.date_added.isoformat() if c.date_added else None,
                }
                for c in self.constituents
            ]
        )
        chg_df = pd.DataFrame(
            [
                {
                    "date": c.date.isoformat(),
                    "added_ticker": c.added_ticker,
                    "added_security": c.added_security,
                    "removed_ticker": c.removed_ticker,
                    "removed_security": c.removed_security,
                    "reason": c.reason,
                }
                for c in self.changes
            ]
        )
        cons_df.to_parquet(cache_dir / _CONSTITUENTS_FILE, index=False)
        chg_df.to_parquet(cache_dir / _CHANGES_FILE, index=False)
        (cache_dir / _META_FILE).write_text(
            json.dumps(
                {"fetched_at": self.fetched_at.isoformat(), "source_url": self.source_url}
            )
        )
        return cache_dir

    @classmethod
    def from_parquet(cls, cache_dir: Path) -> "UniverseData":
        cache_dir = Path(cache_dir)
        cons_df = pd.read_parquet(cache_dir / _CONSTITUENTS_FILE)
        chg_df = pd.read_parquet(cache_dir / _CHANGES_FILE)
        meta = json.loads((cache_dir / _META_FILE).read_text())

        def _d(v):
            return date.fromisoformat(v) if isinstance(v, str) and v else None

        constituents = [
            Constituent(
                ticker=r.ticker,
                security=r.security,
                sector=None if pd.isna(r.sector) else r.sector,
                date_added=_d(None if pd.isna(r.date_added) else r.date_added),
            )
            for r in cons_df.itertuples(index=False)
        ]
        changes = [
            MembershipChange(
                date=date.fromisoformat(r.date),
                added_ticker=None if pd.isna(r.added_ticker) else r.added_ticker,
                added_security=None if pd.isna(r.added_security) else r.added_security,
                removed_ticker=None if pd.isna(r.removed_ticker) else r.removed_ticker,
                removed_security=None if pd.isna(r.removed_security) else r.removed_security,
                reason=None if pd.isna(r.reason) else r.reason,
            )
            for r in chg_df.itertuples(index=False)
        ]
        return cls(
            constituents=constituents,
            changes=changes,
            fetched_at=datetime.fromisoformat(meta["fetched_at"]),
            source_url=meta.get("source_url", WIKI_URL),
        )


# --------------------------------------------------------------------------- #
# HTML fetching & parsing
# --------------------------------------------------------------------------- #

def fetch_html(session: requests.Session | None = None, timeout: int = 30) -> str:
    """Download the Wikipedia article HTML."""
    sess = session or requests.Session()
    resp = sess.get(WIKI_URL, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _expand_table(table) -> list[list[str]]:
    """Expand a table's *data* rows into a rectangular grid.

    Handles ``rowspan``/``colspan`` (the change log uses ``rowspan`` on the date
    column when several changes share an effective date) and drops header rows
    (those without any ``<td>``).
    """
    data_rows = [tr for tr in table.find_all("tr") if tr.find("td")]
    grid: list[list[str]] = []
    carry: dict[int, list] = {}  # col -> [text, remaining_rows]

    for tr in data_rows:
        cells = tr.find_all(["td", "th"])
        row: list[str] = []
        col = 0
        ci = 0
        while ci < len(cells) or carry:
            if col in carry:
                text, rem = carry[col]
                row.append(text)
                rem -= 1
                if rem:
                    carry[col] = [text, rem]
                else:
                    del carry[col]
                col += 1
                continue
            if ci >= len(cells):
                ahead = [c for c in carry if c > col]
                if not ahead:
                    break
                col = min(ahead)
                continue
            cell = cells[ci]
            ci += 1
            text = _clean(cell.get_text(" ", strip=True))
            colspan = int(cell.get("colspan", 1) or 1)
            rowspan = int(cell.get("rowspan", 1) or 1)
            for _ in range(colspan):
                row.append(text)
                if rowspan > 1:
                    carry[col] = [text, rowspan - 1]
                col += 1
        grid.append(row)
    return grid


def _parse_iso(text: str) -> date | None:
    m = _ISO_DATE_RE.search(text or "")
    return date.fromisoformat(m.group(1)) if m else None


def _parse_change_date(text: str) -> date | None:
    text = (text or "").strip()
    if not text:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return _parse_iso(text)


def parse_html(html: str, fetched_at: datetime | None = None) -> UniverseData:
    """Parse the article HTML into a :class:`UniverseData`.

    Pure function (no network) so it can be unit-tested against saved fixtures.
    """
    soup = BeautifulSoup(html, "html.parser")

    cons_table = soup.find("table", id="constituents")
    chg_table = soup.find("table", id="changes")
    if cons_table is None or chg_table is None:
        raise ValueError(
            "Could not locate the 'constituents'/'changes' tables — "
            "Wikipedia layout may have changed."
        )

    constituents: list[Constituent] = []
    for r in _expand_table(cons_table):
        if len(r) < 2 or not r[0]:
            continue
        constituents.append(
            Constituent(
                ticker=r[0],
                security=r[1],
                sector=r[2] if len(r) > 2 and r[2] else None,
                date_added=_parse_iso(r[5]) if len(r) > 5 else None,
            )
        )

    changes: list[MembershipChange] = []
    for r in _expand_table(chg_table):
        if len(r) < 6:
            r = r + [""] * (6 - len(r))
        d = _parse_change_date(r[0])
        if d is None:
            continue
        add_t = r[1] or None
        rem_t = r[3] or None
        if not add_t and not rem_t:
            continue
        changes.append(
            MembershipChange(
                date=d,
                added_ticker=add_t,
                added_security=r[2] or None,
                removed_ticker=rem_t,
                removed_security=r[4] or None,
                reason=r[5] or None,
            )
        )

    if fetched_at is None:
        fetched_at = datetime.now(timezone.utc)
    return UniverseData(
        constituents=constituents, changes=changes, fetched_at=fetched_at
    )


# --------------------------------------------------------------------------- #
# Cache-aware loading & public API
# --------------------------------------------------------------------------- #

def load_universe(
    *,
    force_refresh: bool = False,
    max_age: timedelta = DEFAULT_MAX_AGE,
    cache_dir: Path | None = None,
    session: requests.Session | None = None,
) -> UniverseData:
    """Load universe data, refreshing from Wikipedia only when the cache is stale.

    On a network/parse failure a usable (if stale) cache is returned with a
    warning; only a cold cache re-raises.
    """
    cache_dir = Path(cache_dir or default_cache_dir())
    cached: UniverseData | None = None
    if UniverseData.cache_exists(cache_dir):
        try:
            cached = UniverseData.from_parquet(cache_dir)
        except Exception as exc:  # corrupt cache — treat as cold
            warnings.warn(f"Ignoring unreadable universe cache: {exc}", stacklevel=2)

    if cached is not None and not force_refresh and not cached.is_stale(max_age):
        return cached

    try:
        data = parse_html(fetch_html(session=session))
        data.to_parquet(cache_dir)
        return data
    except Exception as exc:
        if cached is not None:
            warnings.warn(
                f"Universe refresh failed ({exc}); using stale cache "
                f"from {cached.fetched_at.isoformat()}.",
                stacklevel=2,
            )
            return cached
        raise


def refresh_universe(cache_dir: Path | None = None) -> UniverseData:
    """Force a fresh scrape and rewrite the Parquet cache."""
    return load_universe(force_refresh=True, cache_dir=cache_dir)


def get_universe(
    as_of_date: str | date | datetime,
    *,
    normalize: bool = False,
    apply_renames: bool = True,
    force_refresh: bool = False,
    max_age: timedelta = DEFAULT_MAX_AGE,
    cache_dir: Path | None = None,
) -> list[str]:
    """Point-in-time S&P 500 membership for ``as_of_date``.

    Parameters
    ----------
    as_of_date:
        The historical date (``"YYYY-MM-DD"`` or a ``date``/``datetime``).
    normalize:
        If ``True``, return yfinance-style tickers (``BRK.B`` -> ``BRK-B``).
    apply_renames:
        If ``True`` (default), translate current symbols back to the symbol that
        was in use on ``as_of_date`` (e.g. ``META`` -> ``FB`` before 2022-06-09).
    """
    data = load_universe(
        force_refresh=force_refresh, max_age=max_age, cache_dir=cache_dir
    )
    return data.members_as_of(
        as_of_date, normalize=normalize, apply_renames=apply_renames
    )


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

def _demo(dates: Iterable[str] = ("2010-06-30", "2015-06-30", "2020-06-30", "2025-06-30")) -> None:
    data = load_universe()
    print(f"Universe scraped {data.fetched_at.isoformat()} — "
          f"{len(data.constituents)} current members, "
          f"{len(data.changes)} change-log rows\n")
    for d in dates:
        members = data.members_as_of(d)
        sample = ", ".join(members[:8])
        print(f"{d}: {len(members):>3} constituents   e.g. {sample} ...")


if __name__ == "__main__":  # pragma: no cover
    _demo()

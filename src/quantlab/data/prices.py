"""Price data pipeline — batched yfinance downloads, validation, Parquet store.

This module turns a (potentially large, survivorship-bias-free) set of tickers
from :mod:`quantlab.data.universe` into a local, validated, per-ticker Parquet
store of daily OHLCV + adjusted-close bars.

Design goals
------------
* **Resilient batch download.** Tickers are downloaded in batches with retry and
  a polite inter-batch pause. A ticker that yfinance cannot serve (very common
  for *delisted* names — exactly the stocks that matter for killing survivorship
  bias) is logged and counted, never fatal.
* **Data-quality gate.** Every downloaded series is checked for missing values,
  zero-volume sessions, and implausible single-day jumps (>50% by default),
  producing a :class:`QualityReport`.
* **Uniform read API.** :func:`get_prices` reads any subset of tickers/date range
  back from the store, either as a tidy long frame or a wide field panel.

Note on residual survivorship bias: yfinance rarely has full history for
delisted tickers, so even with a point-in-time universe some dropped names will
be missing prices. The download report surfaces this missing rate explicitly.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd
from pydantic import BaseModel, Field

from quantlab.data.aliases import AliasResolver, load_alias_resolver
from quantlab.data.universe import _repo_root, load_universe, normalize_ticker

__all__ = [
    "TickerQuality",
    "QualityReport",
    "download_prices",
    "get_prices",
    "available_tickers",
    "universe_symbols",
    "default_price_dir",
]

log = logging.getLogger("quantlab.data.prices")

#: yfinance field name -> our snake_case column name.
_FIELD_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
_OHLCV = ["open", "high", "low", "close", "adj_close", "volume"]

#: A single-day absolute return above this flags a bar for manual review.
DEFAULT_JUMP_THRESHOLD = 0.5


def default_price_dir() -> Path:
    """Per-ticker Parquet store location: ``<repo>/data/prices`` (git-ignored)."""
    return _repo_root() / "data" / "prices"


# --------------------------------------------------------------------------- #
# Quality models
# --------------------------------------------------------------------------- #

class TickerQuality(BaseModel):
    """Data-quality summary for one ticker's downloaded series."""

    ticker: str
    n_rows: int
    start: date | None = None
    end: date | None = None
    n_missing_close: int = 0
    pct_missing: float = 0.0
    n_zero_volume: int = 0
    n_extreme_moves: int = 0
    extreme_dates: list[date] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the series has no critical quality issues."""
        return self.n_missing_close == 0 and self.n_extreme_moves == 0 and self.n_rows > 0


class QualityReport(BaseModel):
    """Aggregate report for a download run."""

    requested: int
    downloaded: int
    failed: list[str] = Field(default_factory=list)
    per_ticker: list[TickerQuality] = Field(default_factory=list)
    #: Subset of `failed` known to be genuinely delisted (no current symbol) —
    #: expected to be missing from a free data source, not a fixable gap.
    expected_missing: list[str] = Field(default_factory=list)

    @property
    def missing_rate(self) -> float:
        """Fraction of requested tickers that returned no data."""
        return len(self.failed) / self.requested if self.requested else 0.0

    @property
    def unexpected_missing(self) -> list[str]:
        """Failed tickers that are *not* known delistings — worth investigating."""
        known = set(self.expected_missing)
        return [t for t in self.failed if t not in known]

    def flagged(self) -> list[TickerQuality]:
        """Downloaded tickers with quality issues worth reviewing."""
        return [q for q in self.per_ticker if not q.ok]

    def to_frame(self) -> pd.DataFrame:
        """Per-ticker quality table as a DataFrame."""
        return pd.DataFrame(
            [
                {
                    "ticker": q.ticker,
                    "n_rows": q.n_rows,
                    "start": q.start,
                    "end": q.end,
                    "n_missing_close": q.n_missing_close,
                    "pct_missing": round(q.pct_missing, 4),
                    "n_zero_volume": q.n_zero_volume,
                    "n_extreme_moves": q.n_extreme_moves,
                    "ok": q.ok,
                }
                for q in self.per_ticker
            ]
        )

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        flagged = self.flagged()
        lines = [
            "Price data quality report",
            "=" * 48,
            f"Requested tickers : {self.requested}",
            f"Downloaded        : {self.downloaded}",
            f"Failed (no data)  : {len(self.failed)}  "
            f"({self.missing_rate:.1%} missing rate)",
            f"  delisted (expected)  : {len(self.expected_missing)}",
            f"  unexpected missing   : {len(self.unexpected_missing)}",
            f"Flagged for review: {len(flagged)}",
        ]
        if self.unexpected_missing:
            shown = ", ".join(sorted(self.unexpected_missing)[:20])
            more = ("" if len(self.unexpected_missing) <= 20
                    else f" ... (+{len(self.unexpected_missing) - 20})")
            lines.append(f"  unexpected: {shown}{more}")
        if flagged:
            lines.append("  quality issues:")
            for q in flagged[:20]:
                bits = []
                if q.n_missing_close:
                    bits.append(f"{q.n_missing_close} missing")
                if q.n_extreme_moves:
                    bits.append(f"{q.n_extreme_moves} extreme moves")
                if q.n_zero_volume:
                    bits.append(f"{q.n_zero_volume} zero-vol")
                lines.append(f"    {q.ticker}: {', '.join(bits)}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# yfinance fetch + extraction (network isolated here for testability)
# --------------------------------------------------------------------------- #

def _fetch_yf(
    tickers: Sequence[str],
    start: str | date,
    end: str | date,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Thin wrapper around ``yfinance.download`` (monkeypatched in tests)."""
    import yfinance as yf

    return yf.download(
        list(tickers),
        start=str(start),
        end=str(end),
        auto_adjust=auto_adjust,
        group_by="ticker",
        actions=False,
        progress=False,
        threads=True,
    )


def _extract_ticker(raw: pd.DataFrame | None, ticker: str) -> pd.DataFrame | None:
    """Pull one ticker's normalized OHLCV frame out of a yfinance result.

    Handles both the multi-ticker ``(ticker, field)`` MultiIndex layout and the
    flat single-ticker layout. Returns ``None`` when the ticker has no usable
    rows (empty / all-NaN — the delisted-stock case).
    """
    if raw is None or len(raw) == 0:
        return None

    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        level0 = set(cols.get_level_values(0))
        level1 = set(cols.get_level_values(1))
        if ticker in level0:
            sub = raw[ticker].copy()
        elif ticker in level1:
            sub = raw.xs(ticker, axis=1, level=1).copy()
        else:
            return None
    else:
        sub = raw.copy()

    sub = sub.rename(columns=_FIELD_MAP)
    # Keep only known OHLCV columns that are present.
    present = [c for c in _OHLCV if c in sub.columns]
    if not present:
        return None
    sub = sub[present]
    sub = sub.dropna(how="all")
    if sub.empty:
        return None

    sub.index = pd.to_datetime(sub.index)
    sub.index.name = "date"
    return sub


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_ticker(
    df: pd.DataFrame, ticker: str, jump_threshold: float = DEFAULT_JUMP_THRESHOLD
) -> TickerQuality:
    """Compute data-quality metrics for one ticker's price frame."""
    n_rows = len(df)
    if n_rows == 0:
        return TickerQuality(ticker=ticker, n_rows=0)

    close = df["adj_close"] if "adj_close" in df.columns else df["close"]
    n_missing_close = int(close.isna().sum())

    volume = df["volume"] if "volume" in df.columns else pd.Series(dtype=float)
    n_zero_volume = int((volume == 0).sum())

    returns = close.pct_change()
    extreme_mask = returns.abs() > jump_threshold
    extreme_dates = [d.date() for d in df.index[extreme_mask.fillna(False)]]

    idx = df.index
    return TickerQuality(
        ticker=ticker,
        n_rows=n_rows,
        start=idx.min().date(),
        end=idx.max().date(),
        n_missing_close=n_missing_close,
        pct_missing=n_missing_close / n_rows,
        n_zero_volume=n_zero_volume,
        n_extreme_moves=int(extreme_mask.sum()),
        extreme_dates=extreme_dates,
    )


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

def _write_ticker(df: pd.DataFrame, ticker: str, price_dir: Path) -> Path:
    price_dir.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.insert(0, "ticker", ticker)
    out = out.reset_index()  # 'date' becomes a column
    path = price_dir / f"{ticker}.parquet"
    out.to_parquet(path, index=False)
    return path


def available_tickers(price_dir: Path | None = None) -> list[str]:
    """Tickers currently present in the Parquet store."""
    price_dir = Path(price_dir or default_price_dir())
    if not price_dir.exists():
        return []
    return sorted(p.stem for p in price_dir.glob("*.parquet"))


# --------------------------------------------------------------------------- #
# Download orchestration
# --------------------------------------------------------------------------- #

def _batches(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def download_prices(
    tickers: Iterable[str],
    start: str | date,
    end: str | date,
    *,
    price_dir: Path | None = None,
    batch_size: int = 100,
    max_retries: int = 3,
    pause: float = 1.0,
    jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
    auto_adjust: bool = False,
    fetcher: Callable[..., pd.DataFrame] | None = None,
    resolver: AliasResolver | None = None,
) -> QualityReport:
    """Batch-download daily prices for ``tickers`` into the Parquet store.

    Each requested symbol is resolved to its **current** symbol via
    ``resolver`` before fetching (Yahoo only serves history under the current
    symbol), and stored under that current symbol. A requested symbol is
    considered *downloaded* if its current symbol returned data — so a
    historical alias like ``FB`` counts as covered once ``META`` is fetched.

    Failed tickers (no data — typically delisted) are logged and reported, not
    raised. Returns a :class:`QualityReport` covering the whole run.
    """
    price_dir = Path(price_dir or default_price_dir())
    fetch = fetcher or _fetch_yf
    resolver = resolver or load_alias_resolver()

    # De-dup requested symbols (normalized), then map each to its current symbol.
    norm = list(dict.fromkeys(normalize_ticker(t) for t in tickers))
    requested = len(norm)
    canon_of = {t: normalize_ticker(resolver.to_current(t)) for t in norm}
    canon_list = list(dict.fromkeys(canon_of.values()))
    log.info("Downloading %d requested (%d unique current symbols) from %s to %s",
             requested, len(canon_list), start, end)

    got: set[str] = set()
    per_ticker: list[TickerQuality] = []

    for bi, batch in enumerate(_batches(canon_list, batch_size), 1):
        raw: pd.DataFrame | None = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = fetch(batch, start, end, auto_adjust)
                break
            except Exception as exc:  # network / rate-limit / transient
                log.warning(
                    "batch %d attempt %d/%d failed: %s",
                    bi, attempt, max_retries, exc,
                )
                if attempt < max_retries:
                    time.sleep(pause * attempt)  # linear backoff
        if raw is None:
            log.error("batch %d permanently failed; %d symbols unavailable",
                      bi, len(batch))
            continue

        for t in batch:
            sub = _extract_ticker(raw, t)
            if sub is None:
                log.info("no data for %s (likely delisted / unavailable)", t)
                continue
            _write_ticker(sub, t, price_dir)
            per_ticker.append(validate_ticker(sub, t, jump_threshold))
            got.add(t)

        if bi * batch_size < len(canon_list):
            time.sleep(pause)  # be polite between batches

    failed = [t for t in norm if canon_of[t] not in got]
    expected_missing = [
        t for t in failed
        if resolver.is_delisted(t) or resolver.is_delisted(canon_of[t])
    ]
    downloaded = requested - len(failed)

    report = QualityReport(
        requested=requested,
        downloaded=downloaded,
        failed=failed,
        per_ticker=per_ticker,
        expected_missing=expected_missing,
    )
    log.info("Done: %d/%d covered, %.1f%% missing",
             downloaded, requested, report.missing_rate * 100)
    return report


# --------------------------------------------------------------------------- #
# Read API
# --------------------------------------------------------------------------- #

def _coerce_ts(value: str | date | datetime | None) -> pd.Timestamp | None:
    return None if value is None else pd.Timestamp(value)


def get_prices(
    tickers: str | Iterable[str],
    start: str | date | None = None,
    end: str | date | None = None,
    *,
    field: str | None = None,
    price_dir: Path | None = None,
    resolver: AliasResolver | None = None,
) -> pd.DataFrame:
    """Read prices back from the store.

    Parameters
    ----------
    tickers:
        One ticker or an iterable of tickers. Each is resolved through
        ``resolver`` to its current symbol to locate the stored file, so both
        historical and current aliases work (``get_prices("FB")`` and
        ``get_prices("META")`` return the same underlying series). Rows are
        labelled with the *requested* symbol so callers that key off the
        historical ticker (e.g. a point-in-time universe) line up.
    start, end:
        Optional inclusive date bounds.
    field:
        If given (e.g. ``"adj_close"``), return a *wide* frame indexed by date
        with one column per ticker. Otherwise return a tidy *long* frame with
        columns ``[date, ticker, open, high, low, close, adj_close, volume]``.

    Missing tickers (no file in the store) are skipped with a warning.
    """
    price_dir = Path(price_dir or default_price_dir())
    resolver = resolver or load_alias_resolver()
    if isinstance(tickers, str):
        tickers = [tickers]
    # Preserve requested order, de-dup requested labels.
    labels = list(dict.fromkeys(normalize_ticker(t) for t in tickers))

    lo, hi = _coerce_ts(start), _coerce_ts(end)
    frames: list[pd.DataFrame] = []
    for label in labels:
        canon = normalize_ticker(resolver.to_current(label))
        path = price_dir / f"{canon}.parquet"
        if not path.exists():
            log.warning("no price file for %s (current symbol %s); skipping",
                        label, canon)
            continue
        d = pd.read_parquet(path)
        d["date"] = pd.to_datetime(d["date"])
        d["ticker"] = label  # relabel to the requested (historical) symbol
        if lo is not None:
            d = d[d["date"] >= lo]
        if hi is not None:
            d = d[d["date"] <= hi]
        frames.append(d)

    if not frames:
        cols = ["date", "ticker", *_OHLCV]
        return pd.DataFrame(columns=cols)

    long = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])

    if field is None:
        return long.reset_index(drop=True)

    if field not in long.columns:
        raise ValueError(
            f"unknown field {field!r}; choose from {_OHLCV}"
        )
    wide = long.pivot(index="date", columns="ticker", values=field).sort_index()
    wide.columns.name = None
    return wide


# --------------------------------------------------------------------------- #
# Universe integration
# --------------------------------------------------------------------------- #

def universe_symbols(
    start: str | date,
    end: str | date,
    *,
    sample_dates: Sequence[str] | None = None,
    cache_dir: Path | None = None,
) -> list[str]:
    """The union of tickers that were *ever* in the index over ``[start, end]``.

    Combines the point-in-time membership at several sampled dates with every
    add/remove symbol from the change log inside the window, giving the full
    (survivorship-bias-free) download set.
    """
    data = load_universe(cache_dir=cache_dir)
    lo, hi = _coerce_ts(start).date(), _coerce_ts(end).date()

    symbols: set[str] = set()
    if sample_dates is None:
        # Sample the membership roughly annually across the window.
        years = range(lo.year, hi.year + 1)
        sample_dates = [f"{y}-06-30" for y in years]
    for d in sample_dates:
        symbols.update(data.members_as_of(d))

    for ch in data.changes:
        if lo <= ch.date <= hi:
            if ch.added_ticker:
                symbols.add(ch.added_ticker)
            if ch.removed_ticker:
                symbols.add(ch.removed_ticker)

    return sorted(symbols)

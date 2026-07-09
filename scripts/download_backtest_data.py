"""One-off: download the full 2014-2025 price history for the backtest.

Downloads every ticker that was ever in the S&P 500 over 2015-2025 (the
survivorship-bias-free union) plus the SPY benchmark, from 2014-01-01 (to give
the 12-month momentum lookback a full year of history before the 2015 start)
through 2025-12-31. Overwrites the existing per-ticker Parquet store.
"""

from __future__ import annotations

import logging
import sys

from quantlab.data.prices import download_prices, universe_symbols

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

START, END = "2014-01-01", "2026-01-01"


def main() -> None:
    syms = universe_symbols("2015-01-01", "2025-12-31")
    # SPY is the benchmark, not an index member; add it explicitly.
    tickers = sorted(set(syms) | {"SPY"})
    print(f"Downloading {len(tickers)} tickers ({START} -> {END})", flush=True)
    report = download_prices(tickers, START, END, batch_size=100, pause=1.0)
    print(report.summary(), flush=True)


if __name__ == "__main__":
    main()

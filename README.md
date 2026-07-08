# quant-factor-lab

An industrial-grade Python platform for quantitative factor research.

## Vision

`quant-factor-lab` is an end-to-end quantitative research platform that covers
the **full pipeline**:

```
data  →  factors  →  backtest  →  risk  →  portfolio  →  monitor  →  reports
```

The goal is a single, reproducible environment where an idea can travel from raw
data all the way to a risk-managed, optimized portfolio — and then be monitored
for decay in production — without ever leaking future information or falling for
survivorship bias.

Each stage is a first-class, independently testable subpackage:

| Stage | Package | Responsibility |
|-------|---------|----------------|
| Data | `quantlab.data` | Data pipeline and **point-in-time** storage — the survivorship-bias-free foundation. |
| Factors | `quantlab.factors` | Factor library with a common compute interface and neutralization utilities. |
| Backtest | `quantlab.backtest` | Event-driven backtesting engine with realistic costs and no look-ahead. |
| Risk | `quantlab.risk` | Factor risk models: exposures, factor covariance, specific risk. |
| Portfolio | `quantlab.portfolio` | Portfolio construction and optimization under real-world constraints. |
| Monitor | `quantlab.monitor` | Factor-decay tracking and market-regime detection. |
| Reports | `quantlab.reports` | Automated tearsheets and research artifacts. |

## Roadmap

### Milestone 1 — Prove the engine (current)

Build a **survivorship-bias-free, point-in-time data pipeline** and use it to
**reproduce the classic momentum factor** as an end-to-end validation of the
platform. Reproducing a well-documented, well-understood factor is the cleanest
way to prove that the data → factor → backtest chain is correct before any
original research is layered on top.

Success criteria:

- Point-in-time store returns only data available as of any historical date.
- Universe construction includes delisted names (no survivorship bias).
- Cross-sectional momentum factor reproduces the expected long/short spread and
  monotonic decile ordering documented in the literature.

### Beyond Milestone 1

- Fundamental and statistical risk models.
- Mean-variance portfolio optimization with turnover and transaction-cost terms.
- Live factor-decay and regime monitoring with automated alerting.
- Automated, reproducible report generation.

## Data Pipeline

The `quantlab.data` package is the survivorship-bias-free foundation for
everything downstream. It has two layers.

### 1. Point-in-time universe (`quantlab.data.universe`)

Most equity backtests silently use *today's* index membership for *past* dates,
so stocks that were later dropped (often the losers) vanish from history and
inflate returns. This module reconstructs membership *as it actually was*.

- Scrapes two tables from the Wikipedia article *"List of S&P 500 companies"*:
  the **current constituents** and the **selected changes** log (every
  addition/removal with an effective date).
- `get_universe(as_of_date)` walks the change log backwards from the current
  snapshot, undoing every change effective after `as_of_date`, to return the
  tickers that were truly in the index on that date.
- Known **ticker renames** (e.g. `FB → META` on 2022-06-09) are resolved both
  when returning the period-correct symbol *and* when matching change-log
  entries during reconstruction.
- Results are cached to Parquet under `data/` (git-ignored) with a staleness
  window; a stale cache is used as a fallback if a refresh fails.

```python
from quantlab.data import get_universe
get_universe("2015-06-30")          # ~507 tickers as of mid-2015 (Meta appears as "FB")
```

### 2. Price data pipeline (`quantlab.data.prices`)

Turns a (survivorship-bias-free) ticker set into a validated, per-ticker Parquet
store of daily **OHLCV + adjusted close**.

- **Batched yfinance downloads** with retry + linear backoff and a polite
  inter-batch pause. A ticker yfinance cannot serve (typically *delisted*) is
  logged and counted, never fatal.
- **Data-quality gate**: detects missing closes, zero-volume sessions, and
  implausible single-day jumps (>50% by default, *flagged for review* — not
  dropped), producing a `QualityReport` with an explicit **missing rate**.
- **Uniform read API**: `get_prices()` returns either a tidy long frame or a
  wide field panel for any subset of tickers/dates.
- **Ticker-rename resolution** (`quantlab.data.aliases`): companies that changed
  symbol while staying listed (`FB→META`, `ANTM→ELV`, `CTL→LUMN`, …) are mapped
  to their *current* symbol for download (Yahoo only serves history there) while
  historical symbols still resolve on read — `get_prices("FB")` and
  `get_prices("META")` return the same series. Renames are auto-inferred from
  the change log where possible and otherwise maintained in
  `configs/ticker_renames.yaml`, which also lists genuinely delisted names so
  the quality report separates *expected* misses from ones worth investigating.

```python
from quantlab.data import universe_symbols, download_prices, get_prices

tickers = universe_symbols("2020-01-01", "2025-12-31")   # union incl. delisted
report = download_prices(tickers, "2020-01-01", "2025-12-31")
print(report.summary())                                  # missing rate + flags
px = get_prices(["AAPL", "MSFT"], field="adj_close")     # wide panel
```

### Known limitations

The universe is honest about where it is *approximate*:

- **Change-log completeness.** Wikipedia's changes table is *"selected"* and
  gets sparser the further back you go, so reconstructed counts drift by a
  handful of names before ~2012 (recent years are accurate).
- **Dual-class ticker counts.** Companies with two share classes (`GOOGL/GOOG`,
  `FOX/FOXA`, `NWS/NWSA`) mean the *ticker* count sits slightly above 500 even
  though there are 500 *companies*.
- **Residual survivorship bias from missing prices.** A point-in-time universe
  correctly *includes* delisted names, but yfinance often has **no price
  history** for them (bankruptcies, buyouts). Those tickers show up in the
  download report's missing rate — so the bias is *surfaced and measured*
  rather than hidden, but it is not fully eliminated with a free data source. A
  licensed constituents/price dataset (e.g. CRSP) would close the remaining gap.

## Project structure

```
quant-factor-lab/
├── src/quantlab/        # the platform (one subpackage per pipeline stage)
│   ├── data/            # data pipeline & point-in-time storage
│   ├── factors/         # factor library
│   ├── backtest/        # event-driven backtest engine
│   ├── risk/            # risk models
│   ├── portfolio/       # portfolio optimization
│   ├── monitor/         # factor decay & regime detection
│   └── reports/         # automated reporting
├── tests/               # test suite
├── notebooks/           # research notebooks
├── docs/                # documentation
└── configs/             # configuration files
```

## Getting started

```bash
# Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the platform in editable mode with dev tooling
pip install -e ".[dev]"

# Run the tests
pytest
```

## Tech stack

pandas · numpy · scipy · matplotlib · yfinance · pyarrow · pydantic · requests · beautifulsoup4 · pytest · pytest-cov

## License

MIT

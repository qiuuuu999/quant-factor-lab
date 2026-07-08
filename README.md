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

pandas · numpy · scipy · matplotlib · yfinance · pyarrow · pydantic · pytest · pytest-cov

## License

MIT

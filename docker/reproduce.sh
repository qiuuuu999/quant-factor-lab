#!/usr/bin/env bash
# One-command reproduction of the full quant-factor-lab research pipeline:
# test the platform, download the price/universe data it needs, then run
# every experiment script in scripts/ in dependency order.
#
# Run via `docker compose up` (see docker-compose.yml) or directly:
#   docker build -t quant-factor-lab . && docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/reports:/app/reports" quant-factor-lab
set -euo pipefail

section() {
    echo ""
    echo "================================================================"
    echo "$1"
    echo "================================================================"
}

section "1/9  Test suite (pytest, hermetic -- no data required)"
pytest

section "2/9  Lint (ruff)"
ruff check .

section "3/9  Download price + universe data (yfinance; network required)"
python scripts/download_backtest_data.py

section "4/9  Momentum backtest (12-1 momentum vs. SPY, 2015-2025)"
python scripts/run_momentum_backtest.py

section "5/9  Factor evaluation (IC / decile / correlation, 4 factors)"
python scripts/run_factor_evaluation.py

section "6/9  Risk attribution (factor risk model + decomposition)"
python scripts/run_risk_attribution.py

section "7/9  Portfolio construction comparison (equal weight / MVO / risk parity)"
python scripts/run_portfolio_comparison.py

section "8/9  Factor health monitoring (CUSUM decay + regime fit)"
python scripts/run_factor_monitoring.py

section "9/9  Regime-adaptive backtest (4-way comparison, 2017-2025)"
python scripts/run_regime_adaptive_backtest.py

section "Done -- all artifacts written under reports/"

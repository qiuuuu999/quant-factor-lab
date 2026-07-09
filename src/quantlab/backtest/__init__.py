"""backtest — event-driven backtesting engine.

Simulates strategies by replaying market events in chronological order through a
portfolio, execution, and accounting loop. The event-driven design mirrors live
trading, models transaction costs and slippage, and guarantees no look-ahead by
construction. Used to validate factors and full strategies against history.
"""

from __future__ import annotations

from quantlab.backtest.costs import CostModel, load_cost_model
from quantlab.backtest.engine import (
    BacktestResult,
    SkippedOrder,
    Trade,
    rebalance_calendar,
    run_backtest,
)

__all__ = [
    "CostModel",
    "load_cost_model",
    "BacktestResult",
    "SkippedOrder",
    "Trade",
    "rebalance_calendar",
    "run_backtest",
]

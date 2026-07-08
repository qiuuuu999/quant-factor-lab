"""backtest — event-driven backtesting engine.

Simulates strategies by replaying market events in chronological order through a
portfolio, execution, and accounting loop. The event-driven design mirrors live
trading, models transaction costs and slippage, and guarantees no look-ahead by
construction. Used to validate factors and full strategies against history.
"""

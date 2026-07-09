"""factors — the factor library.

Defines the common interface for computing cross-sectional and time-series
factors (value, momentum, quality, volatility, ...) from point-in-time data,
along with utilities for winsorization, standardization, and neutralization.
Factors produced here feed the backtest, risk, and portfolio layers.
"""

from quantlab.factors.base import Factor, LookaheadBiasError
from quantlab.factors.momentum import MomentumFactor
from quantlab.factors.low_volatility import LowVolatilityFactor
from quantlab.factors.reversal import ShortTermReversalFactor
from quantlab.factors.liquidity import AmihudIlliquidityFactor
from quantlab.factors.preprocess import deciles, winsorize, zscore

__all__ = [
    "Factor",
    "LookaheadBiasError",
    "MomentumFactor",
    "LowVolatilityFactor",
    "ShortTermReversalFactor",
    "AmihudIlliquidityFactor",
    "winsorize",
    "zscore",
    "deciles",
]

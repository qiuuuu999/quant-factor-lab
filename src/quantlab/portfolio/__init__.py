"""portfolio — portfolio construction and optimization.

Translates factor signals and risk-model estimates into target holdings via
mean-variance and other optimizers, subject to real-world constraints (leverage,
turnover, position and sector limits, transaction costs). Bridges alpha research
and executable portfolios.
"""

from quantlab.portfolio.optimizer import MeanVarianceOptimizer, OptimizationResult
from quantlab.portfolio.risk_parity import RiskParityOptimizer, RiskParityResult
from quantlab.portfolio.turnover_control import turnover, turnover_penalty
from quantlab.portfolio.regime_adaptive import (
    build_regime_adaptive_weights,
    build_static_multifactor_weights,
    composite_score,
    pit_regime_by_date,
    regime_conditioned_factor_weights,
)

__all__ = [
    "MeanVarianceOptimizer",
    "OptimizationResult",
    "RiskParityOptimizer",
    "RiskParityResult",
    "turnover",
    "turnover_penalty",
    "pit_regime_by_date",
    "regime_conditioned_factor_weights",
    "composite_score",
    "build_regime_adaptive_weights",
    "build_static_multifactor_weights",
]

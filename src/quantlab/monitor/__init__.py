"""monitor — factor decay and regime detection.

Tracks the live health of deployed factors: information-coefficient decay,
turnover, crowding, and performance drift, and detects shifts in market regime
that may invalidate a factor's edge. Emits signals that trigger re-research,
re-weighting, or retirement of factors.
"""

from quantlab.monitor.decay import (
    CusumResult,
    FactorHealthReport,
    cusum_test,
    factor_health_report,
    rolling_ic,
)
from quantlab.monitor.regime import (
    REGIMES,
    RegimeICMatrix,
    classify_regime,
    factor_regime_matrix,
    regime_as_of,
)

__all__ = [
    "rolling_ic",
    "CusumResult",
    "cusum_test",
    "FactorHealthReport",
    "factor_health_report",
    "REGIMES",
    "classify_regime",
    "regime_as_of",
    "RegimeICMatrix",
    "factor_regime_matrix",
]

"""risk — risk models.

Estimates the covariance structure of asset returns via factor risk models
(fundamental and statistical), providing exposures, factor covariances, and
specific risk. These estimates drive portfolio optimization constraints and
ex-ante risk attribution and forecasting.
"""

from quantlab.risk.covariance import (
    CovarianceEstimator,
    EWMACovariance,
    LedoitWolfCovariance,
    SampleCovariance,
    estimate_covariance,
)
from quantlab.risk.factor_risk import (
    FactorRiskDecomposition,
    FactorRiskModel,
    build_exposure_panel,
    cross_sectional_regression,
    decompose_portfolio_risk,
)
from quantlab.risk.attribution import ExposureProfile, portfolio_factor_exposure

__all__ = [
    "CovarianceEstimator",
    "SampleCovariance",
    "EWMACovariance",
    "LedoitWolfCovariance",
    "estimate_covariance",
    "build_exposure_panel",
    "cross_sectional_regression",
    "FactorRiskModel",
    "FactorRiskDecomposition",
    "decompose_portfolio_risk",
    "ExposureProfile",
    "portfolio_factor_exposure",
]

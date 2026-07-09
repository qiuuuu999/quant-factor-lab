"""Tests for the risk module (quantlab.risk).

Hermetic: covariance estimators are checked for convergence against simulated
data drawn from a *known* true covariance matrix, and the factor risk
decomposition is checked against a synthetic single-factor model whose true
factor/specific variance split is known by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.risk.attribution import ExposureProfile, portfolio_factor_exposure
from quantlab.risk.covariance import (
    EWMACovariance,
    LedoitWolfCovariance,
    SampleCovariance,
    estimate_covariance,
)
from quantlab.risk.factor_risk import (
    FactorRiskModel,
    cross_sectional_regression,
    decompose_portfolio_risk,
)

# --------------------------------------------------------------------------- #
# Covariance estimators: convergence to a known true covariance
# --------------------------------------------------------------------------- #

_TRUE_VOLS = np.array([0.10, 0.15, 0.20, 0.12])
_TRUE_CORR = np.array([
    [1.00, 0.40, 0.10, 0.00],
    [0.40, 1.00, 0.20, 0.10],
    [0.10, 0.20, 1.00, 0.30],
    [0.00, 0.10, 0.30, 1.00],
])
_TRUE_COV = np.outer(_TRUE_VOLS, _TRUE_VOLS) * _TRUE_CORR
_ASSETS = ["A", "B", "C", "D"]


def _simulate(n_periods: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(np.zeros(len(_ASSETS)), _TRUE_COV, size=n_periods)
    return pd.DataFrame(x, columns=_ASSETS)


def test_sample_covariance_converges_to_true_covariance():
    returns = _simulate(20_000, seed=1)
    est = SampleCovariance().estimate(returns)
    assert np.allclose(est.to_numpy(), _TRUE_COV, atol=0.0015)


def test_ewma_covariance_with_long_halflife_converges_to_true_covariance():
    returns = _simulate(20_000, seed=2)
    # A halflife much longer than the sample gives near-uniform weights, so
    # EWMA should behave like the plain sample covariance.
    est = EWMACovariance(halflife=50_000).estimate(returns)
    assert np.allclose(est.to_numpy(), _TRUE_COV, atol=0.0025)


def test_ewma_covariance_upweights_recent_regime():
    # First half: low vol regime; second half: 3x vol regime. A short halflife
    # should track the recent (high-vol) regime, not the blended average.
    rng = np.random.default_rng(3)
    n = 2000
    low = rng.multivariate_normal(np.zeros(len(_ASSETS)), _TRUE_COV, size=n)
    high = rng.multivariate_normal(np.zeros(len(_ASSETS)), _TRUE_COV * 9.0, size=n)
    returns = pd.DataFrame(np.vstack([low, high]), columns=_ASSETS)

    short = EWMACovariance(halflife=50).estimate(returns)
    long = EWMACovariance(halflife=50_000).estimate(returns)
    # The short-halflife estimate should sit much closer to the recent (high
    # vol) regime's variance than the long-halflife (~blended) estimate.
    assert short.loc["A", "A"] > long.loc["A", "A"]
    assert short.loc["A", "A"] == pytest.approx(_TRUE_COV[0, 0] * 9.0, rel=0.35)


def test_ledoit_wolf_converges_to_true_covariance_with_ample_data():
    returns = _simulate(20_000, seed=4)
    lw = LedoitWolfCovariance()
    est = lw.estimate(returns)
    assert np.allclose(est.to_numpy(), _TRUE_COV, atol=0.005)
    # With T >> N and a well-conditioned true covariance, shrinkage should be small.
    assert 0.0 <= lw.last_shrinkage_ < 0.2


def test_ledoit_wolf_is_invertible_when_sample_covariance_is_not():
    # More assets than observations: the classic ill-conditioned regime this
    # estimator exists for.
    n_assets, n_periods = 50, 20
    rng = np.random.default_rng(5)
    true_cov = np.eye(n_assets) * 0.04
    x = rng.multivariate_normal(np.zeros(n_assets), true_cov, size=n_periods)
    returns = pd.DataFrame(x, columns=[f"T{i}" for i in range(n_assets)])

    sample = SampleCovariance().estimate(returns).to_numpy()
    # Singular (rank <= n_periods - 1 < n_assets): near-zero smallest eigenvalue.
    sample_eigvals = np.linalg.eigvalsh(sample)
    assert sample_eigvals.min() < 1e-8

    lw = LedoitWolfCovariance()
    shrunk = lw.estimate(returns).to_numpy()
    shrunk_eigvals = np.linalg.eigvalsh(shrunk)
    # Full rank / invertible, unlike the sample covariance.
    assert shrunk_eigvals.min() > 1e-8
    assert 0.0 < lw.last_shrinkage_ <= 1.0
    # Well-conditioned: condition number is dramatically better than the
    # (numerically singular) sample estimate.
    assert np.linalg.cond(shrunk) < np.linalg.cond(sample + np.eye(n_assets) * 1e-12)


def test_estimate_covariance_dispatch_matches_direct_call():
    returns = _simulate(500, seed=6)
    direct = SampleCovariance(ddof=1).estimate(returns)
    dispatched = estimate_covariance(returns, method="sample", ddof=1)
    pd.testing.assert_frame_equal(direct, dispatched)

    with pytest.raises(ValueError):
        estimate_covariance(returns, method="not_a_method")


def test_covariance_estimators_require_min_periods():
    returns = _simulate(1, seed=7)
    with pytest.raises(ValueError):
        SampleCovariance(min_periods=5).estimate(returns)


# --------------------------------------------------------------------------- #
# Cross-sectional regression
# --------------------------------------------------------------------------- #

def test_cross_sectional_regression_recovers_exact_noiseless_factor_return():
    tickers = [f"T{i:02d}" for i in range(10)]
    b = np.linspace(-1.5, 1.5, 10)
    exposures = pd.DataFrame({"style": b}, index=tickers)
    true_factor_return = 0.03
    intercept = 0.01
    returns = pd.Series(intercept + b * true_factor_return, index=tickers)

    factor_returns, resid = cross_sectional_regression(returns, exposures)
    assert factor_returns["style"] == pytest.approx(true_factor_return)
    assert resid.abs().max() < 1e-10


def test_cross_sectional_regression_returns_nan_below_min_names():
    tickers = ["A", "B", "C"]
    exposures = pd.DataFrame({"style": [1.0, 2.0, 3.0]}, index=tickers)
    returns = pd.Series([0.01, 0.02, 0.03], index=tickers)
    factor_returns, resid = cross_sectional_regression(returns, exposures, min_names=10)
    assert factor_returns.isna().all()
    assert resid.isna().all()


# --------------------------------------------------------------------------- #
# Single-factor model: risk decomposition recovers the true variance split
# --------------------------------------------------------------------------- #

def test_single_factor_model_recovers_variance_decomposition():
    rng = np.random.default_rng(42)
    n_assets, n_periods = 30, 800
    tickers = [f"T{i:02d}" for i in range(n_assets)]
    dates = pd.date_range("2000-01-31", periods=n_periods, freq="ME")

    b = np.linspace(-1.5, 1.5, n_assets)             # fixed factor exposure, mean 0
    exposures = pd.DataFrame({"style": b}, index=tickers)
    exposures_by_date = {dt: exposures for dt in dates}

    sigma_f = 0.06
    sigma_i = rng.uniform(0.02, 0.05, n_assets)       # per-asset specific vol

    f = rng.normal(0.0, sigma_f, n_periods)
    eps = rng.normal(0.0, 1.0, (n_periods, n_assets)) * sigma_i
    raw_returns = b[None, :] * f[:, None] + eps
    forward_returns = pd.DataFrame(raw_returns, index=dates, columns=tickers)

    model = FactorRiskModel().fit(exposures_by_date, forward_returns)
    factor_cov = model.factor_covariance()
    specific_var = model.specific_variance()

    # Factor variance recovered close to the true sigma_f^2.
    assert factor_cov.loc["style", "style"] == pytest.approx(sigma_f ** 2, rel=0.3)

    # Specific variance recovered close to the true per-asset sigma_i^2.
    recovered = specific_var.reindex(tickers).to_numpy()
    assert np.allclose(recovered, sigma_i ** 2, rtol=0.4)

    weights = pd.Series(1.0 / n_assets, index=tickers)
    decomp = decompose_portfolio_risk(weights, exposures, factor_cov, specific_var)

    true_exposure = float(weights.to_numpy() @ b)
    true_factor_var = true_exposure ** 2 * sigma_f ** 2
    true_specific_var = float(np.sum((weights.to_numpy() ** 2) * sigma_i ** 2))
    true_total = true_factor_var + true_specific_var

    assert decomp.factor_variance == pytest.approx(true_factor_var, rel=0.3)
    assert decomp.specific_variance == pytest.approx(true_specific_var, rel=0.3)
    assert decomp.total_variance == pytest.approx(true_total, rel=0.3)

    # Euler decomposition: per-factor contributions sum exactly to factor_variance.
    assert decomp.factor_contributions.sum() == pytest.approx(decomp.factor_variance)


def test_decompose_portfolio_risk_euler_contributions_sum_to_factor_variance():
    # Two-factor sanity check with a hand-built (non-diagonal) factor covariance.
    tickers = ["A", "B", "C"]
    exposures = pd.DataFrame(
        {"mom": [1.0, -0.5, 0.2], "vol": [-0.3, 0.8, 0.1]}, index=tickers
    )
    factor_cov = pd.DataFrame(
        [[0.02, 0.005], [0.005, 0.01]], index=["mom", "vol"], columns=["mom", "vol"]
    )
    specific_var = pd.Series([0.001, 0.002, 0.0015], index=tickers)
    weights = pd.Series([0.5, 0.3, 0.2], index=tickers)

    decomp = decompose_portfolio_risk(weights, exposures, factor_cov, specific_var)

    e = np.array([weights.to_numpy() @ exposures["mom"].to_numpy(),
                  weights.to_numpy() @ exposures["vol"].to_numpy()])
    expected_factor_var = float(e @ factor_cov.to_numpy() @ e)
    expected_specific_var = float((weights.to_numpy() ** 2 * specific_var.to_numpy()).sum())

    assert decomp.factor_variance == pytest.approx(expected_factor_var)
    assert decomp.specific_variance == pytest.approx(expected_specific_var)
    assert decomp.factor_contributions.sum() == pytest.approx(decomp.factor_variance)
    assert decomp.total_variance == pytest.approx(expected_factor_var + expected_specific_var)


# --------------------------------------------------------------------------- #
# Attribution: portfolio factor-exposure profile
# --------------------------------------------------------------------------- #

def test_portfolio_factor_exposure_is_weighted_average():
    tickers = ["A", "B", "C"]
    exposures = pd.DataFrame(
        {"mom": [2.0, -1.0, 0.0], "vol": [0.0, 1.0, -2.0]}, index=tickers
    )
    weights = pd.Series([0.5, 0.25, 0.25], index=tickers)
    exposure = portfolio_factor_exposure(weights, exposures)
    assert exposure["mom"] == pytest.approx(0.5 * 2.0 + 0.25 * -1.0 + 0.25 * 0.0)
    assert exposure["vol"] == pytest.approx(0.5 * 0.0 + 0.25 * 1.0 + 0.25 * -2.0)


def test_portfolio_factor_exposure_ignores_names_absent_from_exposures():
    exposures = pd.DataFrame({"mom": [1.0, 2.0]}, index=["A", "B"])
    weights = pd.Series([0.5, 0.5], index=["A", "Z"])  # "Z" not in exposures
    exposure = portfolio_factor_exposure(weights, exposures)
    assert exposure["mom"] == pytest.approx(0.5 * 1.0)  # B's weight is 0, Z dropped


def test_exposure_profile_build_reports_names_held_and_exposure():
    tickers = ["A", "B", "C"]
    exposures = pd.DataFrame({"mom": [1.0, -1.0, 0.5]}, index=tickers)
    weights = pd.Series([0.6, 0.0, 0.4], index=tickers)  # B unheld
    profile = ExposureProfile.build(weights, exposures, "2020-01-31")
    assert profile.n_names == 2
    assert profile.as_of == pd.Timestamp("2020-01-31")
    assert profile.exposure["mom"] == pytest.approx(0.6 * 1.0 + 0.4 * 0.5)

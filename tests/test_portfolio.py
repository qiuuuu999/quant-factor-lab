"""Tests for the portfolio module (quantlab.portfolio).

Hermetic: the mean-variance optimizer is checked against a hand-derived
closed-form solution for a two-asset, constraint-free problem (Lagrangian
stationarity has an exact algebraic solution when the only constraint is the
budget constraint); constraint enforcement (cap, budget) and turnover
monotonicity are checked on small synthetic problems; risk parity is checked
against the textbook equal-risk-contribution property.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.portfolio.optimizer import MeanVarianceOptimizer
from quantlab.portfolio.risk_parity import RiskParityOptimizer
from quantlab.portfolio.turnover_control import turnover, turnover_penalty

# --------------------------------------------------------------------------- #
# Mean-variance optimizer: two-asset closed-form solution
# --------------------------------------------------------------------------- #

def _closed_form_two_asset(alpha: np.ndarray, sigma: np.ndarray, k: float) -> np.ndarray:
    """Exact solution of max a'w - (k/2)w'Sw s.t. 1'w=1 (Lagrangian stationarity)."""
    sigma_inv = np.linalg.inv(sigma)
    ones = np.ones(2)
    gamma = (ones @ sigma_inv @ alpha - k) / (ones @ sigma_inv @ ones)
    return (1.0 / k) * sigma_inv @ (alpha - gamma * ones)


def test_mean_variance_matches_two_asset_closed_form_solution():
    alpha = pd.Series({"A": 0.08, "B": 0.05})
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    k = 4.0
    expected = _closed_form_two_asset(alpha.to_numpy(), cov.to_numpy(), k)

    opt = MeanVarianceOptimizer(risk_aversion=k, long_only=False, max_weight=None)
    result = opt.optimize(alpha, cov)

    assert result.success
    assert result.weights.reindex(["A", "B"]).to_numpy() == pytest.approx(expected, abs=1e-5)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_mean_variance_closed_form_holds_across_risk_aversions():
    alpha = pd.Series({"A": 0.10, "B": -0.02})
    cov = pd.DataFrame(
        [[0.06, -0.015], [-0.015, 0.03]], index=["A", "B"], columns=["A", "B"]
    )
    for k in (0.5, 2.0, 10.0):
        expected = _closed_form_two_asset(alpha.to_numpy(), cov.to_numpy(), k)
        opt = MeanVarianceOptimizer(risk_aversion=k, long_only=False, max_weight=None)
        result = opt.optimize(alpha, cov)
        assert result.weights.reindex(["A", "B"]).to_numpy() == pytest.approx(expected, abs=1e-5)


# --------------------------------------------------------------------------- #
# Mean-variance optimizer: constraints
# --------------------------------------------------------------------------- #

def _five_name_problem():
    tickers = [f"T{i}" for i in range(5)]
    # Alpha heavily skewed to T0 so the unconstrained solve would overweight it.
    alpha = pd.Series([0.30, 0.02, 0.02, 0.02, 0.02], index=tickers)
    cov = pd.DataFrame(np.eye(5) * 0.04, index=tickers, columns=tickers)
    return alpha, cov


def test_mean_variance_respects_max_weight_and_budget_constraint():
    alpha, cov = _five_name_problem()
    opt = MeanVarianceOptimizer(risk_aversion=1.0, long_only=True, max_weight=0.30)
    result = opt.optimize(alpha, cov)

    assert result.success
    assert (result.weights <= 0.30 + 1e-6).all()
    assert (result.weights >= -1e-8).all()
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    # The cap should bind on the high-alpha name (unconstrained solve would
    # overweight it well past 30% given its outsized alpha).
    assert result.weights["T0"] == pytest.approx(0.30, abs=1e-4)


def test_mean_variance_long_only_has_no_negative_weights():
    alpha, cov = _five_name_problem()
    alpha["T1"] = -0.10  # a name with negative alpha
    opt = MeanVarianceOptimizer(risk_aversion=2.0, long_only=True, max_weight=None)
    result = opt.optimize(alpha, cov)
    assert (result.weights >= -1e-8).all()
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_mean_variance_respects_target_volatility():
    alpha, cov = _five_name_problem()
    # 5 uncorrelated equal-variance (0.04) names: minimum achievable variance
    # is 0.04/5 = 0.008 (equal weight); the unconstrained (risk_aversion~0)
    # solve concentrates in the high-alpha name, reaching var ~= 0.04. Pick a
    # target strictly between the two so the constraint is both feasible and
    # binding.
    target_vol = 0.12  # var = 0.0144
    opt = MeanVarianceOptimizer(
        risk_aversion=0.01, long_only=True, target_volatility=target_vol,
    )
    result = opt.optimize(alpha, cov)
    assert result.success
    realized_var = float(result.weights.to_numpy() @ cov.to_numpy() @ result.weights.to_numpy())
    assert realized_var <= target_vol ** 2 + 1e-6
    # The constraint should actually bind (not be slack) given the alpha
    # tilt: without it the optimizer would push var well above the target.
    assert realized_var == pytest.approx(target_vol ** 2, abs=1e-4)


def test_max_weight_infeasible_with_budget_constraint_raises():
    alpha, cov = _five_name_problem()
    # 3 names, cap 0.1 each -> max attainable sum is 0.3 < 1: infeasible.
    with pytest.raises(ValueError):
        MeanVarianceOptimizer(long_only=True, max_weight=0.1).optimize(
            alpha.iloc[:3], cov.iloc[:3, :3]
        )


# --------------------------------------------------------------------------- #
# Mean-variance optimizer: turnover control
# --------------------------------------------------------------------------- #

def test_turnover_penalty_monotonically_shrinks_realized_turnover():
    tickers = [f"T{i}" for i in range(4)]
    alpha = pd.Series([0.12, -0.05, 0.08, 0.01], index=tickers)
    cov = pd.DataFrame(
        [[0.05, 0.01, 0.00, 0.00],
         [0.01, 0.04, 0.01, 0.00],
         [0.00, 0.01, 0.06, 0.02],
         [0.00, 0.00, 0.02, 0.03]],
        index=tickers, columns=tickers,
    )
    previous_weights = pd.Series(0.25, index=tickers)  # equal-weight starting book

    lambdas = [0.0, 0.01, 0.05, 0.2, 1.0]
    turnovers = []
    for lam in lambdas:
        opt = MeanVarianceOptimizer(risk_aversion=1.0, long_only=True, turnover_penalty=lam)
        result = opt.optimize(alpha, cov, previous_weights=previous_weights)
        assert result.success
        turnovers.append(result.turnover)

    # Non-increasing as lambda grows (allow tiny numerical slack).
    for prev_t, next_t in zip(turnovers, turnovers[1:]):
        assert next_t <= prev_t + 1e-6
    # And it should actually bind somewhere: the largest lambda trades much
    # less than the unconstrained (lambda=0) solve.
    assert turnovers[-1] < turnovers[0]


def test_turnover_penalty_zero_lambda_ignores_previous_weights():
    alpha, cov = _five_name_problem()
    previous_weights = pd.Series(0.20, index=alpha.index)
    opt = MeanVarianceOptimizer(risk_aversion=1.0, long_only=True, turnover_penalty=0.0)
    with_prev = opt.optimize(alpha, cov, previous_weights=previous_weights)
    without_prev = opt.optimize(alpha, cov, previous_weights=None)
    assert with_prev.weights.to_numpy() == pytest.approx(without_prev.weights.to_numpy(), abs=1e-5)


# --------------------------------------------------------------------------- #
# turnover_control helpers
# --------------------------------------------------------------------------- #

def test_turnover_from_all_cash_is_sum_of_absolute_weights():
    w = pd.Series({"A": 0.6, "B": 0.4})
    assert turnover(w) == pytest.approx(1.0)
    assert turnover(w, pd.Series(dtype=float)) == pytest.approx(1.0)


def test_turnover_handles_disjoint_names():
    w = pd.Series({"A": 0.5, "B": 0.5})
    prev = pd.Series({"B": 0.5, "C": 0.5})
    # A: 0.5-0=0.5, B: 0.5-0.5=0, C: 0-0.5=0.5 -> total 1.0
    assert turnover(w, prev) == pytest.approx(1.0)


def test_turnover_penalty_scales_linearly_with_lambda():
    w = np.array([0.5, 0.3, 0.2])
    prev = np.array([0.2, 0.3, 0.5])
    base = turnover_penalty(w, prev, 1.0)
    assert turnover_penalty(w, prev, 3.0) == pytest.approx(base * 3.0)
    assert turnover_penalty(w, prev, 0.0) == 0.0


# --------------------------------------------------------------------------- #
# Risk parity
# --------------------------------------------------------------------------- #

def test_risk_parity_equal_vol_uncorrelated_gives_equal_weights():
    tickers = [f"T{i}" for i in range(4)]
    cov = pd.DataFrame(np.eye(4) * 0.04, index=tickers, columns=tickers)
    result = RiskParityOptimizer().optimize(cov)
    assert result.success
    assert result.weights.to_numpy() == pytest.approx(np.full(4, 0.25), abs=1e-4)


def test_risk_parity_equalizes_risk_contributions():
    tickers = ["A", "B", "C"]
    vols = np.array([0.10, 0.20, 0.35])
    corr = np.array([
        [1.0, 0.3, 0.1],
        [0.3, 1.0, 0.4],
        [0.1, 0.4, 1.0],
    ])
    cov_arr = np.outer(vols, vols) * corr
    cov = pd.DataFrame(cov_arr, index=tickers, columns=tickers)

    result = RiskParityOptimizer().optimize(cov)
    assert result.success
    # Every name's share of total portfolio variance should be ~1/n.
    assert result.risk_contributions.to_numpy() == pytest.approx(
        np.full(3, 1.0 / 3), abs=1e-3
    )
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    # Higher-vol names get smaller weight to equalize risk contribution.
    assert result.weights["A"] > result.weights["B"] > result.weights["C"]


def test_risk_parity_respects_max_weight():
    tickers = [f"T{i}" for i in range(4)]
    # One much riskier name that ERC would otherwise underweight heavily, and
    # a cap that forces the other three to absorb more risk than they'd like.
    vols = np.array([0.05, 0.05, 0.05, 0.60])
    cov = pd.DataFrame(np.diag(vols ** 2), index=tickers, columns=tickers)
    result = RiskParityOptimizer(max_weight=0.5).optimize(cov)
    assert result.success
    assert (result.weights <= 0.5 + 1e-6).all()
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

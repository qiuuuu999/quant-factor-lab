"""optimizer — mean-variance portfolio construction (Markowitz, SLSQP).

Turns a factor signal (``alpha``, one score per name) and a risk model
(``covariance``, from :mod:`quantlab.risk.covariance` — typically
:class:`~quantlab.risk.covariance.LedoitWolfCovariance`, since the asset-level
cross-section is usually large relative to the history available) into target
portfolio weights by solving

    maximize    alpha' w  -  (risk_aversion / 2) * w' Sigma w  -  lambda * ||w - w_prev||_1
    subject to  sum(w) == 1                                    (fully invested)
                0 <= w_i <= max_weight   (long_only)            or
                -max_weight <= w_i <= max_weight   (long_only=False)
                w' Sigma w <= target_volatility ** 2            (optional)

via ``scipy.optimize.minimize(method="SLSQP")``. The turnover term is handled
by :mod:`quantlab.portfolio.turnover_control`; see that module's docstring for
why it is an L1 (not L2) penalty.

Why not equal weight
---------------------
Equal-weighting the selected names — the baseline in
``scripts/run_momentum_backtest.py`` — implicitly assumes every name in the
selection carries the same conviction (alpha magnitude) *and* the same
marginal risk. Neither is generally true: two momentum names in the same
decile can have very different realized vol and very different correlation to
the rest of the book. Mean-variance optimization uses the covariance matrix
the risk module already estimates to size each position by its
return-per-unit-of-marginal-risk, shrinking (or excluding) names that are
individually attractive but redundant with — or a risk concentration
alongside — names already held, and diversifying away correlated risk an
equal-weight sleeve cannot see. The cost is added sensitivity to estimation
error in ``alpha``/``covariance``, which is why a well-conditioned covariance
estimator (Ledoit-Wolf) and turnover control both matter here in practice —
see ``docs/portfolio.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quantlab.portfolio.turnover_control import turnover_penalty

__all__ = ["OptimizationResult", "MeanVarianceOptimizer"]


@dataclass
class OptimizationResult:
    """Output of :meth:`MeanVarianceOptimizer.optimize`."""

    weights: pd.Series
    objective_value: float
    success: bool
    message: str
    turnover: float


def _objective(
    w: np.ndarray,
    alpha: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    lam: float,
    prev_w: np.ndarray | None,
) -> float:
    expected_return = float(alpha @ w)
    risk = float(w @ sigma @ w)
    obj = -expected_return + 0.5 * risk_aversion * risk
    if lam > 0.0 and prev_w is not None:
        obj += turnover_penalty(w, prev_w, lam)
    return obj


class MeanVarianceOptimizer:
    """Constrained Markowitz optimizer solved with SLSQP.

    Parameters
    ----------
    risk_aversion:
        ``k`` in the objective above. Larger values weight risk reduction more
        heavily relative to expected return; ``k -> 0`` approaches a pure
        alpha-maximizing (unconstrained-risk) solve.
    long_only:
        If ``True`` (default), weights are bounded below at ``0``. If
        ``False``, shorting is allowed down to ``-max_weight`` (or unbounded
        if ``max_weight`` is ``None``).
    max_weight:
        Per-name weight cap (e.g. ``0.02`` for a 2% single-name limit).
        ``None`` means uncapped.
    target_volatility:
        Optional ceiling on portfolio volatility (annualization-free — same
        units as ``covariance``): adds the constraint ``w'Sigma w <=
        target_volatility ** 2``. ``None`` omits the constraint.
    turnover_penalty:
        ``lambda`` in the objective above. ``0`` (default) disables turnover
        control; larger values increasingly anchor the solution to
        ``previous_weights`` passed to :meth:`optimize`. See
        :mod:`quantlab.portfolio.turnover_control`.
    """

    def __init__(
        self,
        *,
        risk_aversion: float = 1.0,
        long_only: bool = True,
        max_weight: float | None = None,
        target_volatility: float | None = None,
        turnover_penalty: float = 0.0,
    ):
        if risk_aversion < 0:
            raise ValueError("risk_aversion must be >= 0")
        if max_weight is not None and max_weight <= 0:
            raise ValueError("max_weight must be > 0")
        if target_volatility is not None and target_volatility <= 0:
            raise ValueError("target_volatility must be > 0")
        if turnover_penalty < 0:
            raise ValueError("turnover_penalty must be >= 0")
        self.risk_aversion = risk_aversion
        self.long_only = long_only
        self.max_weight = max_weight
        self.target_volatility = target_volatility
        self.turnover_penalty = turnover_penalty

    def _bounds(self, n: int) -> list[tuple[float, float]]:
        lower = 0.0 if self.long_only else -(self.max_weight if self.max_weight is not None else np.inf)
        upper = self.max_weight if self.max_weight is not None else np.inf
        if upper < lower:
            raise ValueError(
                f"max_weight ({self.max_weight}) leaves no feasible weight "
                f"(lower bound would be {lower})"
            )
        if self.long_only and self.max_weight is not None and self.max_weight * n < 1.0:
            raise ValueError(
                f"max_weight={self.max_weight} with n={n} names cannot reach "
                f"the fully-invested constraint sum(w)==1 "
                f"(max attainable sum is {self.max_weight * n:.4f})"
            )
        return [(lower, upper)] * n

    def optimize(
        self,
        alpha: pd.Series,
        covariance: pd.DataFrame,
        *,
        previous_weights: pd.Series | None = None,
    ) -> OptimizationResult:
        """Solve for the optimal weights over ``alpha.index``.

        Parameters
        ----------
        alpha:
            Expected-return / signal score per name, indexed by ticker. This
            is the universe the optimizer allocates over.
        covariance:
            ``ticker x ticker`` covariance matrix (e.g. from
            :func:`quantlab.risk.covariance.estimate_covariance`); reindexed
            to ``alpha.index``. Must cover every name in ``alpha``.
        previous_weights:
            Prior period's weights, indexed by ticker. Used both as the
            turnover-penalty anchor and (when ``turnover_penalty > 0``) to
            seed the initial guess. Names absent from ``alpha`` are ignored;
            names in ``alpha`` absent here are treated as previously unheld
            (``0``).

        Returns
        -------
        OptimizationResult
        """
        tickers = list(alpha.index)
        n = len(tickers)
        if n == 0:
            raise ValueError("alpha is empty; nothing to optimize")

        a = alpha.to_numpy(dtype=float)
        sigma = covariance.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
        if np.isnan(sigma).any():
            raise ValueError("covariance does not fully cover alpha's names")

        prev_w = None
        if previous_weights is not None:
            prev_w = previous_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
        elif self.turnover_penalty > 0.0:
            prev_w = np.zeros(n)  # anchor to an all-cash book

        bounds = self._bounds(n)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        x0 = prev_w.copy() if prev_w is not None else np.full(n, 1.0 / n)
        x0 = np.clip(x0, lo, hi)

        constraints: list[dict] = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]
        if self.target_volatility is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda w: self.target_volatility ** 2 - float(w @ sigma @ w),
            })

        result = minimize(
            _objective,
            x0,
            args=(a, sigma, self.risk_aversion, self.turnover_penalty, prev_w),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )

        weights = pd.Series(result.x, index=tickers, name="weight")
        realized_turnover = float(
            np.sum(np.abs(result.x - prev_w)) if prev_w is not None else np.sum(np.abs(result.x))
        )
        return OptimizationResult(
            weights=weights,
            objective_value=float(result.fun),
            success=bool(result.success),
            message=str(result.message),
            turnover=realized_turnover,
        )

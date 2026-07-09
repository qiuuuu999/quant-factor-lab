"""risk_parity — equal-risk-contribution weighting as a control group.

:class:`~quantlab.portfolio.optimizer.MeanVarianceOptimizer` sizes positions
by ``alpha``, so its output is only as good as the signal fed into it. Risk
parity answers a deliberately narrower question — *ignore alpha entirely, and
size each position so it contributes equally to total portfolio variance* —
which makes it a useful control group in a construction-method comparison:
any outperformance the mean-variance sleeve shows over risk parity is
attributable to the alpha signal being used, not merely to
smarter-than-equal-weight risk balancing (which risk parity also does).

For weights ``w`` and covariance ``Sigma``, name ``i``'s (marginal) risk
contribution to portfolio variance ``w'Sigma w`` is

    RC_i = w_i * (Sigma w)_i

(the Euler decomposition used identically in
:mod:`quantlab.risk.factor_risk`: ``sum_i RC_i == w'Sigma w`` exactly, since
variance is homogeneous of degree 2 in ``w``). Equal risk contribution (ERC)
solves for the ``w`` that equalizes every ``RC_i``, found here as the
long-only, fully-invested minimizer of the sum of squared deviations from the
mean risk contribution:

    minimize    sum_i (RC_i(w) - mean(RC(w))) ** 2
    subject to  sum(w) == 1,  0 <= w_i <= max_weight

via ``scipy.optimize.minimize(method="SLSQP")``, mirroring
:mod:`quantlab.portfolio.optimizer`. The objective's global minimum is exactly
``0`` at the true ERC solution regardless of the overall variance scale (both
sides of ``RC_i == RC_j`` scale together), so no normalization is needed for
the solver to converge to the right point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = ["RiskParityResult", "RiskParityOptimizer"]


@dataclass
class RiskParityResult:
    """Output of :meth:`RiskParityOptimizer.optimize`."""

    weights: pd.Series
    risk_contributions: pd.Series   # ticker -> share of total portfolio variance
    success: bool
    message: str


def _risk_contributions(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return w * (sigma @ w)


def _objective(w: np.ndarray, sigma: np.ndarray) -> float:
    rc = _risk_contributions(w, sigma)
    return float(np.sum((rc - rc.mean()) ** 2))


class RiskParityOptimizer:
    """Long-only equal-risk-contribution optimizer.

    Parameters
    ----------
    max_weight:
        Optional per-name weight cap (same semantics as
        :class:`~quantlab.portfolio.optimizer.MeanVarianceOptimizer`).
        ``None`` means uncapped (still implicitly bounded by the ERC
        objective itself, which favors diversification).
    """

    def __init__(self, *, max_weight: float | None = None):
        if max_weight is not None and max_weight <= 0:
            raise ValueError("max_weight must be > 0")
        self.max_weight = max_weight

    def optimize(self, covariance: pd.DataFrame) -> RiskParityResult:
        """Solve for equal-risk-contribution weights over ``covariance``'s names."""
        tickers = list(covariance.index)
        n = len(tickers)
        if n == 0:
            raise ValueError("covariance is empty; nothing to optimize")
        if self.max_weight is not None and self.max_weight * n < 1.0:
            raise ValueError(
                f"max_weight={self.max_weight} with n={n} names cannot reach "
                f"the fully-invested constraint sum(w)==1"
            )

        sigma = covariance.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
        upper = self.max_weight if self.max_weight is not None else 1.0
        bounds = [(0.0, upper)] * n
        x0 = np.full(n, 1.0 / n)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        result = minimize(
            _objective, x0, args=(sigma,), method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-16},
        )

        weights = pd.Series(result.x, index=tickers, name="weight")
        rc = _risk_contributions(result.x, sigma)
        total_var = float(result.x @ sigma @ result.x)
        rc_share = pd.Series(rc / total_var if total_var > 0 else rc, index=tickers, name="risk_contribution")
        return RiskParityResult(
            weights=weights,
            risk_contributions=rc_share,
            success=bool(result.success),
            message=str(result.message),
        )

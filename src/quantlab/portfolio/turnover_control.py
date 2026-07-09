"""turnover_control — penalizing unnecessary trading in portfolio construction.

A mean-variance solve re-run at every rebalance chases small, noisy shifts in
``alpha`` with a full re-optimization, and every share moved pays the cost
model in :mod:`quantlab.backtest.costs` (commission + slippage). Without a
brake, an optimizer will happily trade a name from 4.9% to 5.1% to capture a
negligible expected-return improvement, paying real transaction costs for
imaginary alpha precision.

The standard fix is to fold turnover directly into the objective as an L1
penalty on the weight change:

    penalty(w) = lambda * sum(|w_i - w_prev_i|)

L1 (not L2) is deliberate: it is the natural convex proxy for per-share
transaction costs, which are themselves roughly linear in traded notional (see
:class:`quantlab.backtest.costs.CostModel` — commission and slippage are both
linear in ``|shares|``/``|notional|``). An L2 penalty would over-penalize large,
high-conviction rebalances relative to many small ones; L1 charges every unit
of turnover the same marginal rate, and — unlike L2 — admits exact "don't
trade" corners in the constrained solution when the expected-return
improvement doesn't clear the penalty.

``lambda`` (``turnover_penalty`` on
:class:`~quantlab.portfolio.optimizer.MeanVarianceOptimizer`) trades off
signal responsiveness against trading cost: ``lambda == 0`` recovers the
unconstrained solve every period; larger ``lambda`` increasingly anchors the
solution to ``previous_weights``, monotonically shrinking realized turnover
(``tests/test_portfolio.py`` verifies this empirically).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["turnover", "turnover_penalty"]


def turnover(weights: pd.Series, previous_weights: pd.Series | None = None) -> float:
    """One-way turnover: ``sum(|w_i - w_prev_i|)`` over the union of names.

    ``previous_weights=None`` (or an empty Series) is treated as an all-cash
    starting book, so turnover is simply ``sum(|w_i|)`` — a full initial
    deployment.
    """
    if previous_weights is None or previous_weights.empty:
        return float(weights.abs().sum())
    idx = weights.index.union(previous_weights.index)
    w = weights.reindex(idx).fillna(0.0)
    prev = previous_weights.reindex(idx).fillna(0.0)
    return float((w - prev).abs().sum())


def turnover_penalty(
    weights: np.ndarray, previous_weights: np.ndarray, lam: float
) -> float:
    """The scalar L1 penalty ``lambda * sum(|w - w_prev|)`` on raw arrays.

    Operates on aligned numpy arrays (not :class:`pandas.Series`) so it can be
    called directly from an optimizer's objective function on every
    iteration without the overhead of index alignment.
    """
    if lam == 0.0:
        return 0.0
    return float(lam * np.sum(np.abs(weights - previous_weights)))

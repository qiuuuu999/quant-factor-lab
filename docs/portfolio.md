# Portfolio Construction

This document defines `quantlab.portfolio` — mean-variance optimization,
risk-parity weighting, and turnover control — and the conventions they
follow. It bridges the factor signals (`quantlab.factors`) and risk model
(`quantlab.risk`) already in this platform to executable target weights
consumable by `quantlab.backtest.engine.run_backtest`.

---

## Why not equal weight

`scripts/run_momentum_backtest.py` selects the top 20% of names by 12-1
momentum each month and equal-weights them. Equal weighting is a defensible
baseline, but it makes an implicit assumption: **every selected name carries
the same conviction (alpha magnitude) and the same marginal risk.** Neither is
generally true — two names in the same momentum decile can have very
different realized volatility and very different correlation to the rest of
the book. Sizing them identically ignores information the platform has
already computed (`quantlab.risk.covariance`) and estimated (the factor
score itself).

Mean-variance optimization uses both pieces of information: it sizes each
position by its expected-return-to-risk trade-off and diversifies away
correlated risk an equal-weight sleeve cannot see. The cost is added
sensitivity to estimation error in `alpha`/`covariance` — a noisy alpha or an
ill-conditioned covariance can produce concentrated, unstable weights. This is
why the optimizer is paired with `LedoitWolfCovariance` (well-conditioned even
when names outnumber observations — see `docs/risk.md`) and with turnover
control (below), rather than used against a raw sample covariance with no
brake on trading.

---

## Mean-variance optimizer

`quantlab.portfolio.optimizer.MeanVarianceOptimizer`

Solves the constrained Markowitz problem

$$
\max_w \quad \alpha' w - \frac{k}{2} w'\Sigma w - \lambda \lVert w - w_{\text{prev}} \rVert_1
\quad \text{s.t.} \quad \mathbf{1}'w = 1,\ \ w_i \in [\text{lo}, \text{hi}],\ \ w'\Sigma w \le \sigma_{\text{target}}^2
$$

via `scipy.optimize.minimize(method="SLSQP")` — a general nonlinear
constrained solver, needed because the target-volatility constraint and the
turnover penalty make this a nonlinear program (a plain QP solver would
suffice for the unconstrained-turnover, no-vol-cap case alone, but not once
those are added).

```python
class MeanVarianceOptimizer:
    def __init__(self, *, risk_aversion=1.0, long_only=True, max_weight=None,
                 target_volatility=None, turnover_penalty=0.0): ...
    def optimize(self, alpha: pd.Series, covariance: pd.DataFrame, *,
                 previous_weights: pd.Series | None = None) -> OptimizationResult: ...
```

- `alpha` — one score per name (the factor signal, e.g. z-scored 12-1
  momentum); it defines the universe optimized over.
- `covariance` — `ticker x ticker`, from `quantlab.risk.covariance`. Must
  fully cover `alpha`'s names.
- `risk_aversion` (`k`) — trades expected-return capture against risk
  reduction. `k -> 0` approaches a pure alpha-maximizing solve (subject to
  constraints); larger `k` pulls toward the minimum-variance corner.
- `long_only`, `max_weight` — box constraints. `long_only=True` bounds weights
  at `[0, max_weight]`; `False` allows shorts down to `-max_weight` (or
  unbounded if `max_weight is None`). The budget constraint `sum(w) == 1` is
  always enforced, so `max_weight * n_names >= 1` is required for feasibility
  (checked eagerly — see below).
- `target_volatility` — optional ceiling on `sqrt(w'Σw)`; adds the inequality
  constraint. Infeasible target/alpha combinations (asking for less risk than
  the minimum-variance portfolio can deliver) surface as a non-`success`
  `OptimizationResult`, not an exception — SLSQP returns its best-effort
  point along with `success=False`.
- `turnover_penalty` (`λ`) — see [Turnover control](#turnover-control) below.

**Feasibility checks.** `max_weight * n < 1` (with `long_only`) is caught
before calling the solver and raises `ValueError` with an explicit message —
SLSQP would otherwise silently return an infeasible point close to its
best-effort optimum rather than a clear error, which is worse than failing
fast at construction time.

### Closed-form verification

For the unconstrained case (only the budget constraint), Lagrangian
stationarity gives an exact solution:

$$
w^* = \frac{1}{k}\Sigma^{-1}(\alpha - \gamma \mathbf{1}), \qquad
\gamma = \frac{\mathbf{1}'\Sigma^{-1}\alpha - k}{\mathbf{1}'\Sigma^{-1}\mathbf{1}}
$$

`tests/test_portfolio.py::test_mean_variance_matches_two_asset_closed_form_solution`
computes this directly with `numpy.linalg` for a two-asset problem and asserts
the SLSQP solve matches to `1e-5`, across several risk-aversion values. This
is the strongest correctness check available for a numerical optimizer: an
exact algebraic answer, not just a qualitative property.

Constraint tests then confirm: no weight exceeds `max_weight`, weights always
sum to `1`, and `w'Σw` never exceeds `target_volatility ** 2` when supplied
(with a small numerical tolerance for the SLSQP convergence criterion).

---

## Turnover control

`quantlab.portfolio.turnover_control`

Every unit of turnover pays the transaction cost model
(`quantlab.backtest.costs.CostModel` — commission + slippage, both linear in
traded shares/notional). Re-solving a mean-variance problem from scratch every
rebalance chases noise in `alpha`, trading a name from 4.9% to 5.1% to capture
a marginal, likely-spurious improvement in the objective.

The fix folds an L1 penalty on the weight change directly into the objective:

$$
\text{penalty}(w) = \lambda \sum_i |w_i - w_{\text{prev},i}|
$$

**L1, not L2.** L1 is the natural convex proxy for the (roughly linear)
transaction cost structure itself — see the module docstring for the full
argument. Practically, it means every unit of turnover is charged the same
marginal rate regardless of trade size, and (unlike a quadratic penalty) it
admits exact "don't trade this name" corners in the solution when the
alpha improvement doesn't clear the cost, rather than merely shrinking every
trade a little.

`turnover(weights, previous_weights)` computes realized one-way turnover
(`sum(|Δw|)`, handling disjoint name sets and an all-cash start).
`turnover_penalty(w, w_prev, lam)` is the raw-array penalty term the
optimizer's objective calls on every SLSQP iteration.

**Monotonicity check.**
`tests/test_portfolio.py::test_turnover_penalty_monotonically_shrinks_realized_turnover`
solves the same problem across `λ ∈ {0, 0.01, 0.05, 0.2, 1.0}` and asserts
realized turnover is non-increasing in `λ` — the qualitative property that
matters for a penalty term (an exact closed form for the penalized problem
isn't available once box constraints bind, so this is checked empirically
rather than algebraically, unlike the base optimizer above).

---

## Risk parity (control group)

`quantlab.portfolio.risk_parity.RiskParityOptimizer`

The mean-variance sleeve is only as good as `alpha`. Risk parity is a
deliberately alpha-free control group: size each position so it contributes
**equally to total portfolio variance**, ignoring the signal entirely. Any
outperformance the mean-variance construction shows over risk parity is then
attributable to the alpha signal, not merely to smarter-than-equal-weight risk
balancing — which risk parity also does, so equal-weight alone is not a
sufficient control.

For weights `w` and covariance `Σ`, name `i`'s risk contribution to portfolio
variance is `RC_i = w_i (Σw)_i` — the same Euler decomposition used in
`quantlab.risk.factor_risk` (`sum_i RC_i = w'Σw` exactly, by homogeneity of
degree 2). Equal risk contribution (ERC) is found as:

$$
\min_w \sum_i \left(RC_i(w) - \overline{RC}(w)\right)^2 \quad \text{s.t.} \quad \mathbf{1}'w=1,\ 0 \le w_i \le \text{max\_weight}
$$

again via SLSQP. The objective's global minimum is exactly `0` at the true
ERC solution regardless of the overall variance scale, so no normalization is
needed for convergence.

`tests/test_portfolio.py` checks the two defining properties directly: (1)
for uncorrelated, equal-variance names, ERC recovers equal weights (the
degenerate case where risk parity coincides with equal weight); (2) for a
general covariance, each name's realized share of total portfolio variance is
equalized (`~1/n`), with lower weight assigned to higher-vol names as expected.

---

## Comparison experiment: three constructions of the 12-1 momentum sleeve

`scripts/run_portfolio_comparison.py` takes the exact top-20%-by-12-1-momentum
selection from `scripts/run_momentum_backtest.py` (point-in-time S&P 500,
monthly rebalance) and builds target weights three ways at every rebalance:

1. **Equal weight** — the existing baseline (`1/n` per selected name).
2. **Mean-variance** — `alpha` = that period's z-scored momentum score among
   the selected names; `covariance` = `LedoitWolfCovariance` fit on trailing
   24 months of returns for the same names; `long_only=True`,
   `max_weight=0.02` (2% single-name cap), `turnover_penalty` anchored to the
   prior period's realized weights.
3. **Risk parity** — same `LedoitWolfCovariance`, same selected names, no
   alpha.

All three run through the identical `run_backtest` execution assumptions
(decide on month-end close, fill at next-session open, defer halted opens,
default cost model) so any difference in the resulting tearsheet is
attributable purely to the weighting scheme, not to execution mechanics. See
`reports/portfolio_comparison/summary.md` for the resulting NAV comparison,
Sharpe, max drawdown, and annualized turnover across the three.

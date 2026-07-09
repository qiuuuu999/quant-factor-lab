# Risk Models

This document defines `quantlab.risk` — covariance estimation, style-factor
risk decomposition, and portfolio exposure attribution — and the conventions
they follow.

---

## Covariance estimators

`quantlab.risk.covariance`

Three estimators of an `asset x asset` (or `factor x factor`) covariance
matrix, behind one interface:

```python
class CovarianceEstimator(ABC):
    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame: ...
```

`returns` is a `period x asset` frame of periodic returns; rows with any `NaN`
are dropped before estimation. `estimate_covariance(returns, method=...,
**kwargs)` dispatches to any of the three by name.

### Sample covariance

`quantlab.risk.covariance.SampleCovariance`

The textbook estimator: `S = (1/(T - ddof)) * X'X` on demeaned returns
(`ddof=1` by default — pandas' `.cov()`).

**Why it breaks down.** With `N` assets and `T` observed periods, `S` has
`N(N+1)/2` free parameters estimated from `N*T` numbers, and it is
**singular** whenever `N > T`. This is the normal case for an equity risk
model: a few hundred names in the universe, a handful of years of monthly
history. A singular covariance matrix cannot be inverted, which is exactly
what a portfolio optimizer or risk-budgeting step needs to do — so a naive
sample covariance is unusable at realistic problem sizes, not just
theoretically suboptimal.

### EWMA covariance

`quantlab.risk.covariance.EWMACovariance`

An exponentially-weighted sample covariance (RiskMetrics-style): period `t`
gets weight `w_t ∝ decay^(T-t)` with `decay = 0.5 ** (1 / halflife)`, weights
normalized to sum to 1, and the covariance uses the weighted mean (not zero) so
it remains a proper covariance rather than a raw second moment:

$$
\hat\Sigma = \frac{1}{1 - \sum_t w_t^2} \sum_t w_t (x_t - \bar x_w)(x_t - \bar x_w)'
$$

The `1 - Σw²` denominator is the weighted analogue of dividing by `T - 1`.
Larger `halflife` → closer to the plain sample covariance; smaller `halflife`
→ more responsive to a recent volatility/correlation regime shift, at the cost
of more noise. Still singular whenever `N > T` (it reweights the same `T`
observations; it does not add information).

### Ledoit-Wolf shrinkage

`quantlab.risk.covariance.LedoitWolfCovariance`

**Motivation.** Neither reweighting scheme above fixes the ill-conditioning:
when the number of assets exceeds the number of sample periods, the sample
covariance matrix (weighted or not) is singular — it has zero or
near-zero eigenvalues and cannot be inverted. Ledoit & Wolf (2004) address
this directly by shrinking the sample covariance toward a **well-conditioned,
full-rank target**: a scaled identity matrix `F = μ·I`, where `μ` is the
average sample variance across assets. The shrunk estimator is a convex
combination:

$$
\hat\Sigma = \delta \cdot \mu I + (1 - \delta) \cdot S
$$

`I` is full rank by construction, so `Σ̂` is invertible for **any** `N` and `T`
— this is what makes it usable exactly where the plain sample covariance
fails. The shrinkage intensity `δ ∈ [0, 1]` is not a tuning parameter: it has
a closed-form estimate that minimizes expected quadratic loss
`E‖Σ̂ − Σ‖²_F`, derived from the data itself (Ledoit & Wolf, 2004, Appendix B):

$$
\delta = \operatorname{clip}\!\left(\frac{\hat\pi}{\hat\gamma \cdot T},\ 0,\ 1\right)
$$

where `π̂` estimates the total sampling variance of the entries of `S`
(`Σᵢⱼ Var(√T · Sᵢⱼ)`, i.e. how noisy the sample covariance is) and `γ̂ = ‖S −
F‖²_F` is the squared distance from the sample covariance to the target (i.e.
how much bias shrinking would introduce). Intuitively: shrink hard when the
sample estimate is noisy relative to how much it disagrees with the target
(small `T`, many assets), shrink little when the sample is already reliable
(`T` large relative to `N`).

`LedoitWolfCovariance().estimate(returns)` stores the fitted intensity on
`last_shrinkage_` for inspection after each call.

**When to reach for it.** Use it whenever the cross-section is large relative
to history — e.g. specific-return covariance across a few hundred stocks with
a few years of data. It is unnecessary (shrinkage → ~0) when `T ≫ N`, e.g. the
4-style-factor covariance in this platform's factor risk model
(`scripts/run_risk_attribution.py` uses the plain sample covariance there for
exactly this reason). `tests/test_risk.py` verifies both regimes: the shrunk
estimate is invertible when the sample covariance is numerically singular
(`N > T`), and the shrinkage intensity is small (and the estimate converges to
the true covariance) when `T ≫ N`.

### Reference

> Ledoit, O., & Wolf, M. (2004). *Honey, I Shrunk the Sample Covariance
> Matrix.* The Journal of Portfolio Management, 30(4), 110–119.
> https://doi.org/10.3905/jpm.2004.110

---

## Factor risk decomposition

`quantlab.risk.factor_risk`

A simplified, price-factor-only cousin of a commercial (Barra/Axioma)
fundamental risk model: the four price factors in `quantlab.factors` (12-1
momentum, low volatility, 1-month reversal, Amihud illiquidity) stand in for
dozens of style + industry factors, and there is no market-cap weighting of
the cross-sectional regression. The mechanics are the same.

### 1. Exposures

`build_exposure_panel(factors, long_prices, rebal_dates, members_by_date, ...)`
computes each factor's point-in-time value for every name at every rebalance
(reusing `quantlab.factors.evaluation.build_factor_panel`, so the no-look-ahead
guarantee carries over), then **winsorizes and z-scores each factor
cross-sectionally on each date** — the standard preparation described in
`docs/factors.md`. The result is a `{date: ticker x factor}` mapping of
exposures, in units of "standard deviations of the universe that day," plus
the shared `date x ticker` forward-return panel.

### 2. Factor and specific returns

`cross_sectional_regression(forward_returns, exposures)` runs one OLS per
period — that period's realized returns regressed on that period's exposures,
with an intercept absorbing the equal-weighted average return (exposures are
mean-zero by construction; without an intercept, that level effect would leak
into the factor coefficients):

$$
r_{i,t} = \alpha_t + \sum_k \beta_{k,t} \cdot e_{i,k,t} + \epsilon_{i,t}
$$

The fitted slopes `β_{k,t}` are that period's **factor returns**; the
residuals `ε_{i,t}` are that period's **specific (idiosyncratic) returns**.
This is the classic Fama-MacBeth cross-sectional regression, run once per
rebalance. `FactorRiskModel.fit(exposures_by_date, forward_returns)` runs it
across every date and stores the two resulting time series
(`factor_returns_`, `specific_returns_`).

### 3. Factor covariance and specific variance

- `FactorRiskModel.factor_covariance(estimator=None)` — any
  `CovarianceEstimator` (default `SampleCovariance`) applied to the
  factor-return time series. With only a handful of style factors and years of
  monthly history, `T ≫ K` and the plain sample covariance is already
  well-conditioned.
- `FactorRiskModel.specific_variance(min_periods=2)` — per-ticker sample
  variance of the residual time series; the standard factor-model assumption
  is that all *cross-sectional* correlation between names is captured by the
  shared factors, so specific returns are treated as uncorrelated across
  names (a diagonal specific-risk matrix).

### 4. Portfolio decomposition

`decompose_portfolio_risk(weights, exposures, factor_covariance,
specific_variance)` (or `FactorRiskModel.decompose(weights, exposures)`) splits
a portfolio's variance:

$$
\operatorname{Var}(portfolio) = \underbrace{e' \Sigma_f e}_{\text{factor variance}} + \underbrace{w' D\, w}_{\text{specific variance}}
$$

where `e = w' B` is the portfolio's factor exposure
(`quantlab.risk.attribution.portfolio_factor_exposure`, the weighted average
of each held name's exposures), `Σ_f` is the factor covariance, and `D` is the
diagonal matrix of per-name specific variances.

`FactorRiskDecomposition.factor_contributions` further splits the factor
variance term across individual factors via an **Euler (marginal
contribution) decomposition**:

$$
\text{contribution}_k = e_k \cdot (\Sigma_f e)_k, \qquad \sum_k \text{contribution}_k = e' \Sigma_f e \text{ exactly}
$$

This follows from Euler's homogeneous-function identity applied to the
quadratic form `e'Σe` (degree 2, so `e·∇(e'Σe) = 2·e'Σe`, and `∇(e'Σe) =
2Σe`) — it is the same identity risk systems use to attribute portfolio
volatility to individual positions.

**Correctness check.** `tests/test_risk.py` builds a synthetic single-factor
model (`r_it = b_i·f_t + ε_it` with known `b`, `σ_f`, `σ_i`), fits the model
above, and asserts the recovered `factor_covariance` and `specific_variance`
converge to the true `σ_f²` and `σ_i²`, and that
`decompose_portfolio_risk` reproduces the true, closed-form portfolio variance
split.

---

## Attribution: portfolio factor-exposure profile

`quantlab.risk.attribution`

Answers a narrower question than the full decomposition: **how many standard
deviations of each factor does this portfolio carry, net of its individual
names?**

```python
portfolio_factor_exposure(weights, exposures) -> pd.Series   # factor -> exposure
```

is the holdings-weighted average of each name's factor exposure,
`exposure_k = Σᵢ wᵢ · exposure_{i,k}`. Because `exposures` is cross-sectionally
z-scored, the result reads directly in "standard deviations of the universe
that day" — e.g. an equal-weight top-decile-momentum book reads roughly
`+1.5σ` on momentum. `ExposureProfile.build(weights, exposures, as_of)` bundles
this with the snapshot date and number of names held, with a `.summary()` for
quick printing.

This module has no dependency on how the exposures or risk model were built —
`factor_risk.py` is the only consumer of `portfolio_factor_exposure`, not the
other way around — so it stays a plain, dependency-free leaf reusable by any
future portfolio-construction or reporting code.

---

## Demo: the 12-1 momentum strategy, 2015-2025

`scripts/run_risk_attribution.py` runs the pipeline above on the exact
top-decile 12-1 momentum portfolio from `reports/momentum_12_1/` (point-in-time
S&P 500, equal-weight top 20%, monthly rebalance) and writes, under
`reports/risk_attribution/`:

- `exposure_profile.png` — the strategy's factor exposure at its most recent
  rebalance
- `risk_decomposition.png` — that snapshot's factor-vs-specific variance split
- `exposure_timeseries.csv` — the full 2015-2025 exposure history
- `summary.md` — both tables, plus the fitted factor covariance matrix

As expected for a momentum sort: large positive momentum exposure (`~+1.5σ`,
by construction of the selection rule), modest **negative** low-volatility and
illiquidity exposure (winning momentum names in a large-cap universe tend to
be higher-volatility, more liquid mega-caps — consistent with the negative
correlation between momentum and low-vol/illiquidity in
`reports/factor_eval/factor_correlation.png`), and near-zero reversal exposure
(momentum's one-month skip makes it close to orthogonal to reversal by
construction — see `docs/factors.md`). The bulk of the strategy's variance
(~85% in the most recent snapshot) is **factor-driven**, not diversifiable
name-specific risk — exactly what a concentrated, single-signal factor book
should look like.

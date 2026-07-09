"""regime_adaptive — regime-conditioned dynamic factor weighting.

`quantlab.monitor.regime` already answers "where does each factor work" in
retrospect (the factor x regime mean-IC matrix in
`reports/monitor/regime_heatmap.png`). This module turns that into a *live*
allocation rule: at every rebalance, combine the factor library into one
composite score using weights derived from **each factor's own
regime-conditioned track record up to that point in time**, then hold the top
names by composite score, equal-weighted.

Point-in-time discipline
-------------------------
Two distinct look-ahead traps have to be closed for this to be a legitimate
backtest, not a retrospective analysis wearing a backtest's clothes:

1. **The regime label itself.** `quantlab.monitor.regime.classify_regime`'s
   default mode splits volatility into low/high at the *full-sample* median —
   correct for a retrospective health report, wrong for a trading decision,
   since the median of the whole 2015-2025 series is not knowable in 2017. This
   module always calls it with `expanding=True` (see that function's
   docstring), so day `t`'s label depends only on prices up to and including
   `t`.
2. **The factor-regime fit table.** `quantlab.monitor.regime.factor_regime_matrix`
   averages IC over the *entire* sample — again correct for "how did this
   factor do, on reflection" and wrong for "what should I believe about this
   factor today." `regime_conditioned_factor_weights` below instead computes,
   for a rebalance decision at `as_of`, the mean IC of each factor **using
   only IC observations dated strictly before `as_of`, restricted to periods
   that were themselves in the current regime**. A factor's IC observation at
   formation date `d` requires the forward return from `d` to the next
   formation date to have already realized, which happens exactly at that
   next formation date's close — so `index < as_of` (not `<=`) is the correct
   cutoff: by the time the decision at `as_of` is made, everything dated
   before `as_of` is known, and `as_of`'s own IC is not yet computable.

Weighting rule
--------------
For each factor, `regime_conditioned_factor_weights` computes the
regime-conditioned, strictly-prior mean IC. A factor with too little
regime-matched history (`min_obs`) is dropped for that rebalance — it is not
that its edge is assumed zero, but that there isn't yet enough evidence to
condition on. Surviving factors get a signed weight equal to their mean IC,
so the combination weight is proportional to *how well the factor has
predicted returns in this regime* and its sign encodes *which direction* —
a factor with a negative regime-conditioned IC either has its exposure
inverted (`negative_handling="invert"`, the default: even a factor "working
backwards" in a regime carries information) or is excluded entirely
(`negative_handling="zero"`, a more conservative choice). Weights are then
normalized to sum to 1 in absolute value. If every factor is dropped (e.g.
during an early warm-up period, or a rare regime with no accumulated
history), the rule falls back to an equal, uninverted combination rather than
producing an empty score.

`build_static_multifactor_weights` is the fixed control group: an
un-conditioned, always-equal combination of the same four factors, with no
regime awareness and no historical-IC weighting at all — isolating how much
of any performance difference is attributable to the regime-conditioning
specifically, as opposed to simply combining several factors together.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from quantlab.factors.preprocess import deciles
from quantlab.monitor.regime import classify_regime, regime_as_of

__all__ = [
    "pit_regime_by_date",
    "regime_conditioned_factor_weights",
    "composite_score",
    "build_regime_adaptive_weights",
    "build_static_multifactor_weights",
]


def pit_regime_by_date(
    benchmark_prices: pd.Series,
    dates: Sequence[pd.Timestamp],
    *,
    vol_window: int = 21,
    trend_window: int = 200,
) -> pd.Series:
    """Point-in-time regime label as of each date, forward-filled from the daily classification.

    Thin wrapper around :func:`quantlab.monitor.regime.classify_regime` with
    ``expanding=True`` (see that function's docstring for why this matters)
    composed with :func:`quantlab.monitor.regime.regime_as_of`.
    """
    daily = classify_regime(
        benchmark_prices, vol_window=vol_window, trend_window=trend_window, expanding=True,
    )
    return regime_as_of(daily, dates)


def regime_conditioned_factor_weights(
    ic_by_factor: Mapping[str, pd.Series],
    regime_by_date: pd.Series,
    as_of: pd.Timestamp,
    current_regime: str,
    *,
    negative_handling: str = "invert",
    min_obs: int = 6,
) -> pd.Series:
    """Signed combination weight per factor for one rebalance decision.

    Weight magnitude is proportional to ``|mean IC|`` of the factor,
    conditioned on the current regime and estimated only from IC observations
    dated strictly before ``as_of`` (see the module docstring for why ``<``,
    not ``<=``). The sign is the sign of that mean IC when
    ``negative_handling="invert"`` (a negative-IC factor is used inverted, not
    dropped); with ``negative_handling="zero"`` a negative-IC factor gets
    weight ``0`` instead. A factor with fewer than ``min_obs`` regime-matched
    prior observations is excluded from this rebalance (insufficient
    evidence, not "no edge"). Weights are normalized so
    ``weights.abs().sum() == 1``; if every factor is excluded, falls back to
    an equal (uninverted) weight on every factor rather than an all-zero
    (empty) result.
    """
    if negative_handling not in ("invert", "zero"):
        raise ValueError(f"negative_handling must be 'invert' or 'zero', got {negative_handling!r}")

    signed: dict[str, float] = {}
    for name, ic in ic_by_factor.items():
        prior = ic[ic.index < as_of].dropna()
        prior_regime = regime_by_date.reindex(prior.index)
        matched = prior[prior_regime == current_regime]
        if len(matched) < min_obs:
            signed[name] = 0.0
            continue
        mean_ic = float(matched.mean())
        if mean_ic >= 0.0 or negative_handling == "invert":
            signed[name] = mean_ic
        else:  # negative_handling == "zero" and mean_ic < 0
            signed[name] = 0.0

    weights = pd.Series(signed, dtype=float)
    l1 = float(weights.abs().sum())
    if l1 <= 0.0:
        n = len(weights)
        return pd.Series(1.0 / n, index=weights.index) if n else weights
    return weights / l1


def composite_score(exposures: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Combine one rebalance's ``ticker x factor`` exposures into one score per ticker.

    ``score_i = sum_k weights_k * exposures_{i,k}``. Factors in ``exposures``
    absent from ``weights`` (or vice versa) contribute ``0``.
    """
    w = weights.reindex(exposures.columns).fillna(0.0)
    return exposures.mul(w, axis=1).sum(axis=1)


def _select_top_bucket(score: pd.Series, n_buckets: int, top_buckets: tuple[int, ...]) -> list[str]:
    buckets = deciles(score, n_buckets)
    return buckets[buckets.isin(top_buckets)].index.tolist()


def build_regime_adaptive_weights(
    exposures_by_date: Mapping[pd.Timestamp, pd.DataFrame],
    ic_by_factor: Mapping[str, pd.Series],
    regime_by_date: pd.Series,
    rebal_dates: Sequence[pd.Timestamp],
    *,
    negative_handling: str = "invert",
    min_obs: int = 6,
    n_buckets: int = 10,
    top_buckets: tuple[int, ...] = (9, 10),
) -> tuple[dict[pd.Timestamp, dict[str, float]], pd.DataFrame]:
    """Target weights per rebalance: composite-score top bucket, equal-weighted.

    The composite score at each date uses that date's
    :func:`regime_conditioned_factor_weights` (current regime from
    ``regime_by_date``, strictly-prior IC history from ``ic_by_factor``).
    Dates with no known regime (e.g. before the benchmark's trend/vol windows
    warm up) are skipped.

    Returns
    -------
    ``(weights_by_date, factor_weight_diagnostics)`` — the target weights, and
    a ``date x factor`` frame of the signed combination weight actually used
    each rebalance (for inspecting how the mix shifts with the regime).
    """
    weights_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    diag_rows: dict[pd.Timestamp, pd.Series] = {}

    for dt in rebal_dates:
        if dt not in exposures_by_date:
            continue
        current_regime = regime_by_date.get(dt)
        if current_regime is None or (isinstance(current_regime, float) and np.isnan(current_regime)):
            continue

        fw = regime_conditioned_factor_weights(
            ic_by_factor, regime_by_date, dt, current_regime,
            negative_handling=negative_handling, min_obs=min_obs,
        )
        diag_rows[dt] = fw

        score = composite_score(exposures_by_date[dt], fw)
        selected = _select_top_bucket(score, n_buckets, top_buckets)
        if not selected:
            continue
        w = 1.0 / len(selected)
        weights_by_date[dt] = {t: w for t in selected}

    diagnostics = pd.DataFrame(diag_rows).T.sort_index() if diag_rows else pd.DataFrame()
    return weights_by_date, diagnostics


def build_static_multifactor_weights(
    exposures_by_date: Mapping[pd.Timestamp, pd.DataFrame],
    rebal_dates: Sequence[pd.Timestamp],
    *,
    n_buckets: int = 10,
    top_buckets: tuple[int, ...] = (9, 10),
) -> dict[pd.Timestamp, dict[str, float]]:
    """Target weights per rebalance: fixed equal-weight factor combination (control group).

    No regime conditioning, no IC history — every factor contributes equally
    (``1/n_factors``, unsigned) to the composite score every period. Isolates
    the effect of regime-conditioning by holding the "combine several factors"
    part of the strategy fixed.
    """
    weights_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    for dt in rebal_dates:
        if dt not in exposures_by_date:
            continue
        exp = exposures_by_date[dt]
        score = exp.mean(axis=1)
        selected = _select_top_bucket(score, n_buckets, top_buckets)
        if not selected:
            continue
        w = 1.0 / len(selected)
        weights_by_date[dt] = {t: w for t in selected}
    return weights_by_date

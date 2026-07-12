"""decay — factor IC decay detection via rolling IC and CUSUM change-point testing.

A factor's edge is not a fixed property: it is discovered, traded, and — as
capital crowds in, the underlying inefficiency is arbitraged away, or the
market's structure shifts — it decays. Waiting for a full-sample IC to turn
negative catches this far too late. This module answers a narrower, more
useful question: **has the factor's IC undergone a structural break, and if
so, when?**

Pipeline
--------
1. :func:`rolling_ic` smooths the raw per-period IC series
   (:func:`quantlab.factors.evaluation.information_coefficient`) with a
   trailing window (36 periods / 3 years of monthly rebalances by default) —
   the "current form" figure a health report leads with.
2. :func:`cusum_test` runs a retrospective (single, unknown-time) change-point
   test on the *raw* IC series: it looks for the one break in the mean that
   best explains the data, tests whether that break is statistically
   significant, and reports which side of the break is worse.
3. :func:`factor_health_report` packages both into a
   :class:`FactorHealthReport` — current rolling IC vs. historical mean,
   whether a decay alert is triggered, and the alert date.

CUSUM change-point test
------------------------
For a series ``x_1, ..., x_n`` with sample mean ``x̄`` and sample standard
deviation ``s``, the cumulative sum of mean-centred observations

    S_k = sum_{i=1}^{k} (x_i - x̄),  k = 1..n

drifts away from zero around the true (unknown) break point and returns
toward zero elsewhere, so the point that maximises ``|S_k|`` is the
maximum-likelihood estimate of a single change point (Page, 1954). Normalised
by ``s * sqrt(n)``, the statistic ``max_k |S_k| / (s * sqrt(n))`` converges
(under the no-break null) to the supremum of a Brownian bridge, whose
quantiles give parameter-free critical values (same table used for the
Kolmogorov-Smirnov two-sample test): 1.22 / 1.36 / 1.63 at the 90% / 95% / 99%
confidence level. No distributional assumption on the IC series itself is
needed beyond finite variance.

This is a retrospective ("has a break already happened, and where") test, not
a sequential/online monitor — it is run each time a health report is
refreshed, over the full history collected so far.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "rolling_ic",
    "CusumResult",
    "cusum_test",
    "FactorHealthReport",
    "factor_health_report",
]

#: Brownian-bridge supremum critical values for the CUSUM change-point test,
#: keyed by confidence level. Same asymptotic table as the two-sample KS test.
_CUSUM_CRITICAL = {0.90: 1.22, 0.95: 1.36, 0.99: 1.63}

_DEFAULT_WINDOW = 36  # months


def rolling_ic(ic: pd.Series, *, window: int = _DEFAULT_WINDOW, min_periods: int | None = None) -> pd.Series:
    """Trailing rolling mean of a per-period IC series.

    ``min_periods`` defaults to a third of ``window`` so the series is
    populated well before the first full window, at the cost of a noisier
    early estimate.
    """
    clean = ic.dropna().sort_index()
    min_periods = min_periods if min_periods is not None else max(6, window // 3)
    return clean.rolling(window, min_periods=min_periods).mean()


@dataclass
class CusumResult:
    """Result of a retrospective CUSUM change-point test on a series.

    ``path`` is the normalised cumulative-sum path (indexed like the input);
    ``change_point`` is the index value at which ``|path|`` peaks, i.e. the
    estimated break date. ``mean_before`` / ``mean_after`` split the series at
    that point, so their comparison shows *which direction* the break moved.
    """

    path: pd.Series
    change_point: pd.Timestamp | None
    statistic: float
    critical_value: float
    confidence: float
    triggered: bool
    mean_before: float
    mean_after: float

    @property
    def is_decay(self) -> bool:
        """A statistically significant break whose *later* mean is lower."""
        return bool(
            self.triggered
            and not np.isnan(self.mean_before)
            and not np.isnan(self.mean_after)
            and self.mean_after < self.mean_before
        )

    def summary(self) -> str:
        if self.change_point is None:
            return "CUSUM: insufficient data"
        flag = "ALERT (decay)" if self.is_decay else ("break, not decay" if self.triggered else "no break")
        return (
            f"CUSUM {flag}: stat {self.statistic:.2f} vs critical "
            f"{self.critical_value:.2f} @ {self.confidence:.0%}, "
            f"break at {pd.Timestamp(self.change_point).date()} "
            f"(mean {self.mean_before:+.4f} -> {self.mean_after:+.4f})"
        )


def cusum_test(x: pd.Series, *, confidence: float = 0.95) -> CusumResult:
    """Retrospective single change-point test on the mean of ``x``.

    See the module docstring for the statistic and its critical values.
    Fewer than two clean observations returns an all-NaN, non-triggered
    result rather than raising.
    """
    if confidence not in _CUSUM_CRITICAL:
        raise ValueError(f"confidence must be one of {sorted(_CUSUM_CRITICAL)}")
    critical = _CUSUM_CRITICAL[confidence]

    clean = x.dropna().sort_index()
    n = len(clean)
    if n < 2:
        return CusumResult(pd.Series(dtype=float), None, np.nan, critical, confidence, False, np.nan, np.nan)

    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    cumsum = (clean - mean).cumsum()

    if not std or np.isnan(std):
        return CusumResult(cumsum * np.nan, None, np.nan, critical, confidence, False, np.nan, np.nan)

    path = cumsum / (std * np.sqrt(n))
    k_star = int(path.abs().to_numpy().argmax())
    change_point = clean.index[k_star]
    statistic = float(path.abs().iloc[k_star])
    triggered = statistic > critical

    before, after = clean.iloc[: k_star + 1], clean.iloc[k_star + 1 :]
    mean_before = float(before.mean()) if len(before) else np.nan
    mean_after = float(after.mean()) if len(after) else np.nan

    return CusumResult(
        path, change_point, statistic, critical, confidence, triggered, mean_before, mean_after,
    )


@dataclass
class FactorHealthReport:
    """A factor's current-form snapshot: rolling IC, CUSUM break test, alert flag."""

    name: str
    ic: pd.Series
    rolling: pd.Series
    cusum: CusumResult
    current_rolling_ic: float
    historical_mean_ic: float
    decay_alert: bool
    alert_date: pd.Timestamp | None

    def summary(self) -> str:
        flag = "DECAY ALERT" if self.decay_alert else "OK"
        alert = f", alert @ {self.alert_date.date()}" if self.alert_date is not None else ""
        return (
            f"{self.name:>16}  [{flag}]  current {self.current_rolling_ic:+.4f} "
            f"vs historical {self.historical_mean_ic:+.4f}{alert}"
        )


def factor_health_report(
    name: str, ic: pd.Series, *, window: int = _DEFAULT_WINDOW, confidence: float = 0.95
) -> FactorHealthReport:
    """Build a :class:`FactorHealthReport` from a factor's per-period IC series."""
    roll = rolling_ic(ic, window=window)
    cusum = cusum_test(ic, confidence=confidence)
    clean = ic.dropna()

    roll_clean = roll.dropna()
    current = float(roll_clean.iloc[-1]) if len(roll_clean) else np.nan
    historical = float(clean.mean()) if len(clean) else np.nan

    return FactorHealthReport(
        name=name,
        ic=ic,
        rolling=roll,
        cusum=cusum,
        current_rolling_ic=current,
        historical_mean_ic=historical,
        decay_alert=cusum.is_decay,
        alert_date=pd.Timestamp(cusum.change_point) if cusum.is_decay else None,
    )

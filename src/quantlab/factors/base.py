"""base — the Factor abstract base class and point-in-time guardrails.

Every factor implements a single method, :meth:`Factor._compute`, and inherits a
public :meth:`Factor.compute` template that enforces the platform's most
important invariant: **no look-ahead bias**. A factor value for ``as_of_date``
may only be computed from data available on or before ``as_of_date``.

The guard works in two layers:

1. *Interception* — if the caller passes a price frame that contains any row
   dated after ``as_of_date``, :meth:`compute` raises :class:`LookaheadBiasError`
   rather than silently trusting the subclass to ignore it.
2. *Defense in depth* — the frame handed to :meth:`_compute` is additionally
   sliced to ``date <= as_of_date``, so a subclass physically cannot see the
   future even if interception is disabled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime

import pandas as pd

__all__ = ["Factor", "LookaheadBiasError", "DATE_COL", "PRICE_COL"]

#: Expected column names in the long price frame produced by
#: :func:`quantlab.data.prices.get_prices`.
DATE_COL = "date"
PRICE_COL = "adj_close"


class LookaheadBiasError(Exception):
    """Raised when a factor is given data dated after its ``as_of_date``."""


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


class Factor(ABC):
    """Abstract base class for a cross-sectional factor.

    Subclasses set :attr:`name` and implement :meth:`_compute`. Callers use the
    public :meth:`compute`, which applies the point-in-time guard, delegates to
    :meth:`_compute`, and returns a value for *every* universe member (``NaN``
    where the factor is undefined).
    """

    #: Human-readable factor name; used as the returned Series name.
    name: str = "factor"

    @abstractmethod
    def _compute(
        self, prices: pd.DataFrame, universe: list[str], as_of_date: date
    ) -> pd.Series:
        """Compute raw factor values. ``prices`` is already PIT-sliced.

        Returns a Series indexed by ticker (subset of ``universe`` is fine; the
        base class reindexes to the full universe).
        """

    def compute(
        self,
        prices: pd.DataFrame,
        universe: list[str],
        as_of_date: str | date | datetime,
        *,
        enforce_pit: bool = True,
    ) -> pd.Series:
        """Compute the factor for ``universe`` as of ``as_of_date``.

        Parameters
        ----------
        prices:
            Long price frame with at least ``date`` and ``adj_close`` columns
            (as returned by :func:`quantlab.data.prices.get_prices`).
        universe:
            Tickers to score (typically the point-in-time index membership).
        as_of_date:
            The formation date. No data after this date may be used.
        enforce_pit:
            If ``True`` (default), raise :class:`LookaheadBiasError` when
            ``prices`` contains rows dated after ``as_of_date``.
        """
        as_of = _as_date(as_of_date)
        self._validate_frame(prices)
        universe = list(universe)

        if not prices.empty:
            max_date = pd.to_datetime(prices[DATE_COL]).max().date()
            if enforce_pit and max_date > as_of:
                raise LookaheadBiasError(
                    f"{type(self).__name__}.compute received data dated "
                    f"{max_date} > as_of_date {as_of}; this would leak future "
                    f"information. Slice prices to <= as_of_date first."
                )

        # Defense in depth: physically remove any future rows.
        sliced = prices[pd.to_datetime(prices[DATE_COL]).dt.date <= as_of]

        raw = self._compute(sliced, universe, as_of)
        if not isinstance(raw, pd.Series):
            raise TypeError(
                f"{type(self).__name__}._compute must return a pandas Series"
            )
        result = raw.reindex(universe)
        result.name = self.name
        return result

    @staticmethod
    def _validate_frame(prices: pd.DataFrame) -> None:
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame")
        missing = {DATE_COL, PRICE_COL} - set(prices.columns)
        if missing:
            raise ValueError(
                f"prices frame is missing required column(s): {sorted(missing)}"
            )

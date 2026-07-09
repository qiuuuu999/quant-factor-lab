"""costs — the transaction-cost model used by the backtest engine.

Every simulated fill pays two frictions, both charged as a cash outflow at
execution time (they *never* move the fill price itself, which keeps the
accounting closed-form and testable):

* **Commission** — a fixed fee per share traded (``commission_per_share``).
  This is the broker/exchange-style ticket cost and is symmetric for buys and
  sells: ``|shares| * commission_per_share``.
* **Slippage** — a fixed number of basis points of the traded *notional*
  (``slippage_bps``). This is a stylised market-impact / spread cost:
  ``|shares * price| * slippage_bps / 10_000``.

The total cost of a fill is the sum of the two and is always non-negative,
regardless of trade direction. The engine subtracts it from cash on top of the
share cash-flow, so a round-trip pays the cost twice (once per leg).

Defaults live in ``configs/backtest.yaml`` and are read via
:func:`load_cost_model`; construct :class:`CostModel` directly for tests or
sensitivity sweeps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from quantlab.data.universe import _repo_root

__all__ = ["CostModel", "default_config_path", "load_cost_model"]

log = logging.getLogger("quantlab.backtest.costs")

#: Basis points per unit fraction (1 bp = 0.01% = 1e-4).
_BPS = 1e-4


def default_config_path() -> Path:
    """Location of the default backtest config: ``<repo>/configs/backtest.yaml``."""
    return _repo_root() / "configs" / "backtest.yaml"


class CostModel(BaseModel):
    """Per-share commission plus basis-point slippage on traded notional.

    Parameters
    ----------
    commission_per_share:
        Currency charged per share traded (each leg). Default ``0``.
    slippage_bps:
        Slippage in basis points of the absolute traded notional. Default ``0``.
    """

    commission_per_share: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)

    def commission(self, shares: float) -> float:
        """Commission for trading ``shares`` (sign ignored)."""
        return abs(shares) * self.commission_per_share

    def slippage(self, shares: float, price: float) -> float:
        """Slippage cost for ``shares`` filled at ``price`` (sign ignored)."""
        return abs(shares * price) * self.slippage_bps * _BPS

    def cost(self, shares: float, price: float) -> float:
        """Total cost (commission + slippage) of a single fill."""
        return self.commission(shares) + self.slippage(shares, price)

    @classmethod
    def free(cls) -> "CostModel":
        """A zero-cost model (frictionless fills) — handy for tests/benchmarks."""
        return cls(commission_per_share=0.0, slippage_bps=0.0)


def load_cost_model(config_path: Path | None = None) -> CostModel:
    """Build a :class:`CostModel` from the ``costs:`` block of a YAML config.

    Falls back to an all-zero model (with a warning) when the file or block is
    absent, so a missing config degrades to frictionless rather than crashing.
    """
    config_path = Path(config_path or default_config_path())
    if not config_path.exists():
        log.warning("backtest config not found at %s; using zero-cost model",
                    config_path)
        return CostModel()
    doc = yaml.safe_load(config_path.read_text()) or {}
    costs = doc.get("costs", {}) or {}
    return CostModel(**costs)

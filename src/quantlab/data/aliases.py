"""aliases — ticker rename / alias resolution for the equity universe.

Companies routinely change ticker symbols while staying listed (``FB -> META``,
``ANTM -> ELV``, ``CTL -> LUMN``, ...). This breaks price pipelines in two ways:

* **Download**: Yahoo only serves history under the *current* symbol, so a
  point-in-time universe that (correctly) asks for the *historical* symbol
  ``FB`` gets nothing.
* **Query**: research code may refer to either the old or new symbol and expects
  the same underlying series.

:class:`AliasResolver` maps any historical symbol to its current symbol
(following multi-hop chains, e.g. ``VIAB -> VIAC -> PARA``) and back. Rename
records come from three sources, merged with explicit config winning:

1. auto-inferred from the Wikipedia change log (a company added under a new
   ticker the same day it is removed under the old one),
2. the built-in :data:`quantlab.data.universe.TICKER_RENAMES`,
3. a hand-maintained YAML file (``configs/ticker_renames.yaml``) for the cases
   inference cannot see (pure symbol changes leave no add/remove row).

The YAML also lists genuinely *delisted* tickers (acquisitions, take-privates)
so the price-quality report can label them "expected missing" rather than
treating them as a fixable gap.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from quantlab.data.universe import (
    TICKER_RENAMES,
    UniverseData,
    _repo_root,
    normalize_ticker,
)

__all__ = [
    "RenameRecord",
    "DelistRecord",
    "AliasResolver",
    "load_alias_resolver",
    "infer_renames_from_changes",
    "default_config_path",
]

log = logging.getLogger("quantlab.data.aliases")


def default_config_path() -> Path:
    return _repo_root() / "configs" / "ticker_renames.yaml"


def _sym(ticker: str) -> str:
    """Canonical comparison form for a symbol."""
    return normalize_ticker(str(ticker).strip().upper())


def _tokens(name: str | None) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t}


def _same_company(a: str | None, b: str | None) -> bool:
    """Heuristic: do two 'security' names refer to the same company?

    True when the smaller name's tokens are a subset of the larger's (so
    "Everest Group" ~ "Everest Re Group", "DuPont" ~ "DuPont") and at least one
    shared token is a real word (>= 4 chars), which rejects unrelated swaps like
    "New Corp" vs "Old Industries".
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if not small.issubset(big):
        return False
    return any(len(t) >= 4 for t in small)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

class RenameRecord(BaseModel):
    # `effective` is exposed as `date` in YAML/kwargs via its alias. (The field
    # cannot be *named* `date` because that shadows the `date` type in the class
    # namespace under `from __future__ import annotations`.)
    model_config = ConfigDict(populate_by_name=True)

    old: str
    new: str
    effective: date | None = Field(default=None, alias="date")
    note: str | None = None
    source: str = "config"


class DelistRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    ticker: str
    effective: date | None = Field(default=None, alias="date")
    note: str | None = None
    source: str = "config"


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #

class AliasResolver:
    """Bidirectional historical<->current ticker mapping."""

    def __init__(
        self,
        renames: list[RenameRecord] | None = None,
        delisted: list[DelistRecord] | None = None,
    ):
        self.renames = list(renames or [])
        self.delisted_records = list(delisted or [])

        # old -> new (later records override earlier ones).
        self._direct: dict[str, str] = {}
        for r in self.renames:
            old, new = _sym(r.old), _sym(r.new)
            if old and new and old != new:
                self._direct[old] = new
        self._delisted: set[str] = {_sym(d.ticker) for d in self.delisted_records}

    # -- forward ------------------------------------------------------------ #

    def to_current(self, ticker: str) -> str:
        """Resolve a (possibly historical) symbol to its current symbol.

        Follows rename chains transitively; unknown symbols pass through
        unchanged.
        """
        cur = _sym(ticker)
        seen: set[str] = set()
        while cur in self._direct and cur not in seen:
            seen.add(cur)
            cur = self._direct[cur]
        return cur

    def is_rename(self, ticker: str) -> bool:
        return _sym(ticker) in self._direct

    # -- reverse ------------------------------------------------------------ #

    def aliases_of(self, current: str) -> list[str]:
        """All historical symbols that resolve to ``current`` (excluding it)."""
        cur = _sym(current)
        return sorted(o for o in self._direct if self.to_current(o) == cur and o != cur)

    # -- delisting ---------------------------------------------------------- #

    def is_delisted(self, ticker: str) -> bool:
        return _sym(ticker) in self._delisted

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"AliasResolver(renames={len(self._direct)}, "
            f"delisted={len(self._delisted)})"
        )


# --------------------------------------------------------------------------- #
# Inference + loading
# --------------------------------------------------------------------------- #

def infer_renames_from_changes(data: UniverseData) -> list[RenameRecord]:
    """Infer renames from change-log rows that add and remove the same company.

    A pure ticker change sometimes shows up as a same-day add (new ticker) and
    remove (old ticker) of the same security name. Those are safe to infer.
    """
    out: list[RenameRecord] = []
    for c in data.changes:
        if not (c.added_ticker and c.removed_ticker):
            continue
        if _sym(c.added_ticker) == _sym(c.removed_ticker):
            continue
        if _same_company(c.added_security, c.removed_security):
            out.append(
                RenameRecord(
                    old=c.removed_ticker,
                    new=c.added_ticker,
                    date=c.date,
                    note=f"{c.removed_security} -> {c.added_security}",
                    source="inferred",
                )
            )
    return out


def _renames_from_builtin() -> list[RenameRecord]:
    out: list[RenameRecord] = []
    for current, history in TICKER_RENAMES.items():
        for old, eff in history:
            out.append(
                RenameRecord(old=old, new=current, date=eff, source="builtin")
            )
    return out


def _load_yaml(config_path: Path) -> tuple[list[RenameRecord], list[DelistRecord]]:
    if not config_path.exists():
        log.warning("alias config not found at %s; using built-ins only", config_path)
        return [], []
    doc = yaml.safe_load(config_path.read_text()) or {}
    renames = [RenameRecord(**{**r, "source": "config"}) for r in doc.get("renames", [])]
    delisted = [DelistRecord(**{**d, "source": "config"}) for d in doc.get("delisted", [])]
    return renames, delisted


# Cache the default resolver so repeated get_prices calls do not re-read YAML.
_DEFAULT_CACHE: dict[str, AliasResolver] = {}


def load_alias_resolver(
    config_path: Path | None = None,
    *,
    universe_data: UniverseData | None = None,
    use_cache: bool = True,
) -> AliasResolver:
    """Build an :class:`AliasResolver` from inference + built-ins + YAML config.

    ``universe_data`` enables change-log inference; omit it (the default) to
    build a network-free resolver from the built-in map and the YAML file only.
    Precedence when the same old symbol appears twice: config > built-in >
    inferred.
    """
    config_path = Path(config_path or default_config_path())
    key = str(config_path)

    if universe_data is None and use_cache and key in _DEFAULT_CACHE:
        return _DEFAULT_CACHE[key]

    yaml_renames, yaml_delisted = _load_yaml(config_path)
    inferred = infer_renames_from_changes(universe_data) if universe_data else []

    # Order = ascending precedence (later overrides earlier in the resolver).
    renames = inferred + _renames_from_builtin() + yaml_renames
    resolver = AliasResolver(renames=renames, delisted=yaml_delisted)

    if universe_data is None and use_cache:
        _DEFAULT_CACHE[key] = resolver
    return resolver

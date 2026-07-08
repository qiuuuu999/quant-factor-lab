"""data — data pipeline and point-in-time (PIT) storage.

Responsible for ingesting raw market and fundamental data, normalizing it, and
persisting it in a point-in-time store so that every backtest sees only the
information that was actually available on each historical date. This layer is
the foundation for eliminating survivorship and look-ahead bias across the
entire platform.
"""

from quantlab.data.universe import (
    Constituent,
    MembershipChange,
    UniverseData,
    get_universe,
    load_universe,
    refresh_universe,
)
from quantlab.data.prices import (
    QualityReport,
    TickerQuality,
    available_tickers,
    download_prices,
    get_prices,
    universe_symbols,
)
from quantlab.data.aliases import (
    AliasResolver,
    infer_renames_from_changes,
    load_alias_resolver,
)

__all__ = [
    "Constituent",
    "MembershipChange",
    "UniverseData",
    "get_universe",
    "load_universe",
    "refresh_universe",
    "QualityReport",
    "TickerQuality",
    "available_tickers",
    "download_prices",
    "get_prices",
    "universe_symbols",
    "AliasResolver",
    "load_alias_resolver",
    "infer_renames_from_changes",
]

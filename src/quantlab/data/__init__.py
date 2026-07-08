"""data — data pipeline and point-in-time (PIT) storage.

Responsible for ingesting raw market and fundamental data, normalizing it, and
persisting it in a point-in-time store so that every backtest sees only the
information that was actually available on each historical date. This layer is
the foundation for eliminating survivorship and look-ahead bias across the
entire platform.
"""

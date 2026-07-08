"""monitor — factor decay and regime detection.

Tracks the live health of deployed factors: information-coefficient decay,
turnover, crowding, and performance drift, and detects shifts in market regime
that may invalidate a factor's edge. Emits signals that trigger re-research,
re-weighting, or retirement of factors.
"""

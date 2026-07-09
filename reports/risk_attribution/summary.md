# Risk Attribution — 12-1 Momentum Strategy, 2015-2025

- **Portfolio**: point-in-time S&P 500, equal-weight top 20% by 12-1 momentum, monthly rebalance (identical selection to `reports/momentum_12_1/`)
- **Risk model**: 4-factor cross-sectional regression (momentum, low-vol, reversal, liquidity exposures; sample covariance of the fitted factor returns)
- **Snapshot date**: 2025-12-31 (100 names held)

## Factor exposure profile (snapshot)

| Factor | Exposure (σ) |
| --- | --- |
| momentum_12_1 | +1.50 |
| low_vol_252 | -0.51 |
| reversal_1m | +0.01 |
| amihud_illiq_21 | -0.35 |

## Factor exposure profile (average over 2015-2025 rebalances)

| Factor | Avg. exposure (σ) |
| --- | --- |
| momentum_12_1 | +1.45 |
| low_vol_252 | -0.09 |
| reversal_1m | -0.03 |
| amihud_illiq_21 | -0.23 |

## Risk decomposition (snapshot)

- Total variance: 0.000423
- Factor variance: 0.000360 (85.1%)
- Specific variance: 0.000063 (14.9%)

| Factor | Contribution | Share of factor variance |
| --- | --- | --- |
| momentum_12_1 | +0.000308 | 85.6% |
| low_vol_252 | +0.000033 | 9.2% |
| reversal_1m | -0.000000 | -0.1% |
| amihud_illiq_21 | +0.000019 | 5.3% |

## Factor covariance matrix (monthly)

| | momentum_12_1 | low_vol_252 | reversal_1m | amihud_illiq_21 |
| --- | --- | --- | --- | --- |
| momentum_12_1 | 0.00016 | 0.00007 | -0.00004 | -0.00002 |
| low_vol_252 | 0.00007 | 0.00034 | -0.00005 | 0.00000 |
| reversal_1m | -0.00004 | -0.00005 | 0.00010 | 0.00001 |
| amihud_illiq_21 | -0.00002 | 0.00000 | 0.00001 | 0.00005 |

## Figures

- `exposure_profile.png` — snapshot factor exposure (std. dev.)
- `risk_decomposition.png` — factor vs. specific variance split

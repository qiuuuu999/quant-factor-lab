# Factor Evaluation — S&P 500 (point-in-time), 2015-2025

Signal-quality diagnostics for the four pure-price factors. Factor values are formed point-in-time (look-ahead guard on); forward returns look strictly forward (formation close to next formation close). Rank-based, so invariant to winsorization/standardisation. This measures *signal quality*, not tradable P&L — the backtest engine (t+1 open fills, costs) is the tradable measure.

## Headline

| Factor | Mean IC | ICIR | IC t-stat | Hit rate | Decile L/S (ann.) | Top decile | Bottom decile | Monotonicity | Rank autocorr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| momentum_12_1 | -0.0036 | -0.017 | -0.19 | 52% | -5.33% | +9.45% | +10.27% | -0.08 | +0.90 |
| low_vol_252 | -0.0047 | -0.019 | -0.22 | 51% | -9.32% | +9.70% | +13.16% | -0.56 | +0.99 |
| reversal_1m | -0.0045 | -0.028 | -0.32 | 50% | -3.80% | +8.33% | +12.52% | -0.56 | +0.27 |
| amihud_illiq_21 | -0.0109 | -0.082 | -0.94 | 44% | -7.03% | +7.02% | +15.73% | -0.59 | +0.95 |

## Cross-factor rank correlation

Average cross-sectional Spearman ρ between factor values.

| | momentum_12_1 | low_vol_252 | reversal_1m | amihud_illiq_21 |
| --- | --- | --- | --- | --- |
| momentum_12_1 | +1.00 | +0.18 | -0.03 | -0.20 |
| low_vol_252 | +0.18 | +1.00 | -0.02 | -0.24 |
| reversal_1m | -0.03 | -0.02 | +1.00 | +0.07 |
| amihud_illiq_21 | -0.20 | -0.24 | +0.07 | +1.00 |

## Figures

- `<factor>_ic.png` — per-period rank IC with rolling and full-sample mean
- `<factor>_deciles.png` — annualised return by factor decile (monotonicity)
- `factor_correlation.png` — cross-factor rank-correlation heatmap

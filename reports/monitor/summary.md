# Factor Health Monitoring — S&P 500 (point-in-time), 2015-2025

CUSUM change-point test on each factor's per-period rank IC (rolling window 36 months); SPY rolling volatility / 200-day trend regime classification. See `docs/monitoring.md` for methodology.

## Factor health

| Factor | Status | Current rolling IC | Historical mean IC | Alert date | CUSUM stat | Critical value |
| --- | --- | --- | --- | --- | --- | --- |
| momentum_12_1 | OK | +0.0037 | -0.0036 | — | 0.48 | 1.36 |
| low_vol_252 | OK | -0.0406 | -0.0047 | — | 0.57 | 1.36 |
| reversal_1m | OK | -0.0529 | -0.0045 | — | 1.35 | 1.36 |
| amihud_illiq_21 | OK | -0.0206 | -0.0109 | — | 0.49 | 1.36 |

## Factor – regime fit (mean IC)

| Factor | low_vol_up | low_vol_down | high_vol_up | high_vol_down | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| momentum_12_1 | +0.0403 | +nan | -0.0593 | -0.0191 | low_vol_up | high_vol_up |
| low_vol_252 | +0.0276 | +nan | -0.0253 | -0.0517 | low_vol_up | high_vol_down |
| reversal_1m | -0.0195 | +nan | -0.0111 | +0.0466 | high_vol_down | low_vol_up |
| amihud_illiq_21 | -0.0323 | +nan | +0.0133 | +0.0007 | high_vol_up | low_vol_up |

## Figures

- `<factor>_health.png` — per-period IC, rolling mean, CUSUM alert marker
- `regime_heatmap.png` — factor x regime mean-IC heatmap

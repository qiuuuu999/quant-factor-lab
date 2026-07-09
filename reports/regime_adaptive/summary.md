# Regime-Adaptive Strategy Comparison, 2017-2025

- **Factors**: 12-1 momentum, low volatility, 1-month reversal, Amihud illiquidity
- **Regime-Adaptive**: composite score = regime-conditioned-IC-weighted combination of the four z-scored factors (negative_handling='invert', min_obs=6); top 20% by composite score, equal-weighted
- **Static Multi-Factor**: same selection rule, fixed equal (1/4 each) factor combination -- no regime conditioning
- **Pure Momentum**: 12-1 momentum only, top 20%, equal-weighted (identical to `reports/momentum_12_1/`)
- **Regime/IC warm-up**: 2015-01-01 .. 2017-01-01 (not traded; builds the regime-conditioned IC history the live period conditions on)
- **Live window**: 2017-01-01 .. 2025-12-31
- **Execution**: signal on month-end close, fill at next-session open (t+1), halted opens rolled forward
- **Costs**: 0.005/share commission, 5.0 bps slippage
- **Benchmark**: SPY

| Metric | Regime-Adaptive | Static Multi-Factor | Pure Momentum | SPY |
| --- | --- | --- | --- | --- |
| Total Return | 174.65% | 70.45% | 163.39% | 249.70% |
| CAGR | 11.92% | 6.12% | 11.40% | 14.97% |
| Annual Volatility | 24.80% | 19.46% | 20.45% | 18.48% |
| Sharpe Ratio | 0.58 | 0.40 | 0.63 | 0.85 |
| Max Drawdown | -49.30% | -42.52% | -37.73% | -33.72% |
| Max DD Peak | 2020-02-12 | 2020-02-05 | 2020-02-19 | 2020-02-19 |
| Max DD Trough | 2020-03-23 | 2020-03-23 | 2020-03-23 | 2020-03-23 |
| Calmar Ratio | 0.24 | 0.14 | 0.30 | 0.44 |
| Monthly Win Rate | 65.42% | 57.94% | 64.49% | 71.03% |
| Annual Turnover | 14.02x | 10.36x | 5.45x | 0.00x |

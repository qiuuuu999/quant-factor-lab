# Portfolio Construction Comparison — 12-1 Momentum, 2015-2025

- **Signal**: 12-1 momentum, top 20% (deciles 9-10), same selection for all three
- **Equal Weight**: `1/n` per selected name
- **Mean-Variance**: alpha = z-scored momentum; covariance = Ledoit-Wolf on trailing 24m returns; long-only, 2% cap, turnover_penalty=0.01, risk_aversion=5.0
- **Risk Parity**: equal risk contribution on the same Ledoit-Wolf covariance, 2% cap, no alpha
- **Execution**: signal on month-end close, fill at next-session open (t+1), halted opens rolled forward
- **Costs**: 0.005/share commission, 5.0 bps slippage
- **Window**: 2015-01-01 .. 2025-12-31
- **Benchmark**: SPY

| Metric | Equal Weight | Mean-Variance | Risk Parity |
| --- | --- | --- | --- |
| Total Return | 189.01% | 171.01% | 164.31% |
| CAGR | 10.16% | 9.51% | 9.26% |
| Annual Volatility | 19.48% | 21.25% | 18.45% |
| Sharpe Ratio | 0.59 | 0.53 | 0.57 |
| Max Drawdown | -37.73% | -37.31% | -37.11% |
| Max DD Peak | 2020-02-19 | 2020-02-19 | 2020-02-19 |
| Max DD Trough | 2020-03-23 | 2020-03-23 | 2020-03-23 |
| Calmar Ratio | 0.27 | 0.25 | 0.25 |
| Monthly Win Rate | 61.07% | 60.31% | 59.54% |
| Annual Turnover | 5.48x | 6.27x | 6.20x |
| Benchmark CAGR | 13.46% | 13.46% | 13.46% |
| Excess CAGR | -3.31% | -3.95% | -4.20% |
| Information Ratio | -0.32 | -0.28 | -0.45 |
| Tracking Error | 8.18% | 10.40% | 8.04% |

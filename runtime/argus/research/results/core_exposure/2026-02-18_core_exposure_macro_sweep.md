# Core Exposure – Macro Sweep
Date: 2026-02-18
Dataset: btcusd_3600s_2019-01-01_to_2025-12-30.csv
Fees: 10 bps
Slippage: 5 bps

## EMA2000 – Bear Cap 0.25
CAGR: 16.22%
Max DD: 42.25%
Calmar: 0.38

## EMA2000 – Bear Cap 0.15
CAGR: 17.91%
Max DD: 37.41%
Calmar: 0.48

## EMA2000 – Bear Cap 0.10
CAGR: 18.69%
Max DD: 35.04%
Calmar: 0.53

Crash Window (2021-07-01 to 2022-12-31):
- Peak→Trough DD Proxy (no fees): -18.36%
- Avg Exposure (window): 0.0921
- Time Exposed >0: 0.3174
- Time Exposure >0.5: 0.0957
- Avg Exposure (worst segment): 0.0743
- Time Exposure >0.5 (worst segment): 0.0645
- 2022 macro_bull %: 0.0457
- 2022 time exposure >0.5 %: 0.0452

## EMA2000 – Bear Cap 0.00
CAGR: 20.79%
Max DD: 31.26%
Calmar: 0.67

Crash Window (2021-07-01 to 2022-12-31):
- Peak→Trough DD Proxy (no fees): -17.11%
- Avg Exposure (window): 0.0713
- Time Exposed >0: 0.1092
- Time Exposure >0.5: 0.0957
- Avg Exposure (worst segment): 0.0472
- Time Exposure >0.5 (worst segment): 0.0645
- 2022 macro_bull %: 0.0457
- 2022 time exposure >0.5 %: 0.0452

Observations:
- Reducing Bear Cap from 0.25 → 0.15 improved both CAGR and drawdown, suggesting 0.25 allowed excessive participation during low-quality bear rallies / chop decay.
- 0.10 continued to improve the risk-adjusted profile (higher Calmar, lower MDD, higher CAGR) with only modestly reduced exposure.
- 0.00 (strict macro-bear suppression) dominated the sweep on full-cycle metrics (highest CAGR, lowest MDD, highest Calmar) while materially reducing average exposure and time-in-market.
- Crash-window behavior also improved moving from 0.10 → 0.00: lower average exposure and a slightly improved peak→trough DD proxy.
- “Time exposure >0.5” remained unchanged between 0.10 and 0.00, likely because those high-exposure periods occur during macro_bull, where bear-cap does not apply. A useful next diagnostic enhancement is splitting exposure stats by macro_bull vs macro_bear.

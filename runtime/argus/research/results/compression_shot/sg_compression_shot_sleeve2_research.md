# Itera Dynamics  
# Sleeve 2 — Compression Shot Research Log  
`sg_compression_shot_v1 → v3`

---

## 1️⃣ Mandate

Sleeve 2 exists to provide **orthogonal alpha** relative to Core (Sleeve 1).

**Core (Sleeve 1) mandate:**
- Structural BTC participation
- EMA2000 macro gating
- Bear exposure cap = 0.00
- Drawdown suppression anchor

**Sleeve 2 mandate:**
- Monetize compression → expansion transitions
- Capture convexity in transitional regimes
- Remain deterministic and stateless
- Preserve closed_only=True safety
- Avoid any Layer 1 (Regime Engine) or Layer 3 (Prime) modifications

This sleeve is not designed to anchor portfolio risk.  
It is designed to add opportunistic convex participation in environments where Core underperforms.

---

# 2️⃣ v1 — Pure Regime-Gated Compression Breakout

### Design

Entry required:
- `regime == VOL_COMPRESSION`
- Bollinger width expansion ≥ 1.20× previous bar
- `close > bb_upper`

Exit triggered on:
- PANIC
- TREND_DOWN
- VOL_EXPANSION
- Any non-compression regime

Exposure sizing:
- `cap_bull = 0.20`
- `cap_bear = 0.05`
- Macro filter via EMA (price > EMA AND slope > 0)

### Results

**Full Cycle (2019–2025):**
- Time in Market: ~0.70%
- Avg Exposure: ~0.08%
- Max Drawdown: ~1.9%
- Slightly negative return

**Crash Window (2021–2022):**
- ENTER_LONG: 7
- Avg hold: ~1.17 bars
- Max hold: 2 bars

### Diagnosis

Primary bottleneck:
- `VOL_COMPRESSION` regime frequency < 0.5%
- Immediate exit when regime label changed

v1 behaved like a one-bar impulse trade.

Conclusion:
Exit logic structurally suppressed persistence.

---

# 3️⃣ v2 — Persistence Fix

### Structural Change

Removed exit on regime churn.

Hard exits only:
- PANIC
- TREND_DOWN
- VOL_EXPANSION

If `in_position`:
- HOLD through regime transitions
- Prime horizon manages duration

No state added.  
No Layer 1 or Layer 3 changes.

### Results

**Full Cycle (2019–2025):**
- Time in Market: ~3.86%
- Avg Exposure: ~0.44%
- Slightly positive return
- Drawdown still controlled

**Crash Window:**
- Entries still sparse
- Avg hold still ~1–2 bars

### Diagnosis

Exit logic corrected.

However:
Entry eligibility remained too restrictive.

Root cause shifted from exit churn → regime scarcity.

---

# 4️⃣ v3 — Compression Memory (Regime Scarcity Fix)

### Structural Innovation

Expanded entry eligibility window.

Eligible if:
- `regime == VOL_COMPRESSION`
**OR**
- `compression_recent == True`

Where:
- `compressed_state = (bb_width ≤ width_compress_max)`
- `compression_recent = any(compressed_state in last mem_bars)`
- Derived purely from df indicators (live-safe)
- No stored regime series
- Deterministic and stateless

New env knobs:
- `SG_COMP_MEMORY_BARS` (default 72)
- `SG_COMP_WIDTH_COMPRESS_MAX` (default 0.06)
- `SG_COMP_MEMORY_MIN_HITS` (default 1)

Hard exits identical to v2.

---

## v3 Results

### Full Cycle (2019–2025)

- Total Return: +50.42%
- CAGR: 6.03%
- Max Drawdown: 7.87%
- Calmar: 0.77
- Avg Exposure: 6.15%
- Time in Market: 38.04%
- Final Equity: $15,042

### 2023–2025 Slice

- CAGR: 1.51%
- Max Drawdown: 6.82%
- Avg Exposure: 6.56%
- Time in Market: 38.44%

### 2021–2022 Crash Window

- Max Exposure: 0.20
- Peak → Trough DD proxy: ~-2.0%
- Exposure remains modest in sustained bear

---

# 5️⃣ Structural Observations

1. Regime scarcity was the dominant bottleneck in v1 and v2.
2. Exit logic was not the limiting factor after v2.
3. Compression memory materially increased opportunity frequency.
4. Sleeve now participates meaningfully (~38% of bars).
5. Drawdowns remain controlled relative to exposure level.
6. No Layer 1 modifications were required.

---

# 6️⃣ Architectural Integrity

All iterations preserved:

- Deterministic behavior
- `closed_only=True` compatibility
- No forming candle leakage
- No persistent state
- No broker/wallet interaction
- Strict `StrategyIntent` contract
- Prime-compatible HOLD semantics

Sleeve 2 remains a clean Layer 2 module.

---

# 7️⃣ Open Research Questions

1. Is compression memory too permissive?
   - Tune `SG_COMP_MEMORY_MIN_HITS`
   - Adjust `SG_COMP_MEMORY_BARS`
   - Refine `SG_COMP_WIDTH_COMPRESS_MAX`

2. Should exit logic remain strictly regime-based?
   - Currently unchanged from v2
   - No time-based failure exits added

3. What is the optimal capital allocation vs Core?
   - Sleeve now capable of meaningful participation
   - Requires Layer 3 integration testing

---

# 8️⃣ Current Status

Core (Sleeve 1):
- Locked
- Production mandate defined

Sleeve 2:
- v3 complete
- Regime scarcity addressed
- Behavior materially improved
- Ready for parameter calibration and portfolio allocation testing

---

# Conclusion

Sleeve 2 evolved through three structural stages:

- v1 — Entry logic correct, persistence broken  
- v2 — Persistence fixed, opportunity too sparse  
- v3 — Opportunity expanded via compression memory  

We are no longer debugging mechanics.

We are calibrating mandate expression.

Sleeve 2 has transitioned from proof-of-concept to a viable orthogonal alpha sleeve within the Itera multi-sleeve architecture.
# Sleeve 2 Mean Reversion Candidates

## 🎯 What's Built

Three Sleeve 2 candidates, all following your exact Layer 2 architecture:

### **Candidate A: RSI Mean Reversion** (`sg_mean_reversion_a.py`)
**Strategy:**
- Entry: RSI(14) < 30 in CHOP/VOL_COMPRESSION regimes
- Exit: RSI(14) > 60 OR regime flip
- Horizon: 12 hours
- Exposure: 0.25 (fixed, conservative)

**Confidence scaling:**
- RSI < 20: 0.75 (extremely oversold)
- RSI < 25: 0.65 (very oversold)
- RSI < 30: 0.55 (moderately oversold)

**Profile:** High frequency, quick entries/exits, density-focused

---

### **Candidate B: Bollinger Mean Reversion** (`sg_mean_reversion_b.py`)
**Strategy:**
- Entry: Price < BB_lower (20, 2.0) in CHOP/VOL_COMPRESSION
- Exit: Price > BB_mid OR regime flip
- Horizon: 18 hours
- Exposure: 0.30 base (volatility-scaled via ATR)

**Confidence scaling:**
- Deviation > 4%: 0.75
- Deviation > 2%: 0.65
- Else: 0.55

**Volatility scaling:**
- Target ATR%: 1.0%
- Scale cap: 1.20x
- Exposure range: 0.30-0.45

**Profile:** Vol-aware, adaptive sizing, medium frequency

---

### **Candidate C: Hybrid (RSI + Bollinger)** (`sg_mean_reversion_c.py`)
**Strategy:**
- Entry: RSI < 35 AND Price < BB_lower (confluence required)
- Exit: RSI > 55 OR Price > BB_mid (either triggers)
- Horizon: 24 hours (longer hold for confluence)
- Exposure: 0.35 base (scaled by signal strength)

**Confidence scaling:**
- Extreme confluence (RSI < 25, px < lower*0.99): 0.85, exposure 0.50
- Strong confluence (RSI < 30): 0.75, exposure 0.35
- Moderate: 0.65, exposure 0.30

**Profile:** Lower frequency, higher conviction, better win rate (theory)

---

## 📋 Architecture Compliance

All three strategies:
✅ Call `classify_regime()` from Layer 1  
✅ Return proper `StrategyIntent` dict (action, confidence, desired_exposure_frac, horizon_hours, reason, meta)  
✅ `closed_only=True` default  
✅ Mirror Layer 1's `dropped_last_row` behavior  
✅ Deterministic (no side effects, no state)  
✅ OHLCV normalization (handles capitalized columns)  
✅ Environment variable configuration  
✅ Fail-closed on missing data/NaN  
✅ PANIC regime force exit  

**Ready to drop into:** `runtime/argus/research/strategies/`

---

## 🔧 Environment Variables

### **Candidate A (RSI)**
```bash
SG_MR_A_RSI_LEN=14              # RSI period
SG_MR_A_RSI_OVERSOLD=30         # Entry threshold
SG_MR_A_RSI_EXIT=60             # Exit threshold
SG_MR_A_HORIZON_HOURS=12        # Hold time
SG_MR_A_EXPOSURE_FRAC=0.25      # Position size
```

### **Candidate B (Bollinger)**
```bash
SG_MR_B_BB_LEN=20               # Bollinger period
SG_MR_B_BB_STD=2.0              # Std dev
SG_MR_B_HORIZON_HOURS=18
SG_MR_B_BASE_EXPOSURE=0.30
SG_MR_B_ATR_LEN=14              # For vol scaling
SG_MR_B_TARGET_ATR_PCT=0.010    # 1% target
SG_MR_B_SCALE_CAP=1.20          # Max scaling
```

### **Candidate C (Hybrid)**
```bash
SG_MR_C_RSI_LEN=14
SG_MR_C_RSI_OVERSOLD=35         # Higher threshold (confluence)
SG_MR_C_RSI_EXIT=55             # Lower exit (faster)
SG_MR_C_BB_LEN=20
SG_MR_C_BB_STD=2.0
SG_MR_C_HORIZON_HOURS=24        # Longer hold
SG_MR_C_BASE_EXPOSURE=0.35
```

---

## 🧪 Testing Plan

### **Step 1: Smoke Test (No Backtest)**
```bash
cd runtime/argus

# Test Candidate A
python -c "
import sys
sys.path.insert(0, r'.')
# Copy sg_mean_reversion_a.py to research/strategies/
from research.strategies.sg_mean_reversion_a import generate_intent
print('Candidate A: OK')
"

# Repeat for B and C
```

### **Step 2: Backtest Comparison**

You'll need to create a backtest runner that:

```python
# Pseudocode - adapt to your backtest harness
from research.engine.backtest import run_backtest

results = {
    "Core Alone": run_backtest(
        strategy="sg_core_exposure_v1",
        start="2020-01-01",
        end="2025-02-21"
    ),
    "Core + Sleeve2A": run_backtest_blend(
        core="sg_core_exposure_v1",
        sleeve2="sg_mean_reversion_a",
        allocation="equal"  # or regime-weighted
    ),
    "Core + Sleeve2B": run_backtest_blend(
        core="sg_core_exposure_v1",
        sleeve2="sg_mean_reversion_b"
    ),
    "Core + Sleeve2C": run_backtest_blend(
        core="sg_core_exposure_v1",
        sleeve2="sg_mean_reversion_c"
    ),
}

# Compare metrics
for name, result in results.items():
    print(f"{name}:")
    print(f"  CAGR: {result.cagr:.2%}")
    print(f"  Max DD: {result.max_dd:.2%}")
    print(f"  Calmar: {result.calmar:.2f}")
    print(f"  Sharpe: {result.sharpe:.2f}")
    print(f"  Trades: {result.trade_count}")
    print()
```

### **Decision Criteria**

**Keep if:**
- `Calmar(Core+Sleeve2) > Calmar(Core) * 1.05` (5% improvement)
- Drawdown not worse
- Trade count reasonable (not excessive churn)

**Pick winner:**
- Highest Calmar
- Best risk-adjusted returns
- Smooth equity curve

---

## 📊 Expected Behavior

### **Candidate A (RSI):**
- **Pros:** Simple, fast entries, clear signals
- **Cons:** May whipsaw in choppy markets
- **Best if:** You want high frequency, density

### **Candidate B (Bollinger):**
- **Pros:** Vol-aware, adaptive sizing
- **Cons:** More complex, may size too big in compression
- **Best if:** You want vol management

### **Candidate C (Hybrid):**
- **Pros:** Higher conviction, better win rate (theory)
- **Cons:** Lower frequency, may miss opportunities
- **Best if:** You want quality over quantity

**My prediction:** Candidate C likely wins on Calmar (lower DD, higher confidence).

---

## 🚀 Integration (After Testing)

### **Option 1: External Strategy (Quick)**
```bash
# Use env vars (no code changes to Prime)
export ARGUS_STRATEGY_MODULE="research.strategies.sg_mean_reversion_c"
export ARGUS_STRATEGY_FUNC="generate_intent"
export PRIME_DRY_RUN="1"

# Run
python -c "import sys; sys.path.insert(0, r'./runtime/argus'); from apex_core.signal_generator import generate_signals; generate_signals()"
```

### **Option 2: Multi-Sleeve Blending (Proper)**

Requires Layer 3 implementation:
1. Load both Core + Sleeve2 strategies
2. Get intents from both
3. Blend exposures (regime-weighted or static)
4. Route capital

**Example blend logic:**
```python
core_intent = core_strategy.generate_intent(df, ctx)
sleeve2_intent = sleeve2_strategy.generate_intent(df, ctx)

if regime == "TREND_UP":
    # Core takes priority
    return core_intent
elif regime in ["CHOP", "VOL_COMPRESSION"]:
    # Sleeve2 takes priority
    return sleeve2_intent
else:
    # Blend or defer to strongest signal
    return max([core_intent, sleeve2_intent], key=lambda x: x['confidence'])
```

---

## 🔍 Known Gaps (What I Can't Do)

❌ **Run backtests** - no access to your data/environment  
❌ **Push to GitHub** - waiting for you to create feature branch  
❌ **Test with real regime engine** - need your env  

✅ **What's ready:**
- All 3 strategies coded
- Architectural compliance verified
- Parameter suggestions provided
- Integration paths documented

---

## 📝 Next Steps

1. **You:** Copy these 3 files to `runtime/argus/research/strategies/`
2. **You:** Run smoke tests (verify imports work)
3. **You:** Backtest Core alone (baseline)
4. **You:** Backtest Core+Sleeve2A, Core+Sleeve2B, Core+Sleeve2C
5. **You:** Compare metrics, pick winner
6. **Me:** Wire winner into production (if needed)

---

## 💬 Questions to Answer (After Backtesting)

1. **Which candidate has best Calmar?**
2. **Does ANY candidate improve Core by >5%?**
3. **Trade frequency acceptable?** (not too high/low)
4. **Drawdown behavior acceptable?**
5. **Want to tune parameters or lock it?**

---

**Files ready. Waiting for your backtest results to proceed.** 🚀

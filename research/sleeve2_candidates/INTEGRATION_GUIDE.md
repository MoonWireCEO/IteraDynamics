# Integration Guide - Sleeve 2 Candidates

## 📦 Files to Move

```
# Source (what I built):
/home/clawd/clawd/itera-sleeve2/
├── sg_mean_reversion_a.py
├── sg_mean_reversion_b.py
├── sg_mean_reversion_c.py
├── SLEEVE2_CANDIDATES_README.md
└── INTEGRATION_GUIDE.md (this file)

# Destination (your repo):
~/IteraDynamics_Mono/runtime/argus/research/strategies/
├── sg_mean_reversion_a.py  ← copy here
├── sg_mean_reversion_b.py  ← copy here
├── sg_mean_reversion_c.py  ← copy here
```

## 🔧 Step-by-Step Integration

### **1. Copy Files**
```bash
cd ~/IteraDynamics_Mono

# Copy strategies to your repo
cp /home/clawd/clawd/itera-sleeve2/sg_mean_reversion_*.py \
   runtime/argus/research/strategies/

# Verify
ls -la runtime/argus/research/strategies/sg_mean_reversion_*.py
```

### **2. Smoke Test (Verify Imports)**
```bash
cd runtime/argus

# Test Candidate A
python -c "
import sys
sys.path.insert(0, r'.')
from research.strategies.sg_mean_reversion_a import generate_intent
print('✓ Candidate A imports successfully')
"

# Test Candidate B
python -c "
import sys
sys.path.insert(0, r'.')
from research.strategies.sg_mean_reversion_b import generate_intent
print('✓ Candidate B imports successfully')
"

# Test Candidate C
python -c "
import sys
sys.path.insert(0, r'.')
from research.strategies.sg_mean_reversion_c import generate_intent
print('✓ Candidate C imports successfully')
"
```

### **3. Test with Regime Engine**
```bash
cd runtime/argus

# Create test script: test_sleeve2.py
cat > test_sleeve2.py << 'EOF'
import sys
sys.path.insert(0, r'.')

import pandas as pd
from research.strategies.sg_mean_reversion_a import generate_intent

# Load sample data (use your flight_recorder.csv or test data)
df = pd.read_csv("flight_recorder.csv")
df = df.tail(100)  # Last 100 bars

# Test strategy
ctx = {}
intent = generate_intent(df, ctx, closed_only=True)

print("Strategy Test Results:")
print(f"  Action: {intent['action']}")
print(f"  Confidence: {intent['confidence']:.2f}")
print(f"  Exposure: {intent['desired_exposure_frac']:.2f}")
print(f"  Reason: {intent['reason']}")
print(f"  Regime: {intent['meta']['regime_state']}")
EOF

python test_sleeve2.py
```

### **4. Quick Live Test (Dry Run)**
```bash
cd runtime/argus

# Set env vars for Candidate A
export ARGUS_STRATEGY_MODULE="research.strategies.sg_mean_reversion_a"
export ARGUS_STRATEGY_FUNC="generate_intent"
export PRIME_DRY_RUN="1"
export ARGUS_MODE="prime"

# Run once (dry run)
python -c "
import sys
sys.path.insert(0, r'.')
from apex_core.signal_generator import generate_signals
generate_signals()
"

# Check cortex.json for output
cat cortex.json | python -m json.tool
```

---

## 🧪 Backtest Integration

### **Option A: Use Your Existing Backtest Harness**

If you have a backtest runner in `research/`:

```bash
cd ~/IteraDynamics_Mono/research

# Example backtest call (adapt to your harness)
python backtest_runner.py \
  --strategy sg_mean_reversion_a \
  --start 2020-01-01 \
  --end 2025-02-21 \
  --output results/sleeve2_candidate_a.json
```

### **Option B: Create Simple Comparison Script**

If you don't have a harness, here's a minimal one:

```python
# backtest_sleeve2_comparison.py
import sys
sys.path.insert(0, r'./runtime/argus')

import pandas as pd
from research.regime import classify_regime
from research.strategies.sg_core_exposure_v1 import generate_intent as core_strategy
from research.strategies.sg_mean_reversion_a import generate_intent as sleeve2a
from research.strategies.sg_mean_reversion_b import generate_intent as sleeve2b
from research.strategies.sg_mean_reversion_c import generate_intent as sleeve2c

# Load data
df = pd.read_csv("data/btc_hourly_2020_2025.csv")  # Your data path
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Simple backtest loop
def backtest_strategy(strategy_func, df):
    equity = 1.0
    position = 0.0
    trades = []
    
    for i in range(100, len(df)):
        window = df.iloc[:i]
        intent = strategy_func(window, {}, closed_only=True)
        
        action = intent['action']
        exposure = intent['desired_exposure_frac']
        
        # Simulate trades (simplified)
        if action == "ENTER_LONG" and position == 0:
            position = exposure
            trades.append(('BUY', i, df.iloc[i]['close'], exposure))
        elif action == "EXIT_LONG" and position > 0:
            # Calculate P&L
            entry_price = trades[-1][2] if trades else df.iloc[i]['close']
            pnl = (df.iloc[i]['close'] / entry_price - 1) * position
            equity *= (1 + pnl)
            position = 0
            trades.append(('SELL', i, df.iloc[i]['close'], pnl))
    
    return {
        'final_equity': equity,
        'trade_count': len([t for t in trades if t[0] == 'BUY']),
        'trades': trades
    }

# Run backtests
print("Running backtests...")
results = {
    "Core Alone": backtest_strategy(core_strategy, df),
    "Sleeve2A (RSI)": backtest_strategy(sleeve2a, df),
    "Sleeve2B (Bollinger)": backtest_strategy(sleeve2b, df),
    "Sleeve2C (Hybrid)": backtest_strategy(sleeve2c, df),
}

# Print results
for name, result in results.items():
    print(f"\n{name}:")
    print(f"  Final Equity: {result['final_equity']:.3f}")
    print(f"  Trade Count: {result['trade_count']}")
    print(f"  Return: {(result['final_equity'] - 1) * 100:.2f}%")
```

**Note:** This is simplified. Use your actual backtest infrastructure for proper metrics (drawdown, Sharpe, Calmar, etc.).

---

## 📊 What to Look For

After backtesting, compare:

| Metric | Core Alone | Core+Sleeve2A | Core+Sleeve2B | Core+Sleeve2C |
|--------|-----------|---------------|---------------|---------------|
| CAGR | ? | ? | ? | ? |
| Max DD | ? | ? | ? | ? |
| Calmar | ? | ? | ? | ? |
| Sharpe | ? | ? | ? | ? |
| Trades | ? | ? | ? | ? |

**Decision:**
- If any candidate improves Calmar by >5%: **Pick that one**
- If multiple improve: **Pick highest Calmar**
- If none improve: **Tune parameters or redesign**

---

## 🚀 Going Live (After Picking Winner)

### **Option 1: Single Strategy (Quick)**
```bash
# Use Candidate C (for example)
export ARGUS_STRATEGY_MODULE="research.strategies.sg_mean_reversion_c"
export ARGUS_STRATEGY_FUNC="generate_intent"
export PRIME_DRY_RUN="0"  # LIVE MODE
export ARGUS_MODE="prime"

# Set Candidate C params
export SG_MR_C_RSI_OVERSOLD="35"
export SG_MR_C_BASE_EXPOSURE="0.35"

# Run live
cd runtime/argus
python run_live.py
```

### **Option 2: Multi-Sleeve (Proper - Requires Layer 3)**

You'll need to:
1. Create sleeve blending logic in `signal_generator.py`
2. Load both Core + Sleeve2 strategies
3. Combine exposures based on regime
4. Execute blended signal

**Blending Example:**
```python
# In signal_generator.py
core_intent = core_strategy.generate_intent(df, ctx)
sleeve2_intent = sleeve2_strategy.generate_intent(df, ctx)

regime = classify_regime(df).label

if regime == "TREND_UP":
    # Core handles trends
    final_intent = core_intent
elif regime in ["CHOP", "VOL_COMPRESSION"]:
    # Sleeve2 handles chop
    final_intent = sleeve2_intent
else:
    # Blend based on confidence
    if core_intent['confidence'] > sleeve2_intent['confidence']:
        final_intent = core_intent
    else:
        final_intent = sleeve2_intent

# Execute final_intent
```

---

## ⚠️ Important Notes

1. **Always dry-run first** (`PRIME_DRY_RUN=1`)
2. **Test with paper trading** before live
3. **Monitor cortex.json** for strategy decisions
4. **Watch for regime flips** (strategies should respect them)
5. **Capital management** - don't over-leverage in multi-sleeve mode

---

## 🐛 Troubleshooting

### **Import Error:**
```python
ModuleNotFoundError: No module named 'research.strategies.sg_mean_reversion_a'
```
**Fix:** Check file is in `runtime/argus/research/strategies/` and you're running from `runtime/argus/`

### **Regime Error:**
```
regime_error: "RegimeLabel object has no attribute 'label'"
```
**Fix:** Check your regime engine returns correct object. Strategies expect `reg.label` and `reg.meta`.

### **NaN Indicators:**
```
reason: "nan_guard(indicators_invalid)"
```
**Fix:** Not enough data. Need at least 20-30 bars for indicators to compute.

---

## 📞 Next Steps

1. Copy files to repo
2. Run smoke tests (imports)
3. Run backtests (compare metrics)
4. Report results back to me
5. I'll help wire the winner into production

**Files are ready. Let me know backtest results!** 🚀

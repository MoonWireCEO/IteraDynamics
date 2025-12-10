import os
import sys
from pathlib import Path

# 1. Define the content of the Regime Detector
code_content = """
import pandas as pd
import numpy as np

class MarketRegimeDetector:
    def __init__(self, volatility_window=20, trend_window=50):
        self.vol_window = volatility_window
        self.trend_window = trend_window

    def detect_regime(self, df):
        if len(df) < self.trend_window:
            return "WARMUP"

        df = df.copy()
        df['returns'] = df['Price'].pct_change()
        df['volatility'] = df['returns'].rolling(window=self.vol_window).std()
        
        # Dynamic Threshold (compare vs last 100 periods)
        long_term_vol = df['volatility'].rolling(window=100).mean().iloc[-1]
        current_vol = df['volatility'].iloc[-1]

        if pd.isna(long_term_vol) or pd.isna(current_vol):
            return "WARMUP"

        vol_state = "HIGH_VOL" if current_vol > long_term_vol else "LOW_VOL"

        # Trend Calculation
        sma = df['Price'].rolling(window=self.trend_window).mean().iloc[-1]
        current_price = df['Price'].iloc[-1]
        threshold_buffer = 0.0005 
        
        if current_price > sma * (1 + threshold_buffer):
            trend_state = "BULL"
        elif current_price < sma * (1 - threshold_buffer):
            trend_state = "BEAR"
        else:
            trend_state = "SIDEWAYS"

        return f"{trend_state}_{vol_state}"

class MetaStrategySelector:
    def get_strategy(self, regime):
        mapping = {
            "BULL_LOW_VOL": "Aggressive_Trend_Follow",
            "BULL_HIGH_VOL": "Conservative_Trend_Follow",
            "BEAR_LOW_VOL": "Short_Selling",
            "BEAR_HIGH_VOL": "Cash_Protection",
            "SIDEWAYS_LOW_VOL": "Mean_Reversion",
            "SIDEWAYS_HIGH_VOL": "Cash_Protection",
            "WARMUP": "Data_Collection"
        }
        return mapping.get(regime, "Cash_Protection")

class PositionSizer:
    def __init__(self, risk_percent=0.10):
        self.risk_percent = risk_percent

    def calculate_size(self, current_balance):
        return round(current_balance * self.risk_percent, 2)
"""

# 2. Determine target path (apex_core/regime_detector.py)
target_path = Path("apex_core") / "regime_detector.py"

# Ensure apex_core exists
if not target_path.parent.exists():
    print(f"❌ Error: {target_path.parent} does not exist.")
    sys.exit(1)

# 3. Write the file
with open(target_path, "w", encoding="utf-8") as f:
    f.write(code_content)

print(f"✅ Successfully created: {target_path}")

# 4. Verify Import
sys.path.append(str(target_path.parent)) # Add apex_core to path
try:
    import regime_detector
    print("✅ Verification: Import successful!")
    print(f"   Classes found: {dir(regime_detector)}")
except ImportError as e:
    print(f"❌ Verification Failed: {e}")
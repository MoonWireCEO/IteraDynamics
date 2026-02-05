import pandas as pd
import numpy as np
import time

class MarketRegimeDetector:
    def __init__(self, volatility_window=20, trend_window=50):
        self.vol_window = volatility_window
        self.trend_window = trend_window
        
        # --- NEW: ANTI-FLICKER STATE ---
        self.current_regime = "WARMUP"
        self.last_switch_time = 0
        self.MIN_REGIME_DURATION = 300  # 5 Minutes (Time Lock)

    def detect_regime(self, df):
        # 1. CALCULATE RAW REGIME (Original Math)
        # ---------------------------------------
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

        proposed_regime = f"{trend_state}_{vol_state}"

        # 2. APPLY ANTI-FLICKER LOGIC
        # ---------------------------
        
        # Case A: First valid regime (Initialization)
        if self.current_regime == "WARMUP":
            self.current_regime = proposed_regime
            self.last_switch_time = time.time()
            return proposed_regime

        # Case B: Regime Change Proposed
        if proposed_regime != self.current_regime:
            # Check if we are inside the "Time Lock"
            time_since_last_switch = time.time() - self.last_switch_time
            
            if time_since_last_switch < self.MIN_REGIME_DURATION:
                # LOCKED: Ignore the new signal, return the old one
                return self.current_regime
            else:
                # UNLOCKED: Commit the switch
                print(f">> 🔄 REGIME SWITCH: {self.current_regime} -> {proposed_regime}")
                self.current_regime = proposed_regime
                self.last_switch_time = time.time()
        
        return self.current_regime

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
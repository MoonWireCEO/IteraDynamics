import pandas as pd
import numpy as np

def build_features(prices_map):
    """
    Takes raw prices, returns rich features (RSI, Bollinger, Volatility).
    Input: {'BTC': df_with_close}
    Output: {'BTC': df_with_features}
    """
    features_map = {}
    
    for symbol, df in prices_map.items():
        # Work on a copy to avoid SettingWithCopy warnings
        df = df.copy()
        
        # 1. Returns
        df["ret_1h"] = df["close"].pct_change()
        
        # 2. Volatility & Safety Logic (CRITICAL FOR GOVERNANCE)
        # Raw Volatility (6-hour rolling std dev)
        df["vol_6h"] = df["ret_1h"].rolling(6).std()
        
        # Volatility Z-Score (How weird is this volatility?)
        # We compare current vol to the average vol of the last 100 hours
        vol_mean = df["vol_6h"].rolling(100).mean()
        vol_std = df["vol_6h"].rolling(100).std()
        
        # Z = (Current - Mean) / StdDev
        # If Z > 2.0, it means volatility is 2 standard deviations higher than normal (Crash/Pump)
        df["vol_z"] = (df["vol_6h"] - vol_mean) / vol_std
        
        # Governance Flag: 1.0 if Volatility is dangerous, 0.0 if safe
        df["high_vol"] = np.where(df["vol_z"] > 2.0, 1.0, 0.0)
        
        # 3. RSI (Relative Strength Index) - 14 periods
        # Measures if the asset is Overbought (>70) or Oversold (<30)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        
        # Avoid division by zero
        loss = loss.replace(0, 0.00001)
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        
        # 4. Bollinger Bands (20 periods, 2 std devs)
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        
        df["bb_upper"] = sma_20 + (2 * std_20)
        df["bb_lower"] = sma_20 - (2 * std_20)
        
        # Feature: Position within the bands (0.0 = Lower Band, 1.0 = Upper Band)
        df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # 5. Cleanup
        # Indicators need history to warm up. Backfill NaNs.
        df = df.bfill().fillna(0.0)
        
        features_map[symbol] = df
        
    return features_map
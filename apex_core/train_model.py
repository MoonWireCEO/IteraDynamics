import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import apex_core.feature_builder as fb

def train_brain():
    print("🧠 BRAIN TRANSPLANT: HIGH FREQUENCY EDITION...")
    
    # 1. Fetch Real Data (7 Days of 1-Minute Data)
    # 7 days is the max yfinance allows for 1m interval
    print("🔌 Downloading 7 days of 1-Minute Bitcoin Data...")
    df = yf.download("BTC-USD", period="7d", interval="1m", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Rename and Keep Only Close (Ignore Volume to match live feed)
    df = df.rename(columns={"Close": "close"})
    df = df[["close"]] 
    
    # 2. Build Features
    print("🛠️  Engineering Features (RSI, Volatility, etc.)...")
    prices_map = {"BTC": df}
    features_map = fb.build_features(prices_map)
    df_features = features_map["BTC"]
    
    # 3. Create Target (The "Teacher")
    # If Price in 1 minute > Price now + 0.05% (Scalp Target), Signal = 1
    # We lower the target threshold because 1-minute moves are smaller.
    future_close = df_features["close"].shift(-1)
    # Target: 0.05% move (approx $50 on $90k BTC)
    df_features["target"] = (future_close > df_features["close"] * 1.0005).astype(int)
    
    df_features = df_features.dropna()

    # 4. Prepare Training Data
    # Explicitly drop 'close' so the model only sees derived features
    X = df_features.select_dtypes(include=[np.number]).drop(columns=["target", "close"], errors="ignore")
    y = df_features["target"]
    
    print(f"📊 Training on {len(X)} minute-candles with {X.shape[1]} features...")
    
    # 5. Train Random Forest
    # n_estimators=100 (100 Decision Trees voting)
    # max_depth=10 (Prevent overfitting)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # 6. Save the Brain
    save_path = "apex_core/models/btc_model_v1.pkl"
    joblib.dump(model, save_path)
    print(f"\n✅ SAVED HIGH-FREQUENCY BRAIN TO: {save_path}")

if __name__ == "__main__":
    train_brain()
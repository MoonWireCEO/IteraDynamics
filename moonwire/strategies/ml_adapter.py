import os
import joblib
import pandas as pd
import numpy as np
import logging
from collections import deque
from pathlib import Path
from moonwire.lib import feature_builder as fb

logger = logging.getLogger(__name__)

class MLStrategyAdapter:
    def __init__(self, symbol, models_dir="models", lookback_window=48):
        self.symbol = symbol
        self.models_dir = Path(models_dir)
        self.lookback_window = lookback_window
        
        # Memory buffer for prices
        self.prices = deque(maxlen=lookback_window + 50) 

        # Cache for loaded models
        self.loaded_models = {}

        # --- THE SWITCHBOARD ---
        self.strategy_map = {
            "Aggressive_Trend_Follow": "random_forest.pkl",
            "Conservative_Trend_Follow": "gradient_boost.pkl",
            "Short_Selling": "logistic_regression.pkl",
            "Mean_Reversion": "svm.pkl",
            "default": "btc_model_v1.pkl"
        }

    def _get_model(self, strategy_name):
        """Lazily loads the requested model."""
        filename = self.strategy_map.get(strategy_name, self.strategy_map["default"])
        
        if filename in self.loaded_models:
            return self.loaded_models[filename]

        model_path = self.models_dir / filename
        
        # Safety fallback
        if not model_path.exists():
            if strategy_name != "default":
                filename = self.strategy_map["default"]
                model_path = self.models_dir / filename

        try:
            if not model_path.exists():
                logger.error(f"❌ CRITICAL: No model found at {model_path}")
                return None

            logger.info(f"Loading Model for Strategy [{strategy_name}]: {filename}")
            model = joblib.load(model_path)
            self.loaded_models[filename] = model
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load model {filename}: {e}")
            return None

    def analyze(self, current_price, strategy="Aggressive_Trend_Follow"):
        """
        Returns a dictionary with the signal and the 'Why'.
        """
        self.prices.append(current_price)

        result = {
            "signal": "HOLD",
            "rsi": 0.0,
            "bb_pos": 0.0,
            "vol_z": 0.0,
            "confidence": 0.0,
            "reason": "Warming Up",
            "active_strategy": strategy
        }

        if len(self.prices) < self.lookback_window:
            return result

        # --- Feature Engineering ---
        df = pd.DataFrame(list(self.prices), columns=["close"])
        prices_map = {self.symbol: df}
        
        try:
            features_map = fb.build_features(prices_map)
            df_features = features_map[self.symbol]
            
            # Get latest row features
            latest = df_features.iloc[[-1]].copy()
            
            # Extract Metrics for the Dashboard
            # Use .get() with defaults to avoid crashes if calc fails
            result["rsi"] = round(latest.get("rsi_14", pd.Series([0])).item(), 1)
            result["bb_pos"] = round(latest.get("bb_pos", pd.Series([0])).item(), 2)
            result["vol_z"] = round(latest.get("vol_z", pd.Series([0])).item(), 2)
            result["reason"] = "Inference Complete"

            # --- CRITICAL FIX: COLUMN TRANSLATION ---
            # Map Live Feature Names -> Training Feature Names
            rename_map = {
                "rsi_14": "RSI",
                "bb_pos": "BB_Pos",
                "vol_z": "Vol_Z"
            }
            inference_data = latest.rename(columns=rename_map)
            
            # Select ONLY the columns the model expects
            expected_cols = ["RSI", "BB_Pos", "Vol_Z"]
            
            # Validation
            missing = [c for c in expected_cols if c not in inference_data.columns]
            if missing:
                result["reason"] = f"Missing features: {missing}"
                return result
                
            inference_data = inference_data[expected_cols]
            
            # --- DYNAMIC PREDICTION ---
            model = self._get_model(strategy)
            
            if model:
                prediction = model.predict(inference_data)[0]
                
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(inference_data)[0] 
                    confidence = probs[prediction]
                else:
                    confidence = 1.0

                result["confidence"] = round(confidence * 100, 1)

                # Decision Logic (1=BUY, 0=SELL)
                if prediction == 1 and confidence > 0.55:
                    result["signal"] = "BUY"
                elif prediction == 0 and confidence > 0.55:
                    result["signal"] = "SELL"
                else:
                    result["signal"] = "HOLD"
                    result["reason"] = "Low Confidence"
            else:
                result["signal"] = "HOLD"
                result["reason"] = "Model Load Failure"
                
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            result["reason"] = f"Error: {str(e)}"

        return result

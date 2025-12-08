import joblib
import pandas as pd
import numpy as np
import logging
from collections import deque
import apex_core.feature_builder as fb

logger = logging.getLogger(__name__)

class MLStrategyAdapter:
    def __init__(self, symbol, model_path, lookback_window=48):
        self.symbol = symbol
        self.model_path = model_path
        self.lookback_window = lookback_window
        
        # Memory buffer for prices
        self.prices = deque(maxlen=lookback_window + 50) 

        self.model = self._load_model()

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return None

    def analyze(self, current_price):
        """
        Returns a dictionary with the signal and the 'Why'.
        """
        self.prices.append(current_price)

        # Default 'Not Ready' state
        result = {
            "signal": "HOLD",
            "rsi": 0.0,
            "bb_pos": 0.0,
            "vol_z": 0.0,
            "confidence": 0.0,
            "reason": "Warming Up"
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
            latest = df_features.iloc[[-1]]
            
            # Extract Metrics for the Dashboard
            # .item() converts numpy value to standard python float
            result["rsi"] = round(latest["rsi_14"].item(), 1)
            result["bb_pos"] = round(latest["bb_pos"].item(), 2)
            result["vol_z"] = round(latest["vol_z"].item(), 2)
            result["reason"] = "Inference Complete"

            # Prepare Data for Model (Drop non-feature columns)
            inference_data = latest.select_dtypes(include=[np.number]).drop(columns=["close", "target"], errors="ignore")
            
            # --- Prediction ---
            if self.model:
                prediction = self.model.predict(inference_data)[0]
                probs = self.model.predict_proba(inference_data)[0] 
                confidence = probs[prediction]
                
                result["confidence"] = round(confidence * 100, 1)

                # Decision Logic
                if prediction == 1 and confidence > 0.55:
                    result["signal"] = "BUY"
                elif prediction == 0 and confidence > 0.55:
                    result["signal"] = "SELL"
                else:
                    result["signal"] = "HOLD"
                    result["reason"] = "Low Confidence"
                
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            result["reason"] = f"Error: {str(e)}"

        return result
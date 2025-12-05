import logging
import joblib
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime, timezone

# --- IMPORTS ---
# 1. Feature Builder is a MODULE, not a class. We import it as 'fb'
import apex_core.feature_builder as fb 

# 2. Governance is a CLASS. We import it directly.
# (Ensure 'GovernanceGate' matches the actual class name in apex_core/governance_apply.py)
from apex_core.governance_apply import GovernanceGate 

logger = logging.getLogger(__name__)

class MLStrategyAdapter:
    """
    Bridges the simple 'LiveEngine' loop with the complex 'Apex Cortex' ML pipeline.
    Handles State (History), Feature Engineering (fb), and Inference (Model).
    """
    def __init__(self, symbol: str, model_path: str, lookback_window: int = 168): 
        self.symbol = symbol.upper()
        self.lookback_window = lookback_window
        
        # 1. Load the Model Artifact (Option 2: Direct Joblib Load)
        logger.info(f"🧠 Loading ML Brain from: {model_path}")
        try:
            self.model = joblib.load(model_path)
            logger.info("✅ Model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e

        # 2. Initialize Governance
        self.governance = GovernanceGate()
        
        # 3. Memory (Rolling Buffer)
        # Stores dicts: {'ts': datetime, 'close': float}
        self.history = deque(maxlen=lookback_window)

    def analyze(self, current_price: float) -> str:
        """
        Called every tick by LiveEngine. Returns 'BUY', 'SELL', or 'HOLD'.
        """
        # A. Capture Data
        now = datetime.now(timezone.utc)
        self.history.append({"ts": now, "close": current_price})
        
        # B. Warm-up Check 
        # Feature builder needs enough rows for Volatility/ATR (min ~24-48)
        if len(self.history) < 48: 
            return "HOLD"
            
        try:
            # C. Structure Data for Feature Builder
            # It expects a dictionary: { "BTC": pd.DataFrame }
            df = pd.DataFrame(list(self.history))
            df = df.set_index("ts").sort_index()
            
            prices_map = {self.symbol: df}

            # D. Run Apex Core Feature Logic
            # This calls the function we reviewed earlier
            features_map = fb.build_features(prices_map)
            
            # Extract our symbol's features
            if self.symbol not in features_map:
                logger.warning(f"Feature builder returned no data for {self.symbol}")
                return "HOLD"
                
            df_features = features_map[self.symbol]
            
            # Extract just the CURRENT row (the one we need to predict on)
            current_row = df_features.iloc[[-1]]
            
            # E. Governance Check (The Guardrails)
            is_safe, reason = self.governance.check_safety(current_row)
            if not is_safe:
                logger.warning(f"🛡️ Governance Block: {reason}")
                return "HOLD"
            
            # F. Inference
            # FIX: Drop the 'ts' column (and any non-numeric data)
            # The model only wants numbers.
            inference_row = current_row.select_dtypes(include=[np.number])
            
            # Debug Log: Check shape to ensure it matches model (Expected: 12 or 13)
            # logger.info(f"Inference Columns: {inference_row.shape[1]}")

            prob = self.model.predict_proba(inference_row)
            
            # Handle different return shapes from sklearn (some return [p_0, p_1], some just p_1)
            if isinstance(prob, np.ndarray) and prob.ndim > 1:
                # Take the probability of Class 1 (Buy/Long)
                signal_strength = prob[0][1] 
            else:
                signal_strength = prob[0]

            # G. Signal Logic
            # TODO: Move these thresholds to a config file
            BUY_THRESHOLD = 0.60
            SELL_THRESHOLD = 0.40
            
            if signal_strength > BUY_THRESHOLD:
                return "BUY"
            elif signal_strength < SELL_THRESHOLD:
                return "SELL"
            
            return "HOLD"

        except Exception as e:
            logger.error(f"⚠️ ML Inference Failed: {e}")
            return "HOLD" # Fail safe
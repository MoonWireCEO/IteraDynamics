# signal_engine/ml/infer.py
from __future__ import annotations
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# --- Bridge to Moonwire ---
try:
    current = Path(__file__).resolve()
    root_dir = current.parents[3] 
    if not (root_dir / "moonwire").exists():
        root_dir = Path(os.getcwd())

    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
    
    from moonwire.strategies.ml_adapter import MLStrategyAdapter
except ImportError:
    MLStrategyAdapter = None

try: from src.cache_instance import cache
except ImportError: cache = None

logger = logging.getLogger(__name__)

def vectorize_features(features, feature_order):
    return np.array([[float(features.get(k, 0.0) or 0.0) for k in feature_order]], dtype=float)

class InferenceEngine:
    def __init__(self, model, feature_order, metadata=None):
        self.model = model
        self.feature_order = feature_order
        self.metadata = metadata or {}
    def predict_proba(self, features, explain=False, top_n=5):
        return {"probability": 0.5}

def infer_asset_signal(symbol: str, strategy: str = None) -> Dict[str, Any]:
    if MLStrategyAdapter is None:
        return {"ok": False, "reason": "Moonwire Adapter not found"}

    try:
        # 1. Get Data
        current_price = 0.0
        df = None
        
        if cache:
            s_data = cache.get_signal(symbol)
            if isinstance(s_data, dict):
                current_price = float(s_data.get('price') or s_data.get('close') or s_data.get('last') or 0.0)

        # Fallback to file if price is 0
        if current_price == 0.0:
            fr_path = root_dir / "flight_recorder.csv"
            if fr_path.exists():
                import pandas as pd
                df = pd.read_csv(fr_path)
                current_price = float(df['Price'].iloc[-1])

        if current_price == 0.0:
            return {"ok": False, "reason": "No price data available"}

        # 2. Execute Moonwire
        models_path = root_dir / "moonwire" / "models"
        adapter = MLStrategyAdapter(symbol, models_dir=models_path)
        
        # --- AMNESIA FIX: PRELOAD HISTORY ---
        if df is not None and not df.empty:
            history = df['Price'].tail(100).tolist()
            if len(history) > 1:
                adapter.prices.extend(history[:-1])
        # ------------------------------------

        result = adapter.analyze(current_price, strategy=strategy)

        # 3. Translate
        direction_map = {"BUY": "long", "SELL": "short", "HOLD": None}
        direction = direction_map.get(result.get("signal"), None)
        conf_percent = result.get("confidence", 0.0)
        confidence = conf_percent / 100.0

        return {
            "ok": True,
            "direction": direction,
            "confidence": confidence,
            "reason": result.get("reason", "ok"),
            "metadata": result
        }

    except Exception as e:
        return {"ok": False, "reason": f"Exception: {e}"}

__all__ = ['InferenceEngine', 'infer_asset_signal', 'vectorize_features']

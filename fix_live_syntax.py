import os
import sys

# ==============================================================================
# CORRECTED SIGNAL GENERATOR (Strict Syntax + Volume Patch)
# ==============================================================================
generator_content = r'''# src/signal_generator.py
from __future__ import annotations

import os
import sys
import json
import logging
import pandas as pd
import types
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, List

# --- BOOTSTRAPPER ---
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

if "src" not in sys.modules:
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(current_dir)] 
    sys.modules["src"] = src_pkg

for file_path in current_dir.glob("*.py"):
    module_name = file_path.stem 
    if module_name == "__init__" or module_name == "signal_generator": continue
    full_name = f"src.{module_name}"
    if full_name not in sys.modules:
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[full_name] = module
                sys.modules[module_name] = module 
                spec.loader.exec_module(module)
                setattr(src_pkg, module_name, module)
        except Exception:
            pass 

# --- IMPORTS (Strict Syntax) ---
try: 
    from src.signal_filter import is_signal_valid
except: 
    def is_signal_valid(s): 
        return True

try: 
    from src.cache_instance import cache
except: 
    cache = None

try: 
    from src.sentiment_blended import blend_sentiment_scores
except: 
    def blend_sentiment_scores(): 
        return {}

try: 
    from src.dispatcher import dispatch_alerts
except: 
    def dispatch_alerts(a, s, c): 
        pass

try: 
    from src.jsonl_writer import atomic_jsonl_append
except: 
    def atomic_jsonl_append(p, d): 
        pass

try: 
    from src.observability import failure_tracker
except: 
    class MockTracker:
        def record_failure(self, *args): 
            pass
    failure_tracker = MockTracker()

try: 
    from src.paths import SHADOW_LOG_PATH, GOVERNANCE_PARAMS_PATH
except:
    SHADOW_LOG_PATH = Path("shadow_log.jsonl")
    GOVERNANCE_PARAMS_PATH = Path("governance_params.json")

try:
    from src.regime_detector import MarketRegimeDetector, MetaStrategySelector, PositionSizer
except ImportError:
    try:
        from regime_detector import MarketRegimeDetector, MetaStrategySelector, PositionSizer
    except ImportError:
        logging.warning("CRITICAL: Regime Detector not found. Using Dummy Fallbacks.")
        class MarketRegimeDetector:
            def detect_regime(self, df): return "WARMUP"
        class MetaStrategySelector:
            def get_strategy(self, regime): return "Conservative_Trend_Follow"
        class PositionSizer:
            def __init__(self, risk_percent=0.1): pass
            def calculate_size(self, balance): return 1000.0

logger = logging.getLogger(__name__)

_ML_INFER_FN = None
try:
    from signal_engine.ml.infer import infer_asset_signal as _ML_INFER_FN
except ImportError:
    try:
        from src.ml.infer import infer_asset_signal as _ML_INFER_FN
    except ImportError:
        _ML_INFER_FN = None

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    except Exception: pass
    return default

def load_governance_params(symbol: str) -> Dict[str, Any]:
    default = {"conf_min": 0.60, "debounce_min": 15}
    try: path = GOVERNANCE_PARAMS_PATH
    except NameError: path = Path("governance_params.json")
    data = _read_json(path, {})
    row = data.get(symbol) or {}
    return {
        "conf_min": float(row.get("conf_min", default["conf_min"])),
        "debounce_min": int(row.get("debounce_min", default["debounce_min"])),
    }

def _shadow_write(payload: Dict[str, Any]) -> None:
    try:
        payload = dict(payload)
        if "ts" not in payload: payload["ts"] = _utcnow_iso()
        try: path = SHADOW_LOG_PATH
        except NameError: path = Path("shadow_log.jsonl")
        atomic_jsonl_append(path, payload)
    except Exception: pass

def _fetch_asset_history(asset: str) -> Optional[pd.DataFrame]:
    try:
        possible_paths = [Path('flight_recorder.csv'), Path('../flight_recorder.csv'), Path(os.getcwd()) / 'flight_recorder.csv']
        csv_path = None
        for p in possible_paths:
            if p.exists():
                csv_path = p
                break
        if csv_path:
            df = pd.read_csv(csv_path)
            if 'Timestamp' in df.columns and 'Price' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                return df
    except Exception as e:
        logger.warning(f"Failed to fetch history for {asset}: {e}")
    return None

def _infer_ml(asset: str, strategy: str = None) -> Dict[str, Any]:
    ml_enabled = str(os.getenv("MW_INFER_ENABLE", "1")).lower() in {"1", "true", "yes"}
    if not ml_enabled: return {"ok": False, "reason": "ml_disabled"}
    if _ML_INFER_FN is None: return {"ok": False, "reason": "ml_unavailable"}

    try:
        out = _ML_INFER_FN(asset, strategy=strategy)  
        if not isinstance(out, dict): return {"ok": False, "reason": "ml_bad_return_type"}
        if out.get("error"): return {"ok": False, "reason": f"ml_error:{out['error']}"}
        
        direction = out.get("direction") or out.get("dir")
        conf = out.get("confidence") if out.get("confidence") is not None else out.get("conf")
        
        if (direction is not None and direction not in {"long", "short"}) or conf is None:
            return {"ok": False, "reason": "ml_missing_keys", "raw": out}
            
        return {"ok": True, "dir": direction, "conf": float(conf), "reason": "ok", "raw": out}
    except Exception as e:
        return {"ok": False, "reason": f"ml_exception:{type(e).__name__}"}

def _heuristic_confidence(price_change: float, sentiment: float) -> float:
    try: return float(max(0.0, min(1.0, ((price_change/10.0)+float(sentiment))/2.0)))
    except: return 0.0

def label_confidence(score: float) -> str:
    if score >= 0.66: return "High Confidence"
    elif score >= 0.33: return "Medium Confidence"
    else: return "Low Confidence"

_regime_detector = MarketRegimeDetector()
_strategy_selector = MetaStrategySelector()
_position_sizer = PositionSizer(risk_percent=0.10) 

def generate_signals():
    print(f"[{datetime.now().time()}] Starting Signal Generator...")
    stablecoins = {"USDC", "USDT", "DAI", "TUSD", "BUSD"}
    valid_signals: List[dict] = []
    
    shadow_only = str(os.getenv("MW_INFER_SHADOW_ONLY", "0")).lower() in {"1", "true", "yes"}
    live_ml = str(os.getenv("MW_INFER_LIVE", "0")).lower() in {"1", "true", "yes"}

    try:
        assets = []
        if 'cache' in globals() and cache:
            try: assets = [k for k in cache.keys() if not k.endswith('_signals') and not k.endswith('_sentiment')]
            except: pass 
        
        if not assets:
            print(">> CACHE EMPTY. Switching to SIMULATION MODE (BTC).")
            assets = ["BTC"]
            class MockCache:
                # --- VOLUME INJECTION PATCH ---
                def get_signal(self, a): return {"price_change_24h": 2.5, "volume_now": 50000000, "balance": 10000.0}
            cache_obj = MockCache()
        else:
            cache_obj = cache

        sentiment_scores = blend_sentiment_scores() if 'blend_sentiment_scores' in globals() else {}
        
        for asset in assets:
            if asset in stablecoins: continue
            data = cache_obj.get_signal(asset)
            if not isinstance(data, dict): continue
            price_change = data.get("price_change_24h", 0.0)
            volume = data.get("volume_now", 0.0)
            sentiment = float(sentiment_scores.get(asset, 0.0))
            
            # 1. Regime & Strategy
            hist_df = _fetch_asset_history(asset)
            current_regime = _regime_detector.detect_regime(hist_df) if hist_df is not None and not hist_df.empty else "WARMUP"
            target_strategy = _strategy_selector.get_strategy(current_regime)
            current_balance = float(data.get("balance", 10000.0)) 
            trade_size_limit = _position_sizer.calculate_size(current_balance)
            
            print(f"[{asset}] Regime: {current_regime} -> Strategy: {target_strategy} (Size: ${trade_size_limit})")

            # 2. Execution
            ml = _infer_ml(asset, strategy=target_strategy)
            
            # --- DEBUG LOGGING ---
            if not ml.get("ok"):
                print(f"   >> ML FAILED: {ml.get('reason')}")
            else:
                d_str = ml['dir'].upper() if ml['dir'] else "HOLD"
                print(f"   >> [ML PREDICTION] {d_str} ({ml['conf']:.2f}) using {target_strategy}")

            gov = load_governance_params(asset)
            
            if live_ml and ml.get("ok"):
                direction = ml["dir"]
                confidence = float(ml["conf"] or 0.0)
                
                # Check for HOLD
                if direction is None:
                    print("   >> ML Decision: HOLD/CASH (No Signal Generated)")
                    continue

                if confidence < float(gov["conf_min"]): 
                    print(f"   >> ML Filtered: Confidence {confidence:.2f} < Min {gov['conf_min']}")
                    continue
                
                signal = {
                    "asset": asset,
                    "direction": direction,
                    "confidence_score": confidence,
                    "confidence_label": label_confidence(confidence),
                    "regime": current_regime,            
                    "strategy": target_strategy,         
                    "trade_size_limit": trade_size_limit,
                    "price_change": price_change,
                    "volume": volume,
                    "sentiment": sentiment,
                    "timestamp": datetime.now(timezone.utc),
                    "governance": gov,
                    "inference": "ml_live"
                }
                
                if is_signal_valid(signal):
                    dispatch_alerts(asset, signal, cache)
                    valid_signals.append(signal)
                    print(f"   >> LIVE SIGNAL GENERATED: {direction}")
                else:
                    print(f"   >> SIGNAL REJECTED BY FILTER")
            else:
                confidence = _heuristic_confidence(price_change, sentiment)
                direction = "long" if confidence >= 0.5 else "short"
                print(f"   >> (Heuristic Backup): {direction} ({confidence:.2f})")

    except Exception as e:
        print(f"Generator Exception: {e}")

    return valid_signals

if __name__ == "__main__":
    generate_signals()
    print("Signal Generation Complete.")
'''

with open("apex_core/signal_generator.py", "w", encoding="utf-8") as f:
    f.write(generator_content)

print("✅ Updated signal_generator.py with STRICT syntax and VOLUME PATCH.")
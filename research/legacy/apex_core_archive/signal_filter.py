# src/signal_filter.py

from datetime import datetime

# Store recent alert timestamps per asset
_recent_alerts = {}

def is_signal_valid(signal):
    asset = signal['asset']
    movement = float(signal['price_change']) # Ensure float
    volume = float(signal['volume'])
    now = signal['timestamp']

    # --- TUNING (Dec 10) ---
    # OLD: MIN_MOVEMENT = 7.0 (Requires 7% pump!)
    # NEW: MIN_MOVEMENT = 0.5 (Just requires the asset is alive/moving)
    MIN_MOVEMENT = 0.5 
    
    # Keep volume filter (10M is reasonable for BTC)
    MIN_VOLUME = 10_000_000 
    
    # Prevent spamming signals every minute
    COOLDOWN_MINUTES = 15 

    # 1. Volume Check (Avoid illiquid junk)
    if volume < MIN_VOLUME:
        return False

    # 2. Movement Check (Avoid dead/flat assets)
    # Changed to ABSOLUTE value so we can trade dips (-2%) or pumps (+2%)
    if abs(movement) < MIN_MOVEMENT:
        return False

    # 3. Cooldown Check
    if asset in _recent_alerts:
        last_time, last_movement = _recent_alerts[asset]
        elapsed = (now - last_time).total_seconds() / 60
        
        # If too soon AND price hasn't moved much since last signal, ignore.
        # (Allows re-entry if price moves significantly, e.g. > 1%)
        if elapsed < COOLDOWN_MINUTES and abs(movement - last_movement) < 1.0:
            return False

    # Log this alert
    _recent_alerts[asset] = (now, movement)
    return True
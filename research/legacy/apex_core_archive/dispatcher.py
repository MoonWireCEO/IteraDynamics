# src/dispatcher.py

import logging

# --- SAFE IMPORTS ---
try:
    from src.cache import SignalCache
except ImportError:
    # Dummy class to prevent crashes if import fails
    class SignalCache:
        def set_signal(self, k, v): pass
        def get_signal(self, k): return None

try:
    from src.emailer import send_email_alert
except ImportError:
    # Dummy function
    def send_email_alert(s, b): pass

logger = logging.getLogger(__name__)

def dispatch_alerts(asset: str, signal: dict, cache: SignalCache):
    logger.info(f"[Dispatch] Alert triggered for {asset}: {signal}")

    # 1. CACHE WRITE
    if cache:
        try:
            # Save signal to cache
            cache.set_signal(asset, signal)

            # Also save to history (as a separate entry)
            history_key = f"{asset}_history"
            # Ensure values are serializable (timestamps can sometimes break json dumps)
            history_entry = {
                "price_change": signal["price_change"],
                "volume": signal["volume"],
                "sentiment": signal["sentiment"],
                "confidence_score": signal["confidence_score"],
                "confidence_label": signal.get("confidence_label", "Unknown"),
                "timestamp": str(signal["timestamp"]) 
            }
            cache.set_signal(history_key, history_entry)
            logger.info(f"[History Write] Key: {history_key}")
        except Exception as e:
            logger.error(f"[Dispatch] Cache write failed: {e}")

    # 2. EMAIL ALERT (FAIL GRACEFULLY)
    try:
        label = signal.get("confidence_label", "Unknown Confidence")
        subject = f"MoonWire Alert: {asset} ({label})"
        body = (
            f"TEST ALERT:\n\n"
            f"Price moved {signal['price_change']}%\n"
            f"Volume: {signal['volume']}\n"
            f"Sentiment Score: {signal['sentiment']:.2f}\n"
            f"Confidence Score: {signal['confidence_score']:.2f} ({label})\n"
            f"Time: {signal['timestamp']} UTC\n"
        )

        # Attempt to send, but don't crash if keys are missing
        send_email_alert(subject, body)
        
    except Exception as e:
        # Log it as a warning, NOT a crash
        logger.warning(f"[Dispatch] Email skipped (Configuration missing): {e}")
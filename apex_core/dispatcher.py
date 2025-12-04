# src/dispatcher.py

import logging
from src.cache import SignalCache
from src.emailer import send_email_alert

logger = logging.getLogger(__name__)

def dispatch_alerts(asset: str, signal: dict, cache: SignalCache):
    logger.info(f"[Dispatch] Alert triggered for {asset}: {signal}")

    # Save signal to cache
    cache.set_signal(asset, signal)

    # Also save to history (as a separate entry)
    history_key = f"{asset}_history"
    history_entry = {
        "price_change": signal["price_change"],
        "volume": signal["volume"],
        "sentiment": signal["sentiment"],
        "confidence_score": signal["confidence_score"],
        "confidence_label": signal.get("confidence_label", "Unknown"),
        "timestamp": signal["timestamp"]
    }
    cache.set_signal(history_key, history_entry)
    logger.info(f"[History Write] Key: {history_key} -> {cache.get_signal(history_key)}")

    # Format and send email alert
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

    send_email_alert(subject, body)
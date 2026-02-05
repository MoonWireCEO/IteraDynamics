from datetime import datetime

def log(message):
    timestamp = datetime.utcnow().isoformat()
    print(f"[{timestamp}] {message}")

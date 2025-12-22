# run_live.py
# 🦅 ARGUS LIVE SCHEDULER - V2.1

import time, schedule, sys, os
from datetime import datetime
from apex_core.signal_generator import generate_signals

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("argus.log", "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message); self.log.write(message); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()

sys.stdout = Logger()
sys.stderr = sys.stdout

def job():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 EXECUTION WINDOW OPEN...")
    try: generate_signals()
    except Exception as e: print(f"❌ SCHEDULER ERROR: {e}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 💤 Cycle complete. Sleeping...")

schedule.every().hour.at(":00").do(job)

if __name__ == "__main__":
    print("🦅 ARGUS LIVE SCHEDULER ONLINE")
    print(f"⏰ System Time: {datetime.now().strftime('%H:%M:%S')}")
    while True:
        schedule.run_pending()
        time.sleep(1)
# run_live.py
# 🦅 ARGUS HOURLY SCHEDULER - V3.0 (LOGGING ENABLED)

import schedule
import time
import subprocess
import sys
import os
from datetime import datetime
import logging
from pathlib import Path

# --- GLOBAL LOGGING SETUP ---
LOG_FILE = Path("data/argus_execution.log")

# Ensure the 'data' directory exists
LOG_FILE.parent.mkdir(exist_ok=True)

# Configure the global logger (Writes to file AND console)
# We use a simple format for the log file itself.
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def job():
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 EXECUTION WINDOW OPEN: Waking Argus...")
    
    # 🔧 FORCE UTF-8 ENCODING FOR THE SUBPROCESS
    my_env = os.environ.copy()
    my_env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [sys.executable, "apex_core/signal_generator.py"], 
            capture_output=True, 
            text=True,
            encoding='utf-8', 
            env=my_env 
        )

        # Log Standard Output (Logs from the bot)
        if result.stdout:
            # Clean up raw output line-by-line before logging
            for line in result.stdout.strip().split('\n'):
                # We log the raw output directly, which contains the timestamp/message we want
                logger.info(line.strip())

        # Log Errors (if any)
        if result.stderr:
            logger.error(f"   [STDERR] {result.stderr}")
            
    except Exception as e:
        logger.error(f"   ❌ CRITICAL SCHEDULER ERROR: {e}")

    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 💤 Argus cycle complete. Sleeping...")

def get_market_price():
    # Retained placeholder function
    return "Active"

# --- SCHEDULING ---
schedule.every().hour.at(":00").do(job)

logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === 🚀 Itera Dynamics: LIVE HOURLY SCHEDULER ===")
logger.info(f"[ SYSTEM ] Encoding Forced: UTF-8")
logger.info(f"           Mode: Hourly Swing (Execution at XX:00)")

while True:
    next_run = schedule.next_run()
    time_left = next_run - datetime.now()
    minutes = int(time_left.total_seconds() // 60)
    
    # Use print for the simple, one-line status update (it's fast and doesn't spam the log file)
    print(f"\r[{datetime.now().strftime('%H:%M')}] ⏳ Next Signal in {minutes} min...   ", end="")
    
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        logger.warning("\n🛑 SCHEDULER STOPPED BY USER.")
        break
    
    schedule.run_pending()
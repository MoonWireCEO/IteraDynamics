# run_live.py
# 🦅 ARGUS LIVE SCHEDULER - V2.3 (SYSTEMD-HARDENED + TZ-AWARE UTC)

import os
import sys
import time
import schedule
import signal
import traceback
from pathlib import Path
from datetime import datetime, timezone

from apex_core.signal_generator import generate_signals

PROJECT_ROOT = Path(__file__).resolve().parent
LOGFILE = PROJECT_ROOT / "argus.log"
CYCLE_LOCK = PROJECT_ROOT / "argus_cycle.lock"

_SHUTDOWN = False


class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(LOGFILE, "a", encoding="utf-8", buffering=1)

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        try:
            os.fsync(self.log.fileno())
        except Exception:
            # Do not crash scheduler due to fsync issues.
            pass


sys.stdout = Logger()
sys.stderr = sys.stdout


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _handle_shutdown(signum, frame):
    global _SHUTDOWN
    print(f"\n[{_utc_ts()}] ⚠️ RECEIVED SIGNAL {signum} — GRACEFUL SHUTDOWN REQUESTED")
    _SHUTDOWN = True


signal.signal(signal.SIGINT, _handle_shutdown)
signal.signal(signal.SIGTERM, _handle_shutdown)


def _acquire_cycle_lock() -> bool:
    """
    Atomic lock acquisition to prevent overlapping cycles or double-running processes.
    Returns True if lock acquired, False if lock already exists.
    """
    try:
        fd = os.open(str(CYCLE_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n{_utc_ts()}\n")
        return True
    except FileExistsError:
        return False


def _release_cycle_lock():
    try:
        if CYCLE_LOCK.exists():
            CYCLE_LOCK.unlink()
    except Exception as e:
        print(f"[{_utc_ts()}] ❌ FAILED TO RELEASE CYCLE LOCK: {e}")


def job():
    if _SHUTDOWN:
        return

    print(f"\n[{_utc_ts()}] 🚀 EXECUTION WINDOW OPEN...")

    if not _acquire_cycle_lock():
        print(f"[{_utc_ts()}] ❌ CYCLE LOCK PRESENT — PREVIOUS CYCLE MAY STILL BE RUNNING. SKIPPING.")
        return

    try:
        generate_signals()
    except Exception:
        print(f"[{_utc_ts()}] ❌ UNCAUGHT ERROR IN EXECUTION CYCLE:")
        traceback.print_exc()
    finally:
        _release_cycle_lock()

    print(f"[{_utc_ts()}] 💤 Cycle complete. Sleeping...")


schedule.every().hour.at(":00").do(job)

if __name__ == "__main__":
    print("🦅 ARGUS LIVE SCHEDULER ONLINE")
    print(f"⏰ UTC Time: {_utc_ts()}")

    while not _SHUTDOWN:
        schedule.run_pending()
        time.sleep(0.5)

    print(f"[{_utc_ts()}] 🛑 ARGUS SCHEDULER SHUTDOWN COMPLETE")

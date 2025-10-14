"""
Grace Watchdog
Global exception catcher + auto-restart. Emits heartbeat to /monitoring/ and logs crashes.
"""

import time
import logging
import subprocess
import os
from datetime import datetime

HEARTBEAT_FILE = "monitoring/heartbeat.log"
RUNNER_CMD = ["python3", "grace_core_runner.py"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grace_watchdog")


def emit_heartbeat():
    with open(HEARTBEAT_FILE, "a") as f:
        f.write(f"{datetime.now().isoformat()} - alive\n")


def run_with_watchdog():
    while True:
        try:
            logger.info("Starting Grace Core Runner...")
            proc = subprocess.Popen(RUNNER_CMD)
            while proc.poll() is None:
                emit_heartbeat()
                time.sleep(10)
            exit_code = proc.returncode
            logger.error(f"Grace Core Runner exited with code {exit_code}")
            emit_heartbeat()
            time.sleep(5)
            logger.info("Restarting Grace Core Runner...")
        except Exception as e:
            logger.exception(f"Watchdog caught exception: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_with_watchdog()

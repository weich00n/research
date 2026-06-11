"""Logging setup (VacSim: utils/logging_utils.py, extended to a real file logger).

One shared logger tree ("fark"): the console shows progress at INFO, while the
log file also captures DEBUG detail — every LLM call with latency, retries,
429 backoffs, and raw responses that failed JSON parsing.
"""

import inspect
import logging
import os
import sys
import time
from contextlib import contextmanager

LOGGER_NAME = "fark"


def get_logger(child=None):
    """Module-level logger, e.g. get_logger("llm") -> "fark.llm".
    Children propagate to the handlers configured by setup_logger."""
    return logging.getLogger(f"{LOGGER_NAME}.{child}" if child else LOGGER_NAME)


def setup_logger(log_path=None, console_level=logging.INFO):
    """Configure the shared logger: plain messages to the console, timestamped
    DEBUG-level records to `log_path`. Safe to call again (handlers are reset)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
        logger.info(f"Logging to {os.path.abspath(log_path)}")

    return logger


@contextmanager
def log_info():
    """VacSim's timing helper: logs start/end/elapsed of the calling method."""
    frame = inspect.currentframe().f_back.f_back  # skip contextmanager wrapper
    method_name = frame.f_code.co_name if frame else "Unknown"
    logger = get_logger()
    start_time = time.time()
    logger.info(f"Running {method_name} "
                f"(start {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))})")
    try:
        yield
    finally:
        logger.info(f"Finished {method_name} (elapsed {time.time() - start_time:.2f}s)")

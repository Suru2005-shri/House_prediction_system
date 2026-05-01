"""
logger.py
---------
Centralised logging setup. Every module imports `get_logger(__name__)`.
Logs go to both the terminal (coloured) and logs/pipeline.log (plain text).
"""

import logging
import os
from datetime import datetime

_INITIALIZED = False
LOG_DIR = "logs"


class _ColourFormatter(logging.Formatter):
    """Add ANSI colour codes to terminal output by log level."""
    GREY    = "\x1b[38;5;240m"
    CYAN    = "\x1b[36m"
    YELLOW  = "\x1b[33m"
    RED     = "\x1b[31m"
    BOLD_RED= "\x1b[31;1m"
    RESET   = "\x1b[0m"

    FORMATS = {
        logging.DEBUG   : GREY     + "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s" + RESET,
        logging.INFO    : CYAN     + "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s" + RESET,
        logging.WARNING : YELLOW   + "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s" + RESET,
        logging.ERROR   : RED      + "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s" + RESET,
    }

    def format(self, record):
        fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def setup_logging(level: int = logging.INFO) -> None:
    """Call once from main.py to configure root logger."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")

    root = logging.getLogger()
    root.setLevel(level)

    # Terminal handler (coloured)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(_ColourFormatter())
    root.addHandler(ch)

    # File handler (plain)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(fh)

    _INITIALIZED = True
    logging.getLogger(__name__).info(f"Logging initialised → {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call setup_logging() once before using."""
    return logging.getLogger(name)

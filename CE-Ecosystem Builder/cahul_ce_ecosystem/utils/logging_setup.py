from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_file: str | Path, console_level: int = logging.ERROR) -> logging.Logger:
    """
    Setup logging with file output (all levels) and console output (errors only by default).
    
    Args:
        log_file: Path to log file
        console_level: Minimum level for console output (default: ERROR = only show critical issues)
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cahul_ce")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File handler - logs everything (DEBUG and above)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(file_fmt)

        # Console handler - only shows errors and critical by default
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(console_level)
        console_fmt = logging.Formatter(fmt="%(levelname)s: %(message)s")
        ch.setFormatter(console_fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

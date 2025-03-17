# src/slide/bayes/util/logger_util.py

import logging
import sys
import os
from datetime import datetime

__all__ = ["setup_logger", "get_logger"]


def setup_logger(
    log_file: str,
    level: int = logging.INFO,
    name: str = "bayes",
    stdout: bool = True,
    mode: str = "a",
):
    """
    Initialize global logger.

    Args:
        log_file: path to log file
        level: logging.INFO / DEBUG / WARNING ...
        name: root logger name
        stdout: also print to stdout
        mode: file open mode ('a' or 'w')

    Returns:
        logging.Logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 防止重复 add handler（Jupyter / 多次 import）
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ===== stdout handler =====
    if stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    # ===== file handler =====
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode=mode)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    logger.info("=" * 80)
    logger.info("Logger initialized")
    logger.info(f"log_file={log_file}")
    logger.info(f"level={logging.getLevelName(level)}")
    logger.info("=" * 80)

    return logger


def get_logger(name: str):
    """
    Get child logger.

    Example:
        logger = get_logger("bayes.bo")
    """
    return logging.getLogger(name)

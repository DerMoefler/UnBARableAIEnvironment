import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"


def get_file_logger(name: str, log_file: Path, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger
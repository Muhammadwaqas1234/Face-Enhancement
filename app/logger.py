# app/logger.py
import logging
from logging.handlers import RotatingFileHandler
import sys
from .config import settings
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "app.log")

def get_logger(name: str = "face_enhance"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s — %(levelname)s — %(name)s — %(message)s"
    )

    handler = RotatingFileHandler(LOG_PATH, maxBytes=10_000_000, backupCount=5)
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    return logger

logger = get_logger()

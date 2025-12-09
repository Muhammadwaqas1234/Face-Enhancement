# app/config.py
from pydantic_settings import BaseSettings
from pydantic import AnyUrl
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "Face Enhancement API"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Super-resolution model: provide relative path under /models or absolute path.
    SR_MODEL_PATH: Optional[str] = None
    SR_MODEL_NAME: str = "edsr"  # or 'fsrcnn', 'lapsrn', etc.
    SR_MODEL_SCALE: int = 2

    # Face detection
    HAAR_SCALE_FACTOR: float = 1.1
    HAAR_MIN_NEIGHBORS: int = 5

    # Enhancement defaults
    CLAHE_CLIP: float = 2.0
    CLAHE_TILE: tuple[int, int] = (8, 8)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instantiate settings
settings = Settings()

# app/utils.py
from typing import Optional, Tuple
import cv2
import numpy as np
from .config import settings
from .logger import logger
import aiofiles
import io

# Haar cascade classifier (OpenCV built-in)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces(img_bgr: np.ndarray, scaleFactor: float = None, minNeighbors: int = None):
    """Detect faces — returns list of (x, y, w, h)."""
    if scaleFactor is None:
        scaleFactor = settings.HAAR_SCALE_FACTOR
    if minNeighbors is None:
        minNeighbors = settings.HAAR_MIN_NEIGHBORS

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    return list(map(tuple, faces))


def detect_largest_face(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    faces = detect_faces(img_bgr)
    if not faces:
        return None
    largest = max(faces, key=lambda r: r[2] * r[3])
    logger.debug(f"Detected {len(faces)} faces, largest: {largest}")
    return largest


def crop_with_margin(img: np.ndarray, bbox: Tuple[int, int, int, int], margin: float = 0.2) -> np.ndarray:
    x, y, w, h = bbox
    H, W = img.shape[:2]
    dx = int(w * margin)
    dy = int(h * margin)
    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(W, x + w + dx)
    y2 = min(H, y + h + dy)
    return img[y1:y2, x1:x2]


def bgr_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


async def read_uploadfile_bytes(upload_file) -> bytes:
    """Async read content of FastAPI UploadFile."""
    async with aiofiles.open(upload_file.file.fileno(), mode='rb') as f:
        # Some platforms don't support reading via fileno; fallback:
        upload_file.file.seek(0)
        return upload_file.file.read()

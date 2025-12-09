# app/enhancement.py
from typing import Tuple, Optional, List
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from .config import settings
from .logger import logger
import os

# ---------------------------
# Converters
# ---------------------------
def to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def from_pil(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# ---------------------------
# Basic enhancements
# ---------------------------
def histogram_equalization_y(img: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def clahe_ycrcb(img: np.ndarray, clip_limit: float = None, grid=(8,8)) -> np.ndarray:
    if clip_limit is None:
        clip_limit = settings.CLAHE_CLIP
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    y_cl = clahe.apply(y)
    out = cv2.merge([y_cl, cr, cb])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

def clahe_lab(img: np.ndarray, clip_limit: float = None, grid=(8,8)) -> np.ndarray:
    if clip_limit is None:
        clip_limit = settings.CLAHE_CLIP
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    l_cl = clahe.apply(l)
    lab_cl = cv2.merge([l_cl, a, b])
    return cv2.cvtColor(lab_cl, cv2.COLOR_LAB2BGR)

# ---------------------------
# Gamma / brightness utils
# ---------------------------
def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255.0
    table = np.clip(table, 0, 255).astype("uint8")
    return cv2.LUT(image, table)

def normalize_brightness_y(img: np.ndarray) -> np.ndarray:
    """Normalize face brightness by equalizing Y channel (clip to avoid overexposure)."""
    return histogram_equalization_y(img)

# ---------------------------
# Retinex: SSR and MSR
# ---------------------------
def single_scale_retinex(img: np.ndarray, sigma: float = 30) -> np.ndarray:
    """SSR on each channel; returns uint8 BGR."""
    img = img.astype(np.float32) + 1.0
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0,0), sigma))
    # scale to 0-255 per channel
    for i in range(retinex.shape[2]):
        channel = retinex[:,:,i]
        channel = (channel - channel.min()) / (channel.max() - channel.min()) * 255.0
        retinex[:,:,i] = channel
    return np.clip(retinex, 0, 255).astype('uint8')

def multi_scale_retinex(img: np.ndarray, sigmas: List[float] = [15, 80, 250]) -> np.ndarray:
    img = img.astype(np.float32) + 1.0
    retinex = np.zeros_like(img)
    for s in sigmas:
        retinex += np.log10(img) - np.log10(cv2.GaussianBlur(img, (0,0), s))
    retinex = retinex / len(sigmas)
    # color restoration and scaling
    for i in range(retinex.shape[2]):
        c = retinex[:,:,i]
        c = (c - c.min()) / (c.max() - c.min()) * 255.0
        retinex[:,:,i] = c
    return np.clip(retinex, 0, 255).astype('uint8')

# ---------------------------
# Smoothing / Denoising / Sharpening
# ---------------------------
def bilateral_smooth(img: np.ndarray, d: int = 9, sigmaColor: float = 75.0, sigmaSpace: float = 75.0) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

def denoise_fastnlmeans(img: np.ndarray, h: float = 10.0, templateWindowSize: int = 7, searchWindowSize: int = 21) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize, searchWindowSize)

def sharpen(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    kernel = np.array([[-1,-1,-1],[-1,9+amount,-1],[-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

# ---------------------------
# Skin smoothing (mask-based)
# ---------------------------
def skin_mask_ycrcb(img: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    # common skin range for Cr/Cb
    mask = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    return mask

def skin_smooth(img: np.ndarray, smooth_amount: float = 0.8) -> np.ndarray:
    """
    Smooth skin areas softly while preserving non-skin (eyes, lips).
    Rough approach:
    - compute skin mask in YCrCb
    - apply bilateral smoothing and combine
    """
    mask = skin_mask_ycrcb(img)
    smooth = bilateral_smooth(img, d=9, sigmaColor=75, sigmaSpace=75)
    mask_f = mask.astype(np.float32)/255.0
    mask_f = cv2.merge([mask_f, mask_f, mask_f])
    out = (img.astype(np.float32) * (1-mask_f) + smooth.astype(np.float32)*mask_f).astype(np.uint8)
    # slight sharpening after smoothing to restore detail
    out = cv2.addWeighted(out, 1.0, sharpen(out, amount=0.2), 0.2, 0)
    return out

# ---------------------------
# HDR-like
# ---------------------------
def hdr_detail_enhance(img: np.ndarray, sigma_s: int = 10, sigma_r: float = 0.15) -> np.ndarray:
    return cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)

# ---------------------------
# Fusion (gamma + clahe)
# ---------------------------
def fusion_simple(img: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    bright = adjust_gamma(img, gamma=gamma)
    clahe = clahe_ycrcb(img)
    fused = cv2.addWeighted(bright, 0.5, clahe, 0.5, 0)
    return fused

# ---------------------------
# Super-resolution support (dnn_superres)
# ---------------------------
_sr = None
def _load_sr():
    global _sr
    if _sr is not None:
        return _sr
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = settings.SR_MODEL_PATH
        if not model_path:
            logger.info("No SR_MODEL_PATH configured; super-resolution disabled.")
            _sr = None
            return None
        if not os.path.exists(model_path):
            logger.warning(f"SR model path {model_path} not found; SR disabled.")
            _sr = None
            return None
        sr.readModel(model_path)
        sr.setModel(settings.SR_MODEL_NAME, settings.SR_MODEL_SCALE)
        _sr = sr
        logger.info("Loaded SR model from %s", model_path)
        return _sr
    except Exception as exc:
        logger.exception("Failed to initialize dnn_superres: %s", exc)
        _sr = None
        return None

def super_resolve(img: np.ndarray) -> np.ndarray:
    sr = _load_sr()
    if sr is None:
        logger.debug("Super-resolution requested but not available; returning original.")
        return img
    try:
        return sr.upsample(img)
    except Exception as exc:
        logger.exception("SR failed: %s", exc)
        return img

# ---------------------------
# Comparison grid utility
# ---------------------------
def make_comparison_grid(imgs: List[np.ndarray], labels: List[str], thumb_size=(360,360), cols=2) -> Image.Image:
    pil_imgs = [to_pil(im) for im in imgs]
    thumbs = []
    for p in pil_imgs:
        p_thumb = p.copy()
        p_thumb.thumbnail(thumb_size)
        thumbs.append(p_thumb)

    rows = (len(thumbs) + cols - 1) // cols
    w = cols * thumb_size[0]
    h = rows * (thumb_size[1] + 30)
    canvas = Image.new("RGB", (w, h), color=(30,30,30))

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    for idx, im in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        x = c * thumb_size[0] + (thumb_size[0] - im.width) // 2
        y = r * (thumb_size[1] + 30)
        canvas.paste(im, (x, y))
        label = labels[idx] if idx < len(labels) else ""
        text_w, _ = draw.textsize(label, font=font)
        draw.text((c*thumb_size[0] + (thumb_size[0]-text_w)//2, y + im.height + 5), label, font=font, fill=(255,255,255))

    return canvas

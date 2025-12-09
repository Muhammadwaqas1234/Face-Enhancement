# app/main.py
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
from PIL import Image
from typing import Optional
from .config import settings
from .logger import logger
from .utils import detect_largest_face, crop_with_margin, bgr_from_bytes
from . import enhancement
import aiofiles

app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "app_name": settings.APP_NAME})


async def _read_image_from_upload(upload_file: UploadFile) -> np.ndarray:
    contents = await upload_file.read()
    img = bgr_from_bytes(contents)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


def _pil_stream_from_image(img_bgr: np.ndarray) -> io.BytesIO:
    pil = enhancement.to_pil(img_bgr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf


@app.post("/enhance")
async def enhance_endpoint(
    file: UploadFile = File(...),
    method: str = Form("clahe_y"),
    margin: float = Form(0.25),
    sr: bool = Form(False),
    gamma: Optional[float] = Form(None),
):
    """
    Supported methods:
      - clahe_y
      - clahe_lab
      - hist_eq
      - hdr
      - fusion
      - ssr
      - msr
      - bilateral
      - denoise
      - skin_smooth
      - sharpen
      - compare
    Query form params:
      - margin: crop margin around detected face
      - sr: boolean to apply super-resolution (if model loaded)
      - gamma: optional gamma correction to apply before/after
    """
    img = await _read_image_from_upload(file)

    bbox = detect_largest_face(img)
    if bbox is not None:
        face = crop_with_margin(img, bbox, margin=margin)
    else:
        face = img.copy()

    # apply selected method
    method = method.lower()
    if method == "clahe_y":
        out = enhancement.clahe_ycrcb(face)
    elif method == "clahe_lab":
        out = enhancement.clahe_lab(face)
    elif method == "hist_eq":
        out = enhancement.histogram_equalization_y(face)
    elif method == "hdr":
        out = enhancement.hdr_detail_enhance(face)
    elif method == "fusion":
        out = enhancement.fusion_simple(face, gamma=gamma if gamma else 0.6)
    elif method == "ssr":
        out = enhancement.single_scale_retinex(face, sigma=30)
    elif method == "msr":
        out = enhancement.multi_scale_retinex(face)
    elif method == "bilateral":
        out = enhancement.bilateral_smooth(face)
    elif method == "denoise":
        out = enhancement.denoise_fastnlmeans(face)
    elif method == "skin_smooth":
        out = enhancement.skin_smooth(face)
    elif method == "sharpen":
        out = enhancement.sharpen(face)
    elif method == "compare":
        imgs = [
            face,
            enhancement.histogram_equalization_y(face),
            enhancement.clahe_ycrcb(face),
            enhancement.fusion_simple(face),
            enhancement.skin_smooth(face),
        ]
        labels = ["Original", "Hist EQ (Y)", "CLAHE (YCrCb)", "Fusion", "SkinSmooth"]
        grid = enhancement.make_comparison_grid(imgs, labels, thumb_size=(360,360), cols=2)
        buf = io.BytesIO()
        grid.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported method {method}")

    # optional gamma correction
    if gamma is not None and method not in ("fusion",):
        try:
            out = enhancement.adjust_gamma(out, float(gamma))
        except Exception as exc:
            logger.exception("Gamma adjustment failed: %s", exc)

    # optional super-resolution
    if sr:
        out = enhancement.super_resolve(out)

    buf = _pil_stream_from_image(out)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "app": settings.APP_NAME})

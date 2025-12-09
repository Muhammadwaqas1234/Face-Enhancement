# Face Enhancement API

A FastAPI application for **enhancing dark, low-contrast, or low-resolution images**, with optional super-resolution. This project uses **CLAHE, histogram equalization, Retinex, HDR-like, and fusion-based techniques** to improve facial details and overall image quality.

---

## Features

- Low-light enhancement with **CLAHE** (YCrCb / Lab)
- Histogram equalization
- HDR-like enhancement (detailEnhance)
- Fusion-based enhancement (Gamma + CLAHE)
- Retinex methods: SSR and MSR
- Skin smoothing, denoising, sharpening
- Optional **super-resolution** (EDSR / FSRCNN / LapSRN)
- FastAPI endpoints for easy integration
- Web interface for uploading and previewing images

---

## Requirements

- Python 3.10+
- FastAPI
- Uvicorn
- OpenCV (`opencv-python`)
- NumPy
- Jinja2
- Optional: Super-resolution models (`dnn_superres` from OpenCV)

Install dependencies:

```bash
pip install fastapi uvicorn opencv-python numpy jinja2
````

Optional:

```bash
pip install opencv-contrib-python
```

---

## Project Structure

```
project_root/
├─ app/
│  ├─ main.py           # FastAPI application
│  ├─ config.py         # Settings & environment configuration
│  ├─ enhance.py        # Image enhancement functions
├─ models/              # Optional super-resolution models
├─ templates/
│  └─ index.html        # Web interface
├─ static/
│  └─ style.css         # Professional CSS for UI
└─ requirements.txt
```

---

## Usage

### Run the API

```bash
uvicorn app.main:app --reload
```

Open your browser:

```
http://127.0.0.1:8000/
```

### API Endpoints

* `GET /` - Web interface
* `POST /enhance` - Upload image and apply enhancement

  * Form data:

    * `file`: image file
    * `method`: enhancement method (CLAHE, histogram, HDR, fusion, SSR, MSR, etc.)
    * `margin`: crop margin (optional)
    * `gamma`: gamma adjustment (optional)
    * `sr`: super-resolution checkbox

Returns the enhanced image as a response.

---

## Adding Super-Resolution Models

1. Download OpenCV DNN super-resolution models (EDSR, FSRCNN, LapSRN).
2. Place them in the `models/` folder.
3. Set the path in `config.py` or `.env`:

```python
SR_MODEL_PATH = "models/edsr_x2.pb"
SR_MODEL_SCALE = 2
SR_MODEL_NAME = "edsr"
```

---

## Notes

* Default settings are in `app/config.py`. You can optionally use a `.env` file to override.
* Designed for local or internal use; for production, consider using HTTPS and authentication.
* Works best with **faces or portraits**, but can enhance general images as well.



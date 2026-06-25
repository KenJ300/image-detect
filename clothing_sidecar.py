#!/usr/bin/env python3
"""
clothing_sidecar.py - tiny single-file detection sidecar for
kesimeg/yolov8n-clothing-detection  (classes: Clothing, Shoes, Bags, Accessories).

POST an image -> get bounding boxes as JSON, sorted by confidence.
"primary" = highest-confidence item (the auto-selected one),
"boxes"   = every detection (the options to show alongside).

------------------------------------------------------------------
SETUP  (on a machine where huggingface.co is reachable):

    pip install fastapi "uvicorn[standard]" ultralytics huggingface_hub pillow python-multipart

RUN:

    python clothing_sidecar.py                 # http://127.0.0.1:8000
    # or:  uvicorn clothing_sidecar:app --host 0.0.0.0 --port 8000 --workers 4

Parallelism: set WORKERS>1 to spawn one process (and model) per worker.
Each worker handles one inference at a time; N workers = N concurrent requests.

The first start downloads + caches the weights (~6 MB); later starts are offline-friendly.

TEST:

    curl -s -F "file=@IMG_5029.WEBP" http://127.0.0.1:8000/detect | python -m json.tool

CALL FROM NODE:

    import fs from "node:fs";
    const fd = new FormData();
    fd.append("file", new Blob([fs.readFileSync("IMG_5029.WEBP")]), "img.webp");
    const r = await fetch("http://127.0.0.1:8000/detect", { method: "POST", body: fd });
    const { primary, boxes } = await r.json();
------------------------------------------------------------------
"""

import io
import os
import threading

# --- config (override via env vars) ---
REPO     = os.getenv("MODEL_REPO", "kesimeg/yolov8n-clothing-detection")
IMGSZ    = int(os.getenv("IMGSZ", "416"))       # 416 is a good speed/accuracy tradeoff on CPU
CONF     = float(os.getenv("CONF", "0.7"))
MIN_CONF = float(os.getenv("MIN_CONF", "0.6"))  # only return detections above this
IOU      = float(os.getenv("IOU", "0.45"))
THREADS  = int(os.getenv("THREADS", "4"))       # torch/OMP threads *per worker*
WORKERS  = int(os.getenv("WORKERS", "1"))       # uvicorn worker processes (each loads its own model)
MAX_SIDE = int(os.getenv("MAX_SIDE", "1280"))   # downscale before inference; 0 = disabled
MAX_DET  = int(os.getenv("MAX_DET", "10"))      # cap NMS output (4 classes; few items per photo)
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")
HOST     = os.getenv("HOST", "127.0.0.1")
PORT     = int(os.getenv("PORT", "8000"))

# ultralytics / BLAS thread limits — set before heavy imports
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import torch
torch.set_num_threads(THREADS)

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download, list_repo_files


def _is_openvino_dir(path: str) -> bool:
    try:
        return os.path.isdir(path) and any(name.endswith(".xml") for name in os.listdir(path))
    except OSError:
        return False


def _predict_kwargs(imgsz: int) -> dict:
    return {
        "imgsz": imgsz,
        "verbose": False,
        "device": "cpu",
        "max_det": MAX_DET,
        "augment": False,
        "save": False,
    }


def _load_model():
    if _is_openvino_dir(MODEL_PATH):
        m = YOLO(MODEL_PATH)
        backend = "openvino"
    else:
        pt = next(f for f in list_repo_files(REPO) if f.endswith(".pt"))
        m = YOLO(hf_hub_download(REPO, pt))
        backend = "pytorch"
    m.predict(Image.new("RGB", (IMGSZ, IMGSZ)), **_predict_kwargs(IMGSZ))  # warmup
    return m, backend


model, MODEL_BACKEND = _load_model()
NAMES = model.names           # e.g. {0:'Clothing', 1:'Shoes', 2:'Bags', 3:'Accessories'}
# Serializes inference within one worker; parallel requests use separate worker processes.
LOCK  = threading.Lock()

app = FastAPI(title="clothing-detection-sidecar")


def _prepare_image(img: Image.Image) -> tuple[Image.Image, float, float]:
    """Downscale large uploads; return (working image, x_scale, y_scale) for box coords."""
    orig_w, orig_h = img.size
    if MAX_SIDE <= 0 or max(orig_w, orig_h) <= MAX_SIDE:
        return img, 1.0, 1.0
    scale = MAX_SIDE / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return resized, orig_w / new_w, orig_h / new_h


@app.get("/health")
def health():
    return {
        "status": "ok",
        "repo": REPO,
        "classes": NAMES,
        "imgsz": IMGSZ,
        "max_side": MAX_SIDE,
        "max_det": MAX_DET,
        "backend": MODEL_BACKEND,
        "model_path": MODEL_PATH if _is_openvino_dir(MODEL_PATH) else REPO,
        "threads_per_worker": THREADS,
        "workers": WORKERS,
    }


@app.post("/detect")
def detect(
    file: UploadFile = File(...),
    conf: float = Query(CONF, ge=0.0, le=1.0),
    imgsz: int = Query(IMGSZ, ge=64, le=1920),
):
    raw = file.file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")  # Pillow handles WEBP/JPEG/PNG
    except Exception:
        raise HTTPException(400, "could not decode image")

    W, H = img.size
    img_in, scale_x, scale_y = _prepare_image(img)
    effective_conf = max(conf, MIN_CONF)
    with LOCK:
        res = model.predict(
            img_in,
            conf=effective_conf,
            iou=IOU,
            **_predict_kwargs(imgsz),
        )[0]

    boxes = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        x1, y1, x2, y2 = (
            round(x1 * scale_x, 1),
            round(y1 * scale_y, 1),
            round(x2 * scale_x, 1),
            round(y2 * scale_y, 1),
        )
        cid = int(b.cls[0])
        boxes.append({
            "label": NAMES[cid],
            "class_id": cid,
            "confidence": round(float(b.conf[0]), 4),
            "box": [x1, y1, x2, y2],          # pixels, original image space
        })

    boxes.sort(key=lambda d: d["confidence"], reverse=True)
    return {
        "width": W,
        "height": H,
        "count": len(boxes),
        "primary": boxes[0] if boxes else None,   # highest-confidence detection
        "boxes": boxes,                           # all detections (the "options")
    }


if __name__ == "__main__":
    import uvicorn

    if WORKERS > 1:
        # Import string required so each worker process loads its own model copy.
        uvicorn.run("clothing_sidecar:app", host=HOST, port=PORT, workers=WORKERS)
    else:
        uvicorn.run(app, host=HOST, port=PORT)

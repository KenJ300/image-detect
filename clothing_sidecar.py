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
    # or:  uvicorn clothing_sidecar:app --host 0.0.0.0 --port 8000

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
REPO    = os.getenv("MODEL_REPO", "kesimeg/yolov8n-clothing-detection")
IMGSZ   = int(os.getenv("IMGSZ", "640"))
CONF    = float(os.getenv("CONF", "0.7"))
MIN_CONF = float(os.getenv("MIN_CONF", "0.6"))  # only return detections above this
IOU     = float(os.getenv("IOU", "0.45"))
THREADS = int(os.getenv("THREADS", "4"))
HOST    = os.getenv("HOST", "127.0.0.1")
PORT    = int(os.getenv("PORT", "8000"))

# keep CPU threads sane *before* torch is imported (avoids oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))

import torch
torch.set_num_threads(THREADS)

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download, list_repo_files


def _load_model():
    # filename-agnostic: grab whatever .pt the repo ships
    pt = next(f for f in list_repo_files(REPO) if f.endswith(".pt"))
    m = YOLO(hf_hub_download(REPO, pt))
    m.predict(Image.new("RGB", (IMGSZ, IMGSZ)), imgsz=IMGSZ, verbose=False)  # warmup
    return m


model = _load_model()
NAMES = model.names           # e.g. {0:'Clothing', 1:'Shoes', 2:'Bags', 3:'Accessories'}
LOCK  = threading.Lock()      # YOLO.predict is not safe to call concurrently

app = FastAPI(title="clothing-detection-sidecar")


@app.get("/health")
def health():
    return {"status": "ok", "repo": REPO, "classes": NAMES, "imgsz": IMGSZ}


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
    effective_conf = max(conf, MIN_CONF)
    with LOCK:
        res = model.predict(img, imgsz=imgsz, conf=effective_conf, iou=IOU, verbose=False)[0]

    boxes = []
    for b in res.boxes:
        x1, y1, x2, y2 = (round(v, 1) for v in b.xyxy[0].tolist())
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
    uvicorn.run(app, host=HOST, port=PORT)

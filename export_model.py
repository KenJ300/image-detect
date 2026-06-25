#!/usr/bin/env python3
"""Download HF weights and export OpenVINO model into MODEL_PATH (used at Docker build)."""

import os
import shutil
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from huggingface_hub import hf_hub_download, list_repo_files
from ultralytics import YOLO

REPO = os.getenv("MODEL_REPO", "kesimeg/yolov8n-clothing-detection")
IMGSZ = int(os.getenv("IMGSZ", "416"))
DEST = Path(os.getenv("MODEL_PATH", "/app/model"))


def main():
    pt = next(f for f in list_repo_files(REPO) if f.endswith(".pt"))
    pt_path = hf_hub_download(REPO, pt)
    exported = YOLO(pt_path).export(format="openvino", imgsz=IMGSZ, half=True)
    shutil.rmtree(DEST, ignore_errors=True)
    shutil.copytree(exported, DEST)
    print(f"OpenVINO model ready at {DEST}")


if __name__ == "__main__":
    main()

FROM python:3.12-slim

WORKDIR /app

# libgl + libglib are sometimes required by OpenCV (pulled in by ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY clothing_sidecar.py export_model.py .

ENV HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/cache/huggingface \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MODEL_PATH=/app/clothing_openvino_model \
    IMGSZ=416 \
    THREADS=4 \
    WORKERS=4 \
    MAX_SIDE=1280 \
    MAX_DET=10

# Download weights + export OpenVINO FP16 model (needs network during `docker build`)
RUN python export_model.py

EXPOSE 8000

CMD ["python", "clothing_sidecar.py"]

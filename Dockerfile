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

COPY clothing_sidecar.py .

ENV HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/cache/huggingface

# Pre-download model weights at build time (needs network during `docker build`)
RUN python -c "\
from huggingface_hub import hf_hub_download, list_repo_files; \
repo='kesimeg/yolov8n-clothing-detection'; \
pt=next(f for f in list_repo_files(repo) if f.endswith('.pt')); \
hf_hub_download(repo, pt)"

EXPOSE 8000

CMD ["uvicorn", "clothing_sidecar:app", "--host", "0.0.0.0", "--port", "8000"]

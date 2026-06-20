FROM python:3.12-slim

WORKDIR /code

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8s weights at build time so first request isn't slow
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

COPY . .

RUN mkdir -p data/test_images outputs/violations models data

# Hugging Face requires exactly port 7860
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

CMD ["python", "app/app.py"]

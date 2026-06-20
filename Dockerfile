FROM python:3.12-slim

WORKDIR /code

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

COPY . .

RUN mkdir -p data/test_images outputs/violations models

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

CMD ["python", "app/app.py"]

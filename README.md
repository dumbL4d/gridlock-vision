# Gridlock Vision

> AI-powered traffic violation detection system using computer vision and deep learning.

Gridlock Vision is an end-to-end system that detects traffic violations from images using YOLOv8 object detection, EasyOCR for license plate recognition, and a Flask web interface for review and analytics.

## Features

- **Vehicle Detection** — detects cars, motorcycles, buses, trucks, bicycles, and persons using YOLOv8
- **Violation Detection** — identifies 5 violation types out of the box:
  - `HELMET_VIOLATION` — motorcycle rider without helmet
  - `TRIPLE_RIDING` — three or more persons on a single motorcycle
  - `ILLEGAL_PARKING` — vehicle parked in a no-parking zone (via zone mask)
  - `NO_SEATBELT` — driver without seatbelt
  - `STOPLINE_VIOLATION` — vehicle crossing a stop line
- **License Plate Recognition** — detects and reads Indian license plates using EasyOCR with regex validation (`[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}`)
- **Evidence Generation** — draws annotated images with bounding boxes, violation badges, and metadata overlay
- **Web Dashboard** — Flask-based UI for uploading images and viewing results with real-time analytics
- **Analytics Engine** — violation type mapping, coverage analysis, daily trends, and BTP dataset integration
- **SQLite Database** — stores violation records with search by plate number and violation type

## Architecture

```
gridlock-vision/
├── app/
│   ├── app.py              # Flask application (6 routes)
│   └── templates/
│       ├── index.html       # Upload page with drag-and-drop
│       └── dashboard.html   # Analytics dashboard with Chart.js
├── src/
│   ├── config.py            # Paths, classes, thresholds
│   ├── preprocess.py        # Image preprocessing (CLAHE, denoise, sharpen, resize)
│   ├── detector.py          # YOLOv8 inference wrapper
│   ├── violation_engine.py  # Violation check logic (5 types)
│   ├── ocr.py               # License plate reader with EasyOCR
│   ├── evidence.py          # Annotated evidence image generation
│   ├── database.py          # SQLite CRUD + stats aggregation
│   └── evaluate.py          # Performance evaluation (mAP, precision, recall, F1, speed)
├── analytics/
│   └── btp_analysis.py      # BTP dataset analysis with charts
├── models/                  # YOLO model files
├── data/                    # Test images and violation database
├── outputs/                 # Generated evidence images
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/dumbL4d/gridlock-vision.git
cd gridlock-vision

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `ultralytics` — YOLOv8 model inference
- `opencv-python` — image processing
- `easyocr` — optical character recognition
- `flask` — web server
- `pandas` — data analysis
- `Pillow` — image handling

## Usage

### Web Interface

```bash
python app/app.py
```

Open `http://localhost:4567` in your browser. Upload an image to detect violations.

### API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Upload page |
| POST | `/analyze` | Upload image for violation analysis |
| GET | `/dashboard` | Analytics dashboard |
| GET | `/violations?plate=MH12` | Search violations by plate |
| GET | `/violations?type=HELMET_VIOLATION` | Search violations by type |
| GET | `/evidence/<filename>` | View annotated evidence image |
| GET | `/api/stats` | Get aggregate statistics |

### Command-Line Testing

Each module can be run independently for testing:

```bash
python src/detector.py          # Test vehicle detection
python src/violation_engine.py  # Test violation detection
python src/ocr.py               # Test license plate reading
python src/evidence.py          # Test evidence generation
python src/database.py          # Test database operations
python src/preprocess.py        # Test image preprocessing
python src/evaluate.py          # Run performance evaluation
python analytics/btp_analysis.py  # Generate BTP analysis charts
```

### Configuration

Edit `src/config.py` to customize:

- `MODEL_PATH` — path to YOLO model file (default: `models/best.pt`, fallback: `yolov8s.pt`)
- `DB_PATH` — SQLite database location
- `OUTPUT_DIR` — directory for annotated evidence images
- `VIOLATION_CLASSES` — list of detectable violation types
- `CONFIDENCE_THRESHOLD` — minimum confidence for detections (default: 0.5)

## Models

The system uses YOLOv8 for object detection. Place your trained model at `models/best.pt`. If no custom model is found, it falls back to `yolov8s.pt` (auto-downloaded by ultralytics).

Secondary models for helmet and seatbelt classification default to the main detection model unless specialized models are provided.

## Violation Detection

| Violation Type | Method | Description |
|----------------|--------|-------------|
| HELMET_VIOLATION | Head crop → secondary YOLO | Detects motorcycle riders without helmet |
| TRIPLE_RIDING | IoU overlap count | ≥3 persons overlapping a motorcycle bounding box |
| ILLEGAL_PARKING | Zone mask overlap | Vehicle bounding box overlap >30% with no-parking zone |
| NO_SEATBELT | Driver crop → secondary YOLO | Detects front-seat driver without seatbelt |
| STOPLINE_VIOLATION | Bbox bottom vs stopline Y | Vehicle bottom edge past the stop line |

## BTP Dataset Coverage

The analytics engine maps BTP (Bangalore Traffic Police) violation types to our detection classes:

- Our system covers **62.5%** of common violation types (10 of 16 mapped)
- Detected classes: `ILLEGAL_PARKING`, `WRONG_WAY`, `HELMET_VIOLATION`, `SPEEDING`, `RED_LIGHT_VIOLATION`, `NO_SEATBELT`, `TRIPLE_RIDING`, `LANE_VIOLATION`, `STOPLINE_VIOLATION`, `PLATE_VIOLATION`
- Uncovered: `DOCUMENT_VIOLATION`, `MOBILE_USAGE`, `OBSTRUCTION`, `DRUNK_DRIVING`, `INDECENCY`, `POLLUTION_VIOLATION`

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**dumbL4d** — [GitHub](https://github.com/dumbL4d)

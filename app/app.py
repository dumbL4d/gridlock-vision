import os
import sys
import time
import uuid
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import ImagePreprocessor
from src.detector import TrafficDetector
from src.violation_engine import ViolationEngine
from src.ocr import PlateReader
from src.evidence import EvidenceGenerator
from src.database import ViolationDB
from src.config import MODEL_PATH, DB_PATH, OUTPUT_DIR

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(
    os.path.dirname(__file__), "..", "data", "test_images"
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

preprocessor = ImagePreprocessor()
detector = TrafficDetector(MODEL_PATH if os.path.exists(MODEL_PATH) else "yolov8s.pt")
engine = ViolationEngine(detector.model)
plate_reader = PlateReader()
evidence_gen = EvidenceGenerator()
db = ViolationDB(DB_PATH)

print("Warming up EasyOCR...")
_ = plate_reader.reader.readtext(np.zeros((100, 100, 3), dtype=np.uint8))
print("EasyOCR ready.")


def generate_case_id():
    short = uuid.uuid4().hex[:6]
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"VL-{ts}-{short}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("jpg", "jpeg", "png"):
        return jsonify({"error": "Unsupported format. Use jpg/jpeg/png"}), 400
    case_id = generate_case_id()
    location = request.form.get("location", "")
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{case_id}.{ext}")
    file.save(save_path)

    start = time.perf_counter()

    try:
        original, pil_img = preprocessor.preprocess(save_path)
        detections, _ = detector.detect(save_path)
        violations = engine.analyze_all(detections, original)
        vehicles = detector.get_vehicles(detections)
        plate_text = None
        for v in vehicles:
            result = plate_reader.extract_plate(original, v["bbox"])
            if result["plate_text"]:
                plate_text = result["plate_text"]
                break
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": location or "Unknown",
            "violation_count": len(violations),
            "case_id": case_id,
        }
        annotated = evidence_gen.generate_evidence(
            original, detections, violations, metadata
        )
        violations_dir = os.path.join(OUTPUT_DIR, "violations")
        annotated_path = evidence_gen.save_evidence(
            annotated, case_id, violations_dir
        )
        if violations:
            db.insert_violation(
                case_id=case_id,
                plate=plate_text,
                violations_list=violations,
                image_path=save_path,
                annotated_path=annotated_path,
                location=location or None,
            )
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    elapsed = round((time.perf_counter() - start) * 1000, 2)

    return jsonify({
        "case_id": case_id,
        "detections_count": len(detections),
        "violations": [
            {
                "type": v["violation_type"],
                "confidence": v["confidence"],
                "description": v["description"],
            }
            for v in violations
        ],
        "plate_number": plate_text,
        "annotated_image_url": f"/evidence/{case_id}.jpg",
        "processing_time_ms": elapsed,
    })


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/violations")
def get_violations():
    plate = request.args.get("plate")
    vtype = request.args.get("type")
    if plate:
        rows = db.search_by_plate(plate)
    elif vtype:
        rows = db.search_by_type(vtype)
    else:
        rows = db.get_all()
    return jsonify(rows)


@app.route("/evidence/<filename>")
def serve_evidence(filename):
    violations_dir = os.path.join(OUTPUT_DIR, "violations")
    if not os.path.isdir(violations_dir):
        os.makedirs(violations_dir, exist_ok=True)
    return send_from_directory(violations_dir, filename)


@app.route("/api/stats")
def api_stats():
    return jsonify(db.get_stats())


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=4567)

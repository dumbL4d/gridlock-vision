import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
DB_PATH = os.path.join(BASE_DIR, "data", "violations.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

VIOLATION_CLASSES = [
    "red_light_violation",
    "speeding",
    "wrong_way",
    "no_helmet",
    "no_seatbelt",
    "illegal_parking",
    "stop_line_violation",
    "lane_violation",
]

CONFIDENCE_THRESHOLD = 0.5

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

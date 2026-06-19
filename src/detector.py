from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class TrafficDetector:
    def __init__(self, model_path: str = "yolov8s.pt"):
        self.model = YOLO(model_path)
        self.target_classes = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

    def _parse_results(self, results):
        detections = []
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box in boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in self.target_classes:
                continue
            conf = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detections.append({
                "class_id": cls_id,
                "label": self.target_classes[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "center_point": (cx, cy),
            })
        return detections

    def detect(self, image_path: str):
        results = self.model(str(image_path))
        return self._parse_results(results), results

    def detect_from_array(self, img_array: np.ndarray):
        results = self.model(img_array)
        return self._parse_results(results), results

    def get_vehicles(self, detections: list) -> list:
        vehicle_ids = {1, 2, 3, 5, 7}
        return [d for d in detections if d["class_id"] in vehicle_ids]

    def get_persons(self, detections: list) -> list:
        return [d for d in detections if d["class_id"] == 0]


if __name__ == "__main__":
    test_dir = Path(__file__).resolve().parent.parent / "data" / "test_images"
    if not test_dir.exists() or not any(test_dir.iterdir()):
        print(f"Place test images in {test_dir} and re-run.")
        exit(0)

    detector = TrafficDetector()
    for img_path in list(test_dir.glob("*.*"))[:3]:
        print(f"\n--- {img_path.name} ---")
        dets, _ = detector.detect(str(img_path))
        print(f"Total target detections: {len(dets)}")
        for d in dets:
            print(f"  {d['label']:>12}  conf={d['confidence']:.2f}  "
                  f"bbox={[int(v) for v in d['bbox']]}  center={d['center_point']}")
        print(f"  Vehicles: {len(detector.get_vehicles(dets))}")
        print(f"  Persons:  {len(detector.get_persons(dets))}")

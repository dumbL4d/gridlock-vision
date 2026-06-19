import os
import cv2
import numpy as np
from datetime import datetime


class EvidenceGenerator:
    COLOR_MAP = {
        "HELMET_VIOLATION": (0, 0, 255),
        "ILLEGAL_PARKING": (0, 165, 255),
        "TRIPLE_RIDING": (128, 0, 128),
        "STOPLINE_VIOLATION": (0, 255, 255),
        "NO_SEATBELT": (255, 0, 0),
    }

    def draw_detections(self, img: np.ndarray,
                        detections: list) -> np.ndarray:
        out = img.copy()
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d['label']} {d['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1),
                          (0, 255, 0), -1)
            cv2.putText(out, label, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return out

    def draw_violations(self, img: np.ndarray,
                        violations: list) -> np.ndarray:
        out = img.copy()
        for v in violations:
            vtype = v["violation_type"]
            color = self.COLOR_MAP.get(vtype, (255, 255, 255))
            x1, y1, x2, y2 = [int(v) for v in v["bbox"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            badge = f"{vtype} {v['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.55, 2)
            cv2.rectangle(out, (x1, y1), (x1 + tw + 8, y1 + th + 8),
                          color, -1)
            cv2.putText(out, badge, (x1 + 4, y1 + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        return out

    def add_metadata_overlay(self, img: np.ndarray,
                             metadata: dict) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        bar_h = 50
        overlay = out.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
        ts = metadata.get("timestamp", datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"))
        location = metadata.get("location", "N/A")
        vcount = metadata.get("violation_count", 0)
        case_id = metadata.get("case_id", "N/A")
        cv2.putText(out, f"Case: {case_id}", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(out, f"Time: {ts}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(out, f"Location: {location}", (w // 3, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(out, f"Violations: {vcount}", (2 * w // 3, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out

    def generate_evidence(self, img: np.ndarray, detections: list,
                          violations: list, metadata: dict) -> np.ndarray:
        out = self.draw_detections(img, detections)
        out = self.draw_violations(out, violations)
        out = self.add_metadata_overlay(out, metadata)
        return out

    def save_evidence(self, img: np.ndarray, case_id: str,
                      output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{case_id}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return path


if __name__ == "__main__":
    from pathlib import Path
    from detector import TrafficDetector
    from violation_engine import ViolationEngine
    from config import OUTPUT_DIR

    test_dir = Path(__file__).resolve().parent.parent / "data" / "test_images"
    if not test_dir.exists() or not any(test_dir.iterdir()):
        print(f"Place test images in {test_dir} and re-run.")
        exit(0)

    detector = TrafficDetector()
    engine = ViolationEngine(detector.model)
    gen = EvidenceGenerator()

    for img_path in list(test_dir.glob("*.*"))[:2]:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        dets, _ = detector.detect(str(img_path))
        violations = engine.analyze_all(dets, img_bgr)
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": "Test Junction",
            "violation_count": len(violations),
            "case_id": f"TEST_{img_path.stem}",
        }
        result = gen.generate_evidence(img_bgr, dets, violations, metadata)
        saved = gen.save_evidence(result, metadata["case_id"], OUTPUT_DIR)
        print(f"Saved: {saved} ({len(violations)} violations)")

import numpy as np
import cv2
from ultralytics import YOLO


def compute_iou(boxA: list, boxB: list) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)


class ViolationEngine:
    def __init__(self, model: YOLO, helmet_model: YOLO = None,
                 seatbelt_model: YOLO = None):
        self.model = model
        self.helmet_model = helmet_model or model
        self.seatbelt_model = seatbelt_model or model

    def check_helmet(self, detections: list, img: np.ndarray) -> list:
        violations = []
        motorcycles = [d for d in detections
                       if d["label"] in ("motorcycle",)]
        persons = [d for d in detections if d["label"] == "person"]
        for mc in motorcycles:
            mx1, my1, mx2, my2 = mc["bbox"]
            crop_top = my1
            crop_bottom = my1 + (my2 - my1) * 0.4
            crop_left = mx1
            crop_right = mx2
            for p in persons:
                px1, py1, px2, py2 = p["bbox"]
                iou = compute_iou(mc["bbox"], p["bbox"])
                if iou < 0.1:
                    continue
                head_top = int(py1)
                head_bottom = int(py1 + (py2 - py1) * 0.35)
                head_left = int(max(crop_left, px1))
                head_right = int(min(crop_right, px2))
                h_img, w_img = img.shape[:2]
                head_top, head_left = max(0, head_top), max(0, head_left)
                head_bottom = min(h_img, head_bottom)
                head_right = min(w_img, head_right)
                if head_bottom <= head_top or head_right <= head_left:
                    continue
                head_crop = img[head_top:head_bottom, head_left:head_right]
                if head_crop.size == 0:
                    continue
                head_crop_rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
                results = self.helmet_model.predict(
                    head_crop_rgb, verbose=False, imgsz=224
                )
                if results[0].probs is not None:
                    top_idx = results[0].probs.top1
                    top_conf = results[0].probs.top1conf.item()
                    pred_class = results[0].names[top_idx]
                    if "helmet" not in pred_class.lower():
                        violations.append({
                            "violation_type": "HELMET_VIOLATION",
                            "confidence": round(top_conf, 3),
                            "bbox": [int(head_left), int(head_top),
                                     int(head_right), int(head_bottom)],
                            "description": (
                                f"Motorcycle rider head detected "
                                f"without helmet (conf={top_conf:.2f})"
                            ),
                        })
                else:
                    helmet_dets = self.helmet_model.predict(
                        head_crop_rgb, verbose=False
                    )[0]
                    has_helmet = False
                    if helmet_dets.boxes is not None:
                        for box in helmet_dets.boxes:
                            lbl = helmet_dets.names[int(box.cls[0].item())]
                            if "helmet" in lbl.lower():
                                has_helmet = True
                                break
                    if not has_helmet:
                        violations.append({
                            "violation_type": "HELMET_VIOLATION",
                            "confidence": 0.5,
                            "bbox": [int(head_left), int(head_top),
                                     int(head_right), int(head_bottom)],
                            "description": (
                                "Motorcycle rider without helmet"
                            ),
                        })
        return violations

    def check_triple_riding(self, detections: list, img: np.ndarray) -> list:
        violations = []
        motorcycles = [d for d in detections if d["label"] == "motorcycle"]
        persons = [d for d in detections if d["label"] == "person"]
        for mc in motorcycles:
            overlapping = []
            for p in persons:
                iou = compute_iou(mc["bbox"], p["bbox"])
                if iou > 0.05:
                    overlapping.append(p)
            if len(overlapping) >= 3:
                violations.append({
                    "violation_type": "TRIPLE_RIDING",
                    "confidence": round(mc["confidence"], 3),
                    "bbox": [int(v) for v in mc["bbox"]],
                    "description": (
                        f"Triple riding detected: {len(overlapping)} "
                        f"persons on one motorcycle"
                    ),
                })
        return violations

    def check_illegal_parking(self, detections: list, img: np.ndarray,
                              zone_mask: np.ndarray = None) -> list:
        violations = []
        if zone_mask is None:
            return violations
        vehicles = [
            d for d in detections
            if d["label"] in ("car", "bus", "truck", "motorcycle", "bicycle")
        ]
        h, w = img.shape[:2]
        if zone_mask.shape != (h, w):
            zone_mask = cv2.resize(
                zone_mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST
            )
        white_pixels = zone_mask.sum()
        if white_pixels == 0:
            return violations
        for v in vehicles:
            x1, y1, x2, y2 = [float(coord) for coord in v["bbox"]]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if 0 <= cx < w and 0 <= cy < h and zone_mask[cy, cx] > 0:
                crop_zone = zone_mask[y1:y2, x1:x2]
                if crop_zone.size > 0:
                    overlap_ratio = crop_zone.sum() / crop_zone.size
                else:
                    overlap_ratio = 0
                if overlap_ratio > 0.3:
                    violations.append({
                        "violation_type": "ILLEGAL_PARKING",
                        "confidence": round(v["confidence"], 3),
                        "bbox": [x1, y1, x2, y2],
                        "description": (
                            f"{v['label'].capitalize()} parked in "
                            f"no-parking zone (overlap={overlap_ratio:.2f})"
                        ),
                    })
        return violations

    def check_seatbelt(self, detections: list, img: np.ndarray) -> list:
        violations = []
        cars = [d for d in detections if d["label"] == "car"]
        for c in cars:
            x1, y1, x2, y2 = [float(v) for v in c["bbox"]]
            driver_left = int(x1)
            driver_right = int(x1 + (x2 - x1) * 0.5)
            driver_top = int(y1)
            driver_bottom = int(y1 + (y2 - y1) * 0.6)
            h_img, w_img = img.shape[:2]
            driver_left, driver_top = max(0, driver_left), max(0, driver_top)
            driver_right = min(w_img, driver_right)
            driver_bottom = min(h_img, driver_bottom)
            if driver_right <= driver_left or driver_bottom <= driver_top:
                continue
            crop = img[driver_top:driver_bottom,
                       driver_left:driver_right]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = self.seatbelt_model.predict(
                crop_rgb, verbose=False, imgsz=224
            )
            if results[0].probs is not None:
                top_idx = results[0].probs.top1
                top_conf = results[0].probs.top1conf.item()
                pred_class = results[0].names[top_idx]
                if "seatbelt" not in pred_class.lower():
                    violations.append({
                        "violation_type": "NO_SEATBELT",
                        "confidence": round(top_conf, 3),
                        "bbox": [driver_left, driver_top,
                                 driver_right, driver_bottom],
                        "description": (
                            f"Driver without seatbelt "
                            f"(conf={top_conf:.2f})"
                        ),
                    })
            else:
                sb_dets = self.seatbelt_model.predict(
                    crop_rgb, verbose=False
                )[0]
                has_seatbelt = False
                if sb_dets.boxes is not None:
                    for box in sb_dets.boxes:
                        lbl = sb_dets.names[int(box.cls[0].item())]
                        if "seatbelt" in lbl.lower():
                            has_seatbelt = True
                            break
                if not has_seatbelt:
                    violations.append({
                        "violation_type": "NO_SEATBELT",
                        "confidence": 0.5,
                        "bbox": [driver_left, driver_top,
                                 driver_right, driver_bottom],
                        "description": "Driver without seatbelt",
                    })
        return violations

    def check_stopline(self, detections: list, img: np.ndarray,
                       stopline_y: int = None) -> list:
        violations = []
        if stopline_y is None:
            return violations
        vehicles = [
            d for d in detections
            if d["label"] in ("car", "bus", "truck", "motorcycle", "bicycle")
        ]
        for v in vehicles:
            _, _, _, y2 = [int(v) for v in v["bbox"]]
            if y2 >= stopline_y:
                violations.append({
                    "violation_type": "STOPLINE_VIOLATION",
                    "confidence": round(v["confidence"], 3),
                    "bbox": [int(v) for v in v["bbox"]],
                    "description": (
                        f"{v['label'].capitalize()} crossed stop line "
                        f"(bottom={y2}, stopline_y={stopline_y})"
                    ),
                })
        return violations

    def analyze_all(self, detections: list, img: np.ndarray,
                    zone_mask: np.ndarray = None,
                    stopline_y: int = None) -> list:
        violations = []
        violations.extend(self.check_helmet(detections, img))
        violations.extend(self.check_triple_riding(detections, img))
        violations.extend(
            self.check_illegal_parking(detections, img, zone_mask)
        )
        violations.extend(self.check_seatbelt(detections, img))
        violations.extend(
            self.check_stopline(detections, img, stopline_y)
        )
        return violations


if __name__ == "__main__":
    from pathlib import Path
    from detector import TrafficDetector

    test_dir = Path(__file__).resolve().parent.parent / "data" / "test_images"
    if not test_dir.exists() or not any(test_dir.iterdir()):
        print(f"Place test images in {test_dir} and re-run.")
        exit(0)

    detector = TrafficDetector()
    engine = ViolationEngine(detector.model)

    for img_path in list(test_dir.glob("*.*"))[:2]:
        print(f"\n========== {img_path.name} ==========")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        dets, _ = detector.detect(str(img_path))
        print(f"Detected objects: {len(dets)}")
        violations = engine.analyze_all(dets, img_bgr)
        if violations:
            print(f"Violations found: {len(violations)}")
            for v in violations:
                print(f"  [{v['violation_type']}] {v['description']}")
        else:
            print("No violations detected.")

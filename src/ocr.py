import re
import cv2
import numpy as np
import easyocr


class PlateReader:
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=False)
        self.plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$")

    def detect_plate_region(self, img: np.ndarray,
                            vehicle_bbox: list) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
        height = y2 - y1
        crop_top = y2 - int(height * 0.4)
        crop_top = max(y1, crop_top)
        crop_left = max(0, x1)
        crop_right = min(img.shape[1], x2)
        crop_bottom = min(img.shape[0], y2)
        if crop_bottom <= crop_top or crop_right <= crop_left:
            return np.zeros((10, 10, 3), dtype=np.uint8)
        crop = img[crop_top:crop_bottom, crop_left:crop_right]
        scale = max(1, 300 / crop.shape[1])
        new_w = int(crop.shape[1] * scale)
        new_h = int(crop.shape[0] * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return crop

    def read_plate(self, plate_img: np.ndarray,
                    debug: bool = False) -> dict:
        if plate_img.size == 0 or plate_img.shape[0] < 5 or plate_img.shape[1] < 5:
            if debug:
                print(f"[DEBUG] Plate image too small: {plate_img.shape}")
            return {"plate_text": None, "confidence": 0.0,
                    "is_valid_format": False, "raw_text": ""}
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        results = self.reader.readtext(enhanced_bgr)
        if debug:
            print(f"[DEBUG] Raw EasyOCR output ({len(results)} candidates):")
            for bbox, text, conf in results:
                print(f"  text='{text}' conf={conf:.3f}")
        best_text = ""
        best_conf = 0.0
        for bbox, text, conf in results:
            if conf < 0.2:
                continue
            cleaned = text.replace(' ', '').replace('-', '').replace('.', '').upper()
            if debug:
                print(f"[DEBUG]   Cleaned: '{text}' -> '{cleaned}'  conf={conf:.3f}")
            if conf > best_conf:
                best_text = cleaned
                best_conf = conf
        is_valid = bool(self.plate_pattern.match(best_text)) if best_text else False
        if debug:
            print(f"[DEBUG] Best text: '{best_text}'  "
                  f"Regex match: {is_valid}")
        return {
            "plate_text": best_text if is_valid else None,
            "confidence": round(best_conf, 3),
            "is_valid_format": is_valid,
            "raw_text": best_text,
        }

    def extract_plate(self, img: np.ndarray, vehicle_bbox: list,
                      debug: bool = False) -> dict:
        x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
        if debug:
            height = y2 - y1
            print(f"[DEBUG] Vehicle bbox: [{x1}, {y1}, {x2}, {y2}]")
            print(f"[DEBUG] Plate region coords: "
                  f"top={max(y1, y2 - int(height * 0.4))}, "
                  f"bottom={min(img.shape[0], y2)}, "
                  f"left={max(0, x1)}, "
                  f"right={min(img.shape[1], x2)}")
        region = self.detect_plate_region(img, vehicle_bbox)
        if debug:
            print(f"[DEBUG] Crop region shape: {region.shape}")
        return self.read_plate(region, debug=debug)

    def draw_plate_annotation(self, img: np.ndarray,
                              vehicle_bbox: list,
                              plate_text: str) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
        label = plate_text if plate_text else "NO_PLATE"
        cv2.rectangle(img, (x1, y2), (x2, y2 + 20),
                      (0, 255, 255), -1)
        cv2.putText(img, label, (x1 + 2, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img


if __name__ == "__main__":
    from pathlib import Path
    from detector import TrafficDetector

    test_dir = Path(__file__).resolve().parent.parent / "data" / "test_images"
    if not test_dir.exists() or not any(test_dir.iterdir()):
        print(f"Place test images in {test_dir} and re-run.")
        exit(0)

    detector = TrafficDetector()
    reader = PlateReader()

    for img_path in list(test_dir.glob("*.*"))[:3]:
        print(f"\n--- {img_path.name} ---")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        dets, _ = detector.detect(str(img_path))
        vehicles = detector.get_vehicles(dets)
        for v in vehicles:
            result = reader.extract_plate(img_bgr, v["bbox"], debug=True)
            status = "VALID" if result["is_valid_format"] else "INVALID"
            text = result["plate_text"] or result["raw_text"] or "N/A"
            print(f"  {v['label']:>12}  plate={text:>15}  "
                  f"conf={result['confidence']:.2f}  [{status}]")
            img_bgr = reader.draw_plate_annotation(
                img_bgr, v["bbox"], result["plate_text"]
            )
        cv2.imshow(str(img_path.name), img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

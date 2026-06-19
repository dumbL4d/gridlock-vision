import os
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict


def compute_iou(boxA: list, boxB: list) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)


class ModelEvaluator:
    def __init__(self, model, test_images_dir: str,
                 ground_truth_json: str):
        self.model = model
        self.test_images_dir = Path(test_images_dir)
        with open(ground_truth_json) as f:
            self.ground_truth = json.load(f)
        self.image_names = sorted(self.ground_truth.keys(),
                                  key=lambda x: self.ground_truth[x].get(
                                      "sort_key", x))

    def evaluate_detection(self, predictions: list, ground_truth: list,
                           iou_threshold: float = 0.5):
        tp, fp, fn = 0, 0, 0
        gt_matched = [False] * len(ground_truth)
        preds_sorted = sorted(predictions,
                              key=lambda x: x["confidence"], reverse=True)
        for pd_ in preds_sorted:
            matched = False
            for j, gt_ in enumerate(ground_truth):
                if gt_matched[j]:
                    continue
                if pd_["label"] != gt_["label"]:
                    continue
                if compute_iou(pd_["bbox"], gt_["bbox"]) >= iou_threshold:
                    gt_matched[j] = True
                    matched = True
                    break
            if matched:
                tp += 1
            else:
                fp += 1
        fn = len(ground_truth) - sum(gt_matched)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return {"tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4)}

    def evaluate_violations(self, predicted_violations: list,
                            true_violations: list):
        true_by_type = defaultdict(list)
        for v in true_violations:
            true_by_type[v["violation_type"]].append(v)
        pred_by_type = defaultdict(list)
        for v in predicted_violations:
            pred_by_type[v["violation_type"]].append(v)
        all_types = set(list(true_by_type.keys()) +
                        list(pred_by_type.keys()))
        per_class = {}
        for vtype in sorted(all_types):
            tp = len(pred_by_type[vtype])
            fp = 0
            fn = len(true_by_type[vtype])
            tp = min(len(pred_by_type[vtype]),
                     len(true_by_type[vtype]))
            fp = max(0, len(pred_by_type[vtype]) - tp)
            fn = max(0, len(true_by_type[vtype]) - tp)
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            per_class[vtype] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        return per_class

    def compute_map(self, all_predictions: dict, all_ground_truths: dict,
                    iou_threshold: float = 0.5):
        class_preds = defaultdict(list)
        class_gts = defaultdict(list)
        for img_name in all_predictions:
            preds = all_predictions[img_name]
            gts = all_ground_truths.get(img_name, {}).get("detections", [])
            for gt_ in gts:
                class_gts[gt_["label"]].append(gt_)
            for pd_ in preds:
                best_iou = 0
                best_gt = None
                for gt_ in gts:
                    if pd_["label"] == gt_["label"]:
                        iou = compute_iou(pd_["bbox"], gt_["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = gt_
                is_match = 1 if (best_gt is not None and
                                 best_iou >= iou_threshold) else 0
                class_preds[pd_["label"]].append({
                    "confidence": pd_["confidence"],
                    "is_match": is_match,
                })
        aps = []
        for cls_name in sorted(class_preds | set(class_gts.keys())):
            preds = sorted(class_preds.get(cls_name, []),
                           key=lambda x: x["confidence"], reverse=True)
            num_gt = len(class_gts.get(cls_name, []))
            if num_gt == 0:
                continue
            tp_cum = 0
            precisions = []
            recalls = []
            for i, p in enumerate(preds):
                tp_cum += p["is_match"]
                fp_cum = (i + 1) - tp_cum
                prec = tp_cum / (tp_cum + fp_cum + 1e-6)
                rec = tp_cum / (num_gt + 1e-6)
                precisions.append(prec)
                recalls.append(rec)
            ap = 0
            for t in np.linspace(0, 1, 11):
                p_max = max([p for p, r in zip(precisions, recalls)
                             if r >= t], default=0)
                ap += p_max / 11
            aps.append(ap)
        return round(np.mean(aps), 4) if aps else 0.0

    def run_full_evaluation(self):
        all_preds = {}
        all_gts = {}
        detection_metrics = {"tp": 0, "fp": 0, "fn": 0}
        all_pred_violations = []
        all_true_violations = []
        per_image_results = []

        for img_name in self.image_names:
            img_path = self.test_images_dir / img_name
            if not img_path.exists():
                continue
            gt_entry = self.ground_truth[img_name]
            gt_detections = gt_entry.get("detections", [])
            gt_violations = gt_entry.get("violations", [])
            preds, _ = self.model.detect(str(img_path))
            viol_engine = __import__("src.violation_engine",
                                     fromlist=["ViolationEngine"])
            engine = viol_engine.ViolationEngine(self.model.model)
            img_bgr = __import__("cv2").imread(str(img_path))
            pred_violations = engine.analyze_all(preds, img_bgr)
            eval_det = self.evaluate_detection(preds, gt_detections)
            detection_metrics["tp"] += eval_det["tp"]
            detection_metrics["fp"] += eval_det["fp"]
            detection_metrics["fn"] += eval_det["fn"]
            all_preds[img_name] = preds
            all_gts[img_name] = gt_entry
            all_pred_violations.extend(pred_violations)
            all_true_violations.extend(gt_violations)
            per_image_results.append({
                "image": img_name,
                "detections": eval_det,
            })

        total_tp = detection_metrics["tp"]
        total_fp = detection_metrics["fp"]
        total_fn = detection_metrics["fn"]
        overall_precision = total_tp / (total_tp + total_fp + 1e-6)
        overall_recall = total_tp / (total_tp + total_fn + 1e-6)
        overall_f1 = 2 * overall_precision * overall_recall / \
                     (overall_precision + overall_recall + 1e-6)

        map_score = self.compute_map(all_preds, all_gts)

        viol_metrics = self.evaluate_violations(all_pred_violations,
                                                all_true_violations)

        print("=" * 68)
        print("  GRIDLOCK VISION — Model Evaluation Report")
        print("=" * 68)
        print(f"\n  Detection Performance (overall)")
        print(f"  {'-' * 40}")
        print(f"  {'TP':>8}   {total_tp}")
        print(f"  {'FP':>8}   {total_fp}")
        print(f"  {'FN':>8}   {total_fn}")
        print(f"  {'Precision':>12} {overall_precision:.4f}")
        print(f"  {'Recall':>12} {overall_recall:.4f}")
        print(f"  {'F1-score':>12} {overall_f1:.4f}")
        print(f"  {'mAP@0.5':>12} {map_score:.4f}")
        print(f"\n  Violation Detection (per class)")
        print(f"  {'Type':<28} {'Precision':>10} {'Recall':>10} "
              f"{'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
        print(f"  {'-' * 75}")
        for vtype, m in sorted(viol_metrics.items()):
            print(f"  {vtype:<28} {m['precision']:>10.4f} "
                  f"{m['recall']:>10.4f} {m['f1']:>8.4f} "
                  f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")
        print("=" * 68)

        results = {
            "detection": {
                "tp": total_tp, "fp": total_fp, "fn": total_fn,
                "precision": round(overall_precision, 4),
                "recall": round(overall_recall, 4),
                "f1": round(overall_f1, 4),
                "mAP_0.5": map_score,
            },
            "violations": viol_metrics,
            "per_image": per_image_results,
            "num_images": len(per_image_results),
        }
        eval_path = os.path.join(
            os.path.dirname(__file__), "..", "analytics",
            "eval_results.json"
        )
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {eval_path}")
        return results

    def benchmark_speed(self, n_images: int = 50):
        image_files = sorted(self.test_images_dir.glob("*.*"))
        if not image_files:
            print("  No images found for speed benchmark.")
            return {}
        selected = image_files[:n_images]
        times = []
        for i, img_path in enumerate(selected):
            start = time.perf_counter()
            self.model.detect(str(img_path))
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        avg_ms = np.mean(times)
        fps = 1000 / avg_ms if avg_ms > 0 else 0
        print(f"\n  Speed Benchmark ({len(selected)} images)")
        print(f"  {'-' * 40}")
        print(f"  {'Avg inference':>16}  {avg_ms:.2f} ms")
        print(f"  {'FPS':>16}  {fps:.2f}")
        print(f"  {'Min':>16}  {min(times):.2f} ms")
        print(f"  {'Max':>16}  {max(times):.2f} ms")
        print(f"  {'Std dev':>16}  {np.std(times):.2f} ms")
        return {"avg_ms": round(avg_ms, 2), "fps": round(fps, 2),
                "min_ms": round(min(times), 2),
                "max_ms": round(max(times), 2),
                "std_ms": round(np.std(times).item(), 2),
                "num_images": len(selected)}


if __name__ == "__main__":
    from pathlib import Path
    from detector import TrafficDetector

    test_dir = Path(__file__).resolve().parent.parent / "data" / "test_images"
    gt_path = os.path.join(os.path.dirname(__file__), "..",
                           "data", "ground_truth.json")
    if not test_dir.exists() or not any(test_dir.iterdir()):
        print(f"Place test images in {test_dir} and re-run.")
        exit(0)
    if not os.path.exists(gt_path):
        dummy_gt = {}
        for img_path in sorted(test_dir.glob("*.*"))[:10]:
            dummy_gt[img_path.name] = {"detections": [], "violations": []}
        with open(gt_path, "w") as f:
            json.dump(dummy_gt, f, indent=2)
        print(f"Created dummy ground truth at {gt_path}. "
              f"Populate with real labels for proper evaluation.")
    detector = TrafficDetector()
    evaluator = ModelEvaluator(detector, str(test_dir), gt_path)
    evaluator.run_full_evaluation()
    evaluator.benchmark_speed(n_images=10)

"""Create a ranked visual error gallery for a YOLO model."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw


NAMES = ["tasit", "insan", "UAP", "UAI"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union else 0.0


def read_labels(label_path: Path, width: int, height: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    rows = []
    if not label_path.is_file():
        return rows
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if len(fields) != 5:
            continue
        class_id, cx, cy, w, h = int(fields[0]), *map(float, fields[1:])
        rows.append((class_id, ((cx - w / 2) * width, (cy - h / 2) * height,
                               (cx + w / 2) * width, (cy + h / 2) * height)))
    return rows


def evaluate_image(result, label_path: Path) -> tuple[int, int, float, list, list]:
    image = Image.open(result.path)
    width, height = image.size
    truths = read_labels(label_path, width, height)
    predictions = []
    if result.boxes is not None:
        for box, cls, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            predictions.append((int(cls), tuple(box), float(conf)))
    matched_truths: set[int] = set()
    matched_predictions: set[int] = set()
    match_ious = []
    for prediction_index, (pred_class, pred_box, _) in enumerate(predictions):
        candidates = [
            (iou(pred_box, truth_box), truth_index)
            for truth_index, (truth_class, truth_box) in enumerate(truths)
            if truth_index not in matched_truths and truth_class == pred_class
        ]
        if candidates:
            best_iou, truth_index = max(candidates)
            if best_iou >= 0.5:
                matched_truths.add(truth_index)
                matched_predictions.add(prediction_index)
                match_ious.append(best_iou)
    false_negatives = len(truths) - len(matched_truths)
    false_positives = len(predictions) - len(matched_predictions)
    mean_iou = sum(match_ious) / len(match_ious) if match_ious else 0.0
    return false_negatives, false_positives, mean_iou, truths, predictions


def draw_result(result, truths, predictions, output_path: Path) -> None:
    image = Image.open(result.path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for class_id, box in truths:
        draw.rectangle(box, outline=(0, 210, 0), width=3)
        draw.text((box[0], max(0, box[1] - 15)), f"GT {NAMES[class_id]}", fill=(0, 210, 0))
    for class_id, box, confidence in predictions:
        draw.rectangle(box, outline=(230, 30, 30), width=2)
        draw.text((box[0], box[1] + 2), f"P {NAMES[class_id]} {confidence:.2f}", fill=(230, 30, 30))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ranked YOLO error gallery")
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", default="val_diagnostic/images")
    parser.add_argument("--labels", default="val_diagnostic/labels")
    parser.add_argument("--output", default="reports/hata_galerisi")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    from ultralytics import YOLO

    output = Path(args.output).resolve()
    model = YOLO(str(Path(args.model).resolve()))
    image_paths = sorted(p for p in Path(args.images).glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    records = []
    for result in model.predict(source=str(Path(args.images).resolve()), imgsz=args.imgsz, batch=1, device=0, workers=0, stream=True, verbose=False):
        label_path = Path(args.labels) / f"{Path(result.path).stem}.txt"
        fn, fp, mean_iou, truths, predictions = evaluate_image(result, label_path)
        records.append({"result": result, "fn": fn, "fp": fp, "mean_iou": mean_iou, "truths": truths, "predictions": predictions, "score": fn + fp + (1.0 - mean_iou)})
    records.sort(key=lambda item: item["score"], reverse=True)
    gallery = []
    for index, record in enumerate(records[:args.limit], start=1):
        filename = f"{index:03d}_{Path(record['result'].path).stem}.jpg"
        draw_result(record["result"], record["truths"], record["predictions"], output / "images" / filename)
        gallery.append({"image": f"images/{filename}", "source": Path(record["result"].path).name, "false_negatives": record["fn"], "false_positives": record["fp"], "mean_iou": record["mean_iou"], "score": record["score"]})
    (output / "gallery.json").write_text(json.dumps(gallery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = "\n".join(f"<tr><td>{x['source']}</td><td>{x['false_negatives']}</td><td>{x['false_positives']}</td><td>{x['mean_iou']:.3f}</td><td><img src='{html.escape(x['image'])}' width='640'></td></tr>" for x in gallery)
    (output / "index.html").write_text("<html><body><h1>Error Gallery</h1><table border='1'><tr><th>Source</th><th>FN</th><th>FP</th><th>Mean IoU</th><th>Image</th></tr>" + rows + "</table></body></html>", encoding="utf-8")
    print(json.dumps({"images_scanned": len(records), "gallery_images": len(gallery), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

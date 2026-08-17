"""YOLO modelinin tanisal val uzerinde degerlendirilmesi."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def degerlendir(model_yolu: Path, data_yolu: Path, output_dir: Path, imgsz: int = 640) -> dict[str, Any]:
    """Ultralytics val kosusunu calistirip temel metrikleri kaydeder."""
    try:
        from ultralytics import YOLO
    except ImportError as hata:
        raise RuntimeError(
            "Ultralytics kurulu degil. Kaggle veya yerel ortamda "
            "pip install ultralytics komutunu calistirin."
        ) from hata

    if not model_yolu.is_file():
        raise FileNotFoundError(f"Model bulunamadi: {model_yolu}")
    if not data_yolu.is_file():
        raise FileNotFoundError(f"Dataset YAML bulunamadi: {data_yolu}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_yolu))
    sonuclar = model.val(
        data=str(data_yolu),
        split="val",
        imgsz=imgsz,
        plots=True,
        project=str(output_dir),
        name="yolo_val",
        exist_ok=True,
    )
    metrikler: dict[str, Any] = {
        "model": str(model_yolu.resolve()),
        "data": str(data_yolu.resolve()),
        "imgsz": imgsz,
        "mAP50": float(sonuclar.box.map50),
        "mAP50_95": float(sonuclar.box.map),
        "AP_sinif": [float(deger) for deger in sonuclar.box.ap],
    }
    (output_dir / "temel_metrikler.json").write_text(
        json.dumps(metrikler, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrikler


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO tanisal val degerlendirmesi")
    parser.add_argument("--model", required=True, help="best.pt yolu")
    parser.add_argument("--data", default="val_diagnostic/data.yaml")
    parser.add_argument("--output", default="reports/degerlendirme")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    metrikler = degerlendir(Path(args.model), Path(args.data), Path(args.output), args.imgsz)
    print(json.dumps(metrikler, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

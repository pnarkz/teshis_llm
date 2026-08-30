"""Evaluate the completed D1 run on the locked diagnostic validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate(
    model_path: Path, data_path: Path, output_dir: Path, imgsz: int, scenario: str = "D1"
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics kurulu degil.") from error
    if not model_path.is_file():
        raise FileNotFoundError(f"D1 modeli bulunamadi: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Diagnostic YAML bulunamadi: {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = YOLO(str(model_path)).val(
        data=str(data_path),
        split="val",
        imgsz=imgsz,
        batch=8,
        device=0,
        plots=True,
        # Goreceli project yolu bazi Ultralytics surumlerinde sessizce
        # runs/detect/<project> altina yaziliyor (D3 kosusunda gozlendi);
        # resolve() bunu onler.
        project=str(output_dir.resolve()),
        # Rapor klasoru sozlesmesi (docs/MIMARI.md): her senaryo klasorunde
        # val ciktisi "gorseller/" adini tasir. Senaryo adini buraya tekrar
        # gomen eski bicim ("<senaryo>_val_diagnostic") klasorler elle
        # duzeltilirken geride kalmisti; test_belge_yollari.py bunu yakalar.
        name="gorseller",
        exist_ok=True,
    )
    # Rule 8: sonuclar genel mAP ile birlikte sinif AP *ve recall* icermelidir.
    # ap_class_index, yalnizca val'de ornegi bulunan siniflarin sirasini verir;
    # sinif adlarini bu indise gore eslemek, sinif atlanirsa kaymayi onler.
    class_names = ["tasit", "insan", "UAP", "UAI"]
    gorulen = [class_names[int(i)] for i in result.box.ap_class_index]
    metrics: dict[str, Any] = {
        "scenario": scenario,
        "model": str(model_path.resolve()),
        "data": str(data_path.resolve()),
        "imgsz": imgsz,
        "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "class_names": gorulen,
        "class_ap50": [float(x) for x in result.box.ap50],
        "class_ap50_95": [float(x) for x in result.box.ap],
        "class_precision": [float(x) for x in result.box.p],
        "class_recall": [float(x) for x in result.box.r],
    }
    path = output_dir / "d1_metrics.json"
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 diagnostic validation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="val_diagnostic/data.yaml")
    parser.add_argument("--output", default="reports/eski_D1_protokol_sapmali")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--scenario", default="D1")
    args = parser.parse_args()
    evaluate(Path(args.model), Path(args.data), Path(args.output), args.imgsz, args.scenario)


if __name__ == "__main__":
    main()

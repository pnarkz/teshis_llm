"""Iki YOLO agirligini ayni val kumesinde karsilastirir."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

SINIFLAR = {0: "tasit", 1: "insan", 2: "UAP", 3: "UAI"}


def data_yaml_uret(val_root: Path, output_root: Path) -> Path:
    """Val klasoru icin Kaggle uyumlu gecici YAML uretir."""
    output_root.mkdir(parents=True, exist_ok=True)
    yaml = (
        f"path: {val_root.resolve().as_posix()}\n"
        "train: images\n"
        "val: images\n"
        "nc: 4\n"
        "names: [tasit, insan, UAP, UAI]\n"
    )
    data_path = output_root / "compare_val.yaml"
    data_path.write_text(yaml, encoding="utf-8")
    return data_path


def tek_model_degerlendir(
    model_yolu: Path,
    data_yolu: Path,
    output_root: Path,
    imgsz: int,
) -> dict[str, Any]:
    """Tek modeli val'de degerlendirir."""
    from ultralytics import YOLO

    model = YOLO(str(model_yolu))
    ad = model_yolu.stem.replace("(", "_").replace(")", "_")
    sonuc = model.val(
        data=str(data_yolu),
        split="val",
        imgsz=imgsz,
        device=0,
        workers=2,
        plots=True,
        # Goreceli project yolu bazi Ultralytics surumlerinde sessizce
        # runs/detect/<project> altina yaziliyor; resolve() bunu onler.
        project=str(output_root.resolve()),
        name=ad,
        exist_ok=True,
    )
    kutu = sonuc.box
    siniflar: dict[str, Any] = {}
    for i, sinif_id in enumerate(kutu.ap_class_index):
        siniflar[SINIFLAR[int(sinif_id)]] = {
            "precision": float(kutu.p[i]),
            "recall": float(kutu.r[i]),
            "mAP50": float(kutu.ap50[i]),
            "mAP50_95": float(kutu.ap[i]),
        }
    return {
        "model": str(model_yolu.resolve()),
        "imgsz": imgsz,
        "mAP50": float(kutu.map50),
        "mAP50_95": float(kutu.map),
        "precision": float(kutu.mp),
        "recall": float(kutu.mr),
        "siniflar": siniflar,
    }


def karsilastir(
    referans: Path,
    aday: Path,
    val_root: Path,
    output_root: Path,
    referans_imgsz: int = 640,
    aday_imgsz: int = 768,
) -> dict[str, Any]:
    """Iki modeli ayni val klasorunde sirayla degerlendirir."""
    try:
        import ultralytics  # noqa: F401
    except ImportError as hata:
        raise RuntimeError("Kaggle ortaminda once pip install ultralytics calistirin.") from hata
    if not referans.is_file():
        raise FileNotFoundError(f"Referans model bulunamadi: {referans}")
    if not aday.is_file():
        raise FileNotFoundError(f"Aday model bulunamadi: {aday}")
    if not (val_root / "images").is_dir():
        raise FileNotFoundError(f"Val images klasoru bulunamadi: {val_root / 'images'}")

    output_root.mkdir(parents=True, exist_ok=True)
    data_yolu = data_yaml_uret(val_root, output_root)
    sonuclar = {
        "evaluation_set": str(val_root.resolve()),
        "referans_imgsz": referans_imgsz,
        "aday_imgsz": aday_imgsz,
        "referans": tek_model_degerlendir(referans, data_yolu, output_root, referans_imgsz),
        "aday": tek_model_degerlendir(aday, data_yolu, output_root, aday_imgsz),
    }
    sonuclar["fark_aday_eksi_referans"] = {
        "mAP50": sonuclar["aday"]["mAP50"] - sonuclar["referans"]["mAP50"],
        "mAP50_95": sonuclar["aday"]["mAP50_95"] - sonuclar["referans"]["mAP50_95"],
        "precision": sonuclar["aday"]["precision"] - sonuclar["referans"]["precision"],
        "recall": sonuclar["aday"]["recall"] - sonuclar["referans"]["recall"],
    }
    (output_root / "model_karsilastirma.json").write_text(
        json.dumps(sonuclar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_root / "model_karsilastirma.csv").open("w", newline="", encoding="utf-8") as dosya:
        alanlar = ["model", "mAP50", "mAP50_95", "precision", "recall"]
        yazar = csv.DictWriter(dosya, fieldnames=alanlar)
        yazar.writeheader()
        for ad in ("referans", "aday"):
            yazar.writerow({"model": ad, **{alan: sonuclar[ad][alan] for alan in alanlar[1:]}})
    return sonuclar


def main() -> None:
    parser = argparse.ArgumentParser(description="Iki YOLO modelini karsilastir")
    parser.add_argument("--referans", required=True)
    parser.add_argument("--aday", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--output", default="reports/model_secimi")
    parser.add_argument("--referans-imgsz", type=int, default=640)
    parser.add_argument("--aday-imgsz", type=int, default=768)
    args = parser.parse_args()
    sonuc = karsilastir(
        Path(args.referans),
        Path(args.aday),
        Path(args.val_root),
        Path(args.output),
        args.referans_imgsz,
        args.aday_imgsz,
    )
    print(json.dumps(sonuc["fark_aday_eksi_referans"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

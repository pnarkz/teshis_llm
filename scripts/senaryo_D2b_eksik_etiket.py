"""Komut girisi: senaryo_D2b_eksik_etiket.

Uygulama: teshis/veri/senaryo_d2b_eksik_etiket.py
Dosya haritasi: docs/KOD_HARITASI.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teshis.egitim.protokol import egitim_kwargs  # noqa: E402


from teshis.veri.senaryo_d2b_eksik_etiket import (
    find_image,
    link_or_copy,
    build_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local GPU D2b training")
    parser.add_argument("--dataset", default="C:/Users/ASUS/Desktop/HYZ/dataset", type=Path)
    parser.add_argument("--model", default="main_model.pt", type=Path)
    parser.add_argument("--output-dataset", default="veri_surumleri/v03_d2b_eksik_etiket", type=Path)
    parser.add_argument("--output-root", default="experiments", type=Path)
    parser.add_argument("--run-name", default="run_D2b_42_local")
    parser.add_argument("--drop-ratio", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA bulunamadi; bu kosu PC GPU ile yapilmalidir.")
    data_yaml = build_dataset(args.dataset.resolve(), args.output_dataset.resolve(), args.drop_ratio, args.seed)
    model = YOLO(str(args.model.resolve()))
    run_name = args.run_name
    model.train(
        data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, epochs=args.epochs,
        device=0, workers=0, seed=args.seed,
        project=str(args.output_root.resolve()), name=run_name,
        exist_ok=True, plots=False, val=True,
        **egitim_kwargs(),
    )
    print(f"best_model={args.output_root.resolve() / run_name / 'weights/best.pt'}")


if __name__ == "__main__":
    main()

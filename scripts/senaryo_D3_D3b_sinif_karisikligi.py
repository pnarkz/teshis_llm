"""Komut girisi: senaryo_D3_D3b_sinif_karisikligi.

Uygulama: teshis/veri/senaryo_d3_d3b_sinif_karisikligi.py
Dosya haritasi: docs/KOD_HARITASI.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teshis.egitim.protokol import egitim_kwargs  # noqa: E402


from teshis.veri.senaryo_d3_d3b_sinif_karisikligi import (
    find_image,
    link_or_copy,
    build_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinif karisikligi senaryosu (D3 / D3b)")
    parser.add_argument("--dataset", default="C:/Users/ASUS/Desktop/HYZ/dataset", type=Path)
    parser.add_argument("--model", default="main_model.pt", type=Path)
    parser.add_argument("--output-dataset", default="veri_surumleri/v04_d3_uap_uai_sinif_karisikligi", type=Path)
    parser.add_argument("--output-root", default="experiments", type=Path)
    parser.add_argument("--scenario", default="D3")
    parser.add_argument("--version", default="v04_d3_uap_uai_sinif_karisikligi")
    parser.add_argument("--run-name", default=None, help="Varsayilan: run_<senaryo>_<seed>_local")
    parser.add_argument(
        "--class-pair", type=int, nargs=2, default=(2, 3), metavar=("A", "B"),
        help="Karistirilacak sinif ID cifti. D3: 2 3 (UAP/UAI), D3b: 0 1 (tasit/insan)",
    )
    parser.add_argument("--swap-ratio", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sadece-veri", action="store_true", help="Yalnizca veri surumunu uret, egitme")
    args = parser.parse_args()

    data_yaml = build_dataset(
        args.dataset.resolve(), args.output_dataset.resolve(), args.swap_ratio, args.seed,
        tuple(args.class_pair), args.scenario, args.version,
    )
    if args.sadece_veri:
        print(f"data_yaml={data_yaml}")
        return

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA bulunamadi; bu senaryo yerel GPU ile kosulmalidir.")
    run_name = args.run_name or f"run_{args.scenario}_{args.seed}_local"
    model = YOLO(str(args.model.resolve()))
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

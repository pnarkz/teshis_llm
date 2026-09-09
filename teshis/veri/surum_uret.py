"""v00/D1 icin komut girisi; uygulamalar ilgili senaryo modullerindedir."""
from __future__ import annotations
import argparse
from pathlib import Path
from .dosyalar import find_image, train_labels_sha256
from .referans_v00_saglikli import build_v00
from .senaryo_d1_sinif_yetersizligi import build_d1


def main() -> None:
    parser = argparse.ArgumentParser(description="Veri surumu uret (v00 saglikli veya D1)")
    parser.add_argument("--surum", choices=("v00", "d1"), default="d1")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", default="senaryolar/veri/d1_sinif_yetersizligi.yaml", type=Path)
    parser.add_argument("--output-root", default="veri_surumleri", type=Path)
    args = parser.parse_args()
    if args.surum == "v00":
        manifest = build_v00(args.dataset.resolve(), args.output_root.resolve())
        print(f"v00 manifest: {manifest}")
        return
    manifest = build_d1(args.dataset.resolve(), args.output_root.resolve(), args.config.resolve())
    print(f"D1 manifest: {manifest}")



if __name__ == "__main__":
    main()

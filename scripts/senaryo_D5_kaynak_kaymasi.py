"""Komut girisi: senaryo_D5_kaynak_kaymasi.

Uygulama: teshis/veri/senaryo_d5_kaynak_alani_kaymasi.py
Dosya haritasi: docs/KOD_HARITASI.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARSAYILAN_CONFIG = ROOT / "senaryolar/veri/d5_kaynak_alani_kaymasi.yaml"


from teshis.veri.senaryo_d5_kaynak_alani_kaymasi import (
    find_image,
    build_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="D5 kaynak/alan kaymasi veri surumu")
    parser.add_argument("--dataset", default="C:/Users/ASUS/Desktop/HYZ/dataset", type=Path)
    parser.add_argument("--config", default=VARSAYILAN_CONFIG, type=Path)
    parser.add_argument(
        "--output-dataset", default="veri_surumleri/v07_d5_kaynak_alani_kaymasi", type=Path
    )
    parser.add_argument("--scenario", default="D5")
    parser.add_argument("--version", default="v07_d5_kaynak_alani_kaymasi")
    parser.add_argument(
        "--izinli", nargs="+", default=None, help="Verilmezse config'ten okunur"
    )
    args = parser.parse_args()

    izinli = args.izinli
    if izinli is None:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        izinli = list(config["parametreler"]["izinli_kaynaklar"])
    build_dataset(
        args.dataset.resolve(), args.output_dataset.resolve(), izinli, args.scenario, args.version
    )


if __name__ == "__main__":
    main()

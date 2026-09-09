"""Komut girisi: senaryo_D4_kucuk_nesne.

Uygulama: teshis/veri/senaryo_d4_kucuk_nesne_sinyal_kaybi.py
Dosya haritasi: docs/KOD_HARITASI.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARSAYILAN_CONFIG = ROOT / "senaryolar/veri/d4_kucuk_nesne_sinyal_kaybi.yaml"


from teshis.veri.senaryo_d4_kucuk_nesne_sinyal_kaybi import (
    find_image,
    link_or_copy,
    build_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="D4 kucuk nesne sinyal kaybi veri surumu")
    parser.add_argument("--dataset", default="C:/Users/ASUS/Desktop/HYZ/dataset", type=Path)
    parser.add_argument("--config", default=VARSAYILAN_CONFIG, type=Path)
    parser.add_argument(
        "--output-dataset", default="veri_surumleri/v06_d4_kucuk_nesne_sinyal_kaybi", type=Path
    )
    parser.add_argument("--scenario", default="D4")
    parser.add_argument("--version", default="v06_d4_kucuk_nesne_sinyal_kaybi")
    parser.add_argument("--esik-px", type=float, default=None, help="Verilmezse config'ten okunur")
    args = parser.parse_args()

    esik = args.esik_px
    if esik is None:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        esik = float(config["parametreler"]["etkin_sqrt_alan_esigi_px"])
    build_dataset(
        args.dataset.resolve(), args.output_dataset.resolve(), esik, args.scenario, args.version
    )


if __name__ == "__main__":
    main()

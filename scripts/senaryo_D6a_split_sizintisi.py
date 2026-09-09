"""Komut girisi: senaryo_D6a_split_sizintisi.

Uygulama: teshis/veri/senaryo_d6a_split_sizintisi.py
Dosya haritasi: docs/KOD_HARITASI.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARSAYILAN_CONFIG = ROOT / "senaryolar/veri/d6a_split_sizintisi.yaml"


from teshis.veri.senaryo_d6a_split_sizintisi import (
    find_image,
    link_or_copy,
    build_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="D6a sizintili degerlendirme kumesi")
    parser.add_argument("--dataset", default="C:/Users/ASUS/Desktop/HYZ/dataset", type=Path)
    parser.add_argument("--tani-seti", default=ROOT / "val_diagnostic", type=Path)
    parser.add_argument("--config", default=VARSAYILAN_CONFIG, type=Path)
    parser.add_argument("--output", default=ROOT / "veri_surumleri/v08_d6a_split_sizintisi", type=Path)
    parser.add_argument("--oran", type=float, default=None, help="Verilmezse config'ten okunur")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    oran = args.oran if args.oran is not None else float(config["parametreler"]["kaynak_sizdirma_orani"])
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    build_dataset(args.dataset.resolve(), args.tani_seti.resolve(), args.output.resolve(), oran, seed)


if __name__ == "__main__":
    main()

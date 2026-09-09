"""Komut girisi: senaryo_D6b_tekrar_agirligi.

Uygulama: teshis/veri/senaryo_d6b_tekrar_agirligi.py
Dosya haritasi: docs/KOD_HARITASI.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARSAYILAN_CONFIG = ROOT / "senaryolar/veri/d6b_tekrar_agirligi.yaml"


from teshis.veri.senaryo_d6b_tekrar_agirligi import (
    find_image,
    _siniflar,
    build_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="D6b tekrar agirligi veri surumu")
    parser.add_argument("--dataset", default="C:/Users/ASUS/Desktop/HYZ/dataset", type=Path)
    parser.add_argument("--config", default=VARSAYILAN_CONFIG, type=Path)
    parser.add_argument("--output", default=ROOT / "veri_surumleri/v09_d6b_tekrar_agirligi", type=Path)
    parser.add_argument("--tekrar", type=int, default=None, help="Verilmezse config'ten okunur")
    parser.add_argument(
        "--secim-orani", type=float, default=0.01,
        help="Tekrarlanacak kare orani (varsayilan %%1).",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tekrar = args.tekrar if args.tekrar is not None else int(config["parametreler"]["hedef_tekrar"])
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    build_dataset(args.dataset.resolve(), args.output.resolve(), tekrar, args.secim_orani, seed)


if __name__ == "__main__":
    main()

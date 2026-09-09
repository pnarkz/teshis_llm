"""Komut girisi: senaryo_E1_overfitting.

Uygulama: teshis/veri/senaryo_e1_overfitting.py
Dosya haritasi: docs/KOD_HARITASI.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from teshis.veri.senaryo_e1_overfitting import (
    find_image,
    tabakali_sec,
    _siniflar,
    build_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-root", default=ROOT / "veri_surumleri", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.dataset, args.output_root, args.seed)


if __name__ == "__main__":
    main()

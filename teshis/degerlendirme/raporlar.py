"""Senaryo adindan rapor klasorunu turetmenin TEK kaynagi.

Bu kural daha once iki yerde ayri ayri yaziliydi (demo/data_loader.py ve
teshis/degerlendirme/kanit.py) ve ikisi kacinilmaz olarak birbirinden
ayrildi: C2 kontrol kosusu eklendiginde demo onu bulabiliyordu ama kanit
uretici bulamiyordu. Iki kopyayi senkron tutmaya calismak yerine kural
buraya tasindi.

Adlandirma sozlesmesi docs/MIMARI.md'de tanimli: **onek = ne tur, sonek =
hangisi.**
"""

from __future__ import annotations

import re
from pathlib import Path

KOK = Path(__file__).resolve().parents[2]

# Konvansiyonla TURETILEMEYEN adlar. Buraya yalnizca gercekten kurala
# uymayan durumlar girer; her yeni senaryo icin girdi eklemek gerekmez.
OZEL_KLASOR: dict[str, str] = {
    "v00_saglikli": "referans_v00",
    "v00_saglikli last_pt": "referans_v00_last_pt",
    "v00n": "yolo26n_referans_v00n",
    "D1n": "yolo26n_senaryo_D1n",
}

# Sartname bolum 8 kontrol kosullari (C1, C2, C3) `kontrol_` onekini alir.
KONTROL_KALIBI = re.compile(r"^C\d")


def klasor_adi(senaryo: str) -> str:
    """Senaryo adindan rapor klasoru adini uretir (dosya var olmasa da).

    Kurallar:
      - ozel bir ad tanimliysa o kullanilir,
      - C1/C2/C3 gibi kontrol kosullari `kontrol_` onekini alir,
      - digerleri `senaryo_` onekini alir,
      - bosluklar her durumda alt cizgiye cevrilir.

    >>> klasor_adi("D4")
    'senaryo_D4'
    >>> klasor_adi("E1 last_pt")
    'senaryo_E1_last_pt'
    >>> klasor_adi("C2 seed7")
    'kontrol_C2_seed7'
    >>> klasor_adi("v00_saglikli")
    'referans_v00'
    """
    if senaryo in OZEL_KLASOR:
        return OZEL_KLASOR[senaryo]
    onek = "kontrol_" if KONTROL_KALIBI.match(senaryo) else "senaryo_"
    return f"{onek}{senaryo.replace(' ', '_')}"


def rapor_klasoru(senaryo: str, kok: Path = KOK) -> Path | None:
    """Var olan rapor klasorunu dondurur; yoksa None."""
    klasor = kok / "reports" / klasor_adi(senaryo)
    return klasor if klasor.is_dir() else None

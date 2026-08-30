"""D serisi senaryolarin paylastigi tek kaynak egitim protokolu.

senaryolar/egitim_protokolu.yaml dosyasini okur. Boylece kos.py,
scripts/senaryo_D2b_eksik_etiket.py, scripts/senaryo_D3_D3b_sinif_karisikligi.py, scripts/kaggle_D2a_lokalizasyon_gurultusu.py ve
scripts/kaggle_D2b_eksik_etiket.py ayni lr0/warmup_epochs/augmentasyon degerlerini
kullanir; senaryolar arasinda yalnizca veri suruma bagli olmayan bir
egitim farki sessizce olusmaz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROTOKOL_YOLU = Path(__file__).resolve().parents[2] / "senaryolar" / "egitim_protokolu.yaml"


def yukle(yol: Path = PROTOKOL_YOLU) -> dict[str, Any]:
    """Protokol YAML dosyasini sozluk olarak okur."""
    with Path(yol).resolve().open("r", encoding="utf-8") as dosya:
        return yaml.safe_load(dosya) or {}


def egitim_kwargs(yol: Path = PROTOKOL_YOLU) -> dict[str, Any]:
    """Ultralytics model.train() cagrisina **ile aktarilacak sabit alanlari dondurur."""
    protokol = yukle(yol)
    try:
        return dict(protokol["sabit"])
    except KeyError as hata:
        raise ValueError(f"{yol} icinde 'sabit' anahtari bulunamadi") from hata


def e_senaryo_ayarlari(kod: str, yol: Path = PROTOKOL_YOLU) -> dict[str, Any]:
    """E serisi bir senaryonun sapmalarini ve kosu ayarlarini dondurur.

    D serisi VERIYI bozar, protokolu sabit tutar. E serisi tam tersidir: veri
    temizdir, bozulan EGITIM PROTOKOLUDUR. Sapmalar YAML'da beyan edilir;
    koda dagilmis CLI bayraklariyla degil. Boylece hangi kosunun protokolden
    nerede ayrildigi tek yerden denetlenebilir.
    """
    protokol = yukle(yol)
    seri = protokol.get("e_serisi", {})
    if kod not in seri:
        raise KeyError(
            f"{kod} e_serisi altinda tanimli degil. Mevcut: {sorted(seri)}"
        )
    return seri[kod]


def egitim_kwargs_e(kod: str, yol: Path = PROTOKOL_YOLU) -> dict[str, Any]:
    """Ortak protokolu E senaryosunun beyan edilmis sapmalariyla birlestirir.

    Sapmalar SABIT protokolde zaten var olan alanlari degistirebilir; olmayan
    bir alan eklemek sessiz bir protokol genislemesi olacagi icin reddedilir.
    """
    taban = egitim_kwargs(yol)
    sapmalar = e_senaryo_ayarlari(kod, yol).get("sapmalar") or {}
    bilinmeyen = sorted(set(sapmalar) - set(taban))
    if bilinmeyen:
        raise ValueError(
            f"{kod} sapmalarinda protokolde olmayan alan(lar) var: {bilinmeyen}. "
            "Once senaryolar/egitim_protokolu.yaml 'sabit' blokuna eklenmelidir."
        )
    return {**taban, **sapmalar}

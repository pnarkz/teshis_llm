"""Proje yapilandirmasini yukleyen yardimcilar.

Yollar hakkinda
---------------
`config.yaml` bir donem yalnizca mutlak kullanici yollari tasiyordu
(C:/Users/.../dataset). Baska bir makinede proje calismiyordu ve bu, "yeniden
uretilebilir" iddiasiyla celisiyordu.

Iki sey eklendi:

1. **`config.local.yaml`** varsa `config.yaml` yerine okunur ve Git disidir.
   Boylece herkes kendi yolunu yazar, ortak dosya temiz kalir.
2. **Goreli yollar** proje kokune gore cozulur. `dataset_root: veri/dataset`
   her makinede ayni yeri gosterir.

Not: demo konsolu bu dosyayi HIC kullanmaz; yalnizca `results.csv` ve
`reports/` altini okur. Yani sunum, veri kok yolu yanlis olsa bile calisir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

KOK = Path(__file__).resolve().parents[1]
VARSAYILAN = KOK / "config.yaml"
YEREL = KOK / "config.local.yaml"

# Proje kokune gore cozulmesi gereken alanlar.
YOL_ALANLARI = ("dataset_yaml", "dataset_root", "output_root")


def varsayilan_yol() -> Path:
    """Okunacak yapilandirma dosyasi: yerel surum varsa o kazanir."""
    return YEREL if YEREL.is_file() else VARSAYILAN


def yukle(yol: str | Path | None = None) -> dict[str, Any]:
    """YAML yapilandirmasini sozluk olarak yukler.

    `yol` verilmezse `config.local.yaml` (varsa) yoksa `config.yaml` okunur.
    """
    hedef = Path(yol) if yol is not None else varsayilan_yol()
    with hedef.resolve().open("r", encoding="utf-8") as dosya:
        return yaml.safe_load(dosya) or {}


def _coz(deger: str) -> str:
    """Goreli bir yolu proje kokune gore cozer; mutlak yol oldugu gibi kalir."""
    yol = Path(deger)
    return str(yol if yol.is_absolute() else (KOK / yol))


def aktif_ortam(config: dict[str, Any]) -> dict[str, Any]:
    """Aktif ortam ayarlarini dondurur; yol alanlari cozulmus olarak."""
    ad = config.get("aktif_ortam", "lokal")
    ortamlar = config.get("ortamlar", {})
    if ad not in ortamlar:
        raise ValueError(f"Bilinmeyen ortam: {ad}")
    ortam = dict(ortamlar[ad])
    for alan in YOL_ALANLARI:
        if isinstance(ortam.get(alan), str):
            ortam[alan] = _coz(ortam[alan])
    ortam.setdefault("output_root", str(KOK))
    return ortam

"""Komut girisi: senaryo_E4_cozunurluk_uyumsuzlugu.

Uygulama: teshis/degerlendirme/senaryo_e4_cozunurluk_uyumsuzlugu.py
Dosya haritasi: docs/KOD_HARITASI.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teshis.egitim.protokol import e_senaryo_ayarlari  # noqa: E402

CIKTI = ROOT / "reports/senaryo_E4"


from teshis.degerlendirme.senaryo_e4_cozunurluk_uyumsuzlugu import (
    rapor_yolu,
    degerlendir,
    _oku,
    tarama_raporu,
    anlamlilik_raporu,
)


def main() -> None:
    ayar = e_senaryo_ayarlari("E4")["kosu_ayarlari"]
    egitim_imgsz, olcumler = ayar["egitim_imgsz"], ayar["olcum_imgsz"]

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degerlendir", action="store_true",
                        help="raporlamadan once her cozunurlukte val kos (GPU gerekir)")
    args = parser.parse_args()

    if args.degerlendir:
        degerlendir(olcumler)

    CIKTI.mkdir(parents=True, exist_ok=True)
    tarama = tarama_raporu(egitim_imgsz, olcumler)
    anlamlilik = anlamlilik_raporu(egitim_imgsz, min(olcumler))
    (CIKTI / "e4_cozunurluk_taramasi.json").write_text(
        json.dumps(tarama, indent=2, ensure_ascii=False), encoding="utf-8")
    (CIKTI / "e4_sinif_anlamlilik.json").write_text(
        json.dumps(anlamlilik, indent=2, ensure_ascii=False), encoding="utf-8")

    basliklar = ("imgsz", "oran", "mAP50", "recall", "prec", "fark")
    print("{:>6} {:>6} {:>7} {:>7} {:>7} {:>8}".format(*basliklar))
    for o in tarama["olcumler"]:
        im = "  <- egitim" if o["imgsz"] == egitim_imgsz else ""
        print("{:>6} {:>6.2f} {:>7.3f} {:>7.3f} {:>7.3f} {:>+8.3f}{}".format(
            o["imgsz"], o["egitimle_orani"], o["mAP50"],
            o["recall"], o["precision"], o["mAP50_kaybi"], im))
    tepe = tarama["bulgular"]["tepe_egitim_cozunurlugunde_mi"]
    print("")
    print("Tepe egitim cozunurlugunde mi:", tepe)
    print("Rapor:", CIKTI)


if __name__ == "__main__":
    main()

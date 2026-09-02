"""Defterdeki her kosu icin hata galerisi uretir.

Neden gerekli
-------------
Sartname bolum 9 kanit sozlesmesi her kosu icin **hata ornekleri** ister:
eksik tespit (FN), yanlis pozitif (FP), sinif karisikligi ve kotu
lokalizasyon. Bu, sozlesmenin en uzun sure acik kalan maddesiydi - 24
kosunun 23'unde galeri yoktu ve `kanit.json` hepsini "eksik" isaretliyordu.

Metrikler bir kosunun NE KADAR bozuldugunu soyler; galeri NEYIN bozuldugunu
gosterir. D3b'nin manset bulgusu bir kez yanlis raporlandi ve duzeltilmesi
ancak gorsel kontrolle mumkun oldu; galeri o kontrolun kalici hali.

Cozunurluk
----------
Her kosu KENDI degerlendirme cozunurlugunde islenir (`imgsz_eval`); E4
512'de olculdugu icin galerisi de 512'de uretilir.

Depo maliyeti
-------------
Galeri gorselleri `.gitignore` kapsamindadir (`reports/**/*.jpg`); depoya
yalnizca `gallery.json` girer. Yerel disk maliyeti galeri basina ~4 MB.

Kullanim:
    python scripts/hata_galerisi_toplu.py            # eksik olanlar
    python scripts/hata_galerisi_toplu.py --liste    # kosmadan durumu goster
    python scripts/hata_galerisi_toplu.py --hepsi    # var olanlari da yenile
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teshis.degerlendirme.raporlar import galeri_adi  # noqa: E402


def galeri_yolu(senaryo: str) -> Path:
    """Kanit uretici ile AYNI kurali kullanir (tek kaynak: raporlar.galeri_adi)."""
    return ROOT / "reports" / galeri_adi(senaryo)


def kosular() -> list[dict[str, str]]:
    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def uret(satir: dict[str, str]) -> bool:
    agirlik = ROOT / satir["weights_path"]
    if not agirlik.is_file():
        print(f"  {satir['scenario']:<22} ATLANDI: agirlik yok")
        return False
    hedef = galeri_yolu(satir["scenario"])
    sonuc = subprocess.run(
        [sys.executable, "-m", "teshis.degerlendirme.hata_galerisi",
         "--model", str(agirlik),
         "--output", str(hedef),
         "--imgsz", satir["imgsz_eval"]],
        cwd=ROOT, capture_output=True, text=True,
    )
    if sonuc.returncode != 0:
        son = (sonuc.stderr or sonuc.stdout or "").strip().splitlines()
        print(f"  {satir['scenario']:<22} HATA: {son[-1][:110] if son else '?'}")
        return False
    # Cikis kodu yeterli kanit degildir; beklenen dosya gercekten olustu mu?
    if not (hedef / "gallery.json").is_file():
        print(f"  {satir['scenario']:<22} HATA: gallery.json yazilmadi")
        return False
    print(f"  {satir['scenario']:<22} tamam (imgsz={satir['imgsz_eval']})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hepsi", action="store_true")
    parser.add_argument("--liste", action="store_true")
    args = parser.parse_args()

    satirlar = kosular()
    var = {
        s["scenario"]: (galeri_yolu(s["scenario"]) / "gallery.json").is_file()
        for s in satirlar
    }

    if args.liste:
        for s in satirlar:
            print(f"  {'var' if var[s['scenario']] else 'YOK':<4} {s['scenario']}")
        print(f"\nvar {sum(var.values())} | yok {len(var) - sum(var.values())}")
        return

    hedefler = [s for s in satirlar if args.hepsi or not var[s["scenario"]]]
    if not hedefler:
        print("Tum galeriler mevcut.")
        return

    print(f"{len(hedefler)} galeri uretilecek\n")
    basarili = sum(uret(s) for s in hedefler)
    print(f"\nbasarili: {basarili}/{len(hedefler)}")


if __name__ == "__main__":
    main()

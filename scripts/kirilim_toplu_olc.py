"""Defterdeki her kosu icin kirilim olcumunu (yeniden) uretir.

Neden gerekli
-------------
`metrikler.py` sonradan `goruntu_kayitlari` alanini uretmeye basladi: kare
basina kaynak ve sinif/bant kirilimi. Bu alan olmadan iki sey yapilamaz:

1. **Sartnameye uygun guven araligi.** Sartname (bolum 8) goruntu birimli ve
   kaynak grubuna gore tabakali bootstrap ister. Toplam sayimlardan bu
   hesaplanamaz; kare basina kayit gerekir. Alansiz kosular Wilson araligi
   kullanmak zorunda kaliyor ve Wilson, kutulari bagimsiz saydigi icin
   araligi ~1.5 kat DAR gosteriyor.
2. **Alt grup gurultu bandi.** `teshis/degerlendirme/gurultu.py` bandi
   kirilim dosyalarindan hesaplar; olcumu olmayan kosu banda katilamaz.

Ayrica bircok kosunun hic kirilim olcumu yoktu; kanit sozlesmesi bunlari
"eksik" olarak isaretliyordu.

Cozunurluk
----------
Her kosu KENDI degerlendirme cozunurlugunde olculur (`imgsz_eval`). E4
kosusu 512'de olculdugu icin kirilimi de 512'de uretilmelidir; 768 ile
olcmek onu baska bir kosuya donustururdu.

Kullanim:
    python scripts/kirilim_toplu_olc.py            # eksik + eski olanlar
    python scripts/kirilim_toplu_olc.py --hepsi    # var olanlari da yenile
    python scripts/kirilim_toplu_olc.py --liste    # kosmadan durumu goster
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KIRILIM = ROOT / "reports/kirilim"


def _durum(run_id: str) -> str:
    """`yok` | `eski` (goruntu_kayitlari icermiyor) | `guncel`."""
    yol = KIRILIM / f"{run_id}.json"
    if not yol.is_file():
        return "yok"
    try:
        icerik = json.loads(yol.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "yok"
    return "guncel" if icerik.get("goruntu_kayitlari") else "eski"


def kosular() -> list[dict[str, str]]:
    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def olc(satir: dict[str, str]) -> bool:
    agirlik = ROOT / satir["weights_path"]
    if not agirlik.is_file():
        print(f"  {satir['scenario']:<22} ATLANDI: agirlik yok ({agirlik.name})")
        return False
    hedef = KIRILIM / f"{satir['run_id']}.json"
    sonuc = subprocess.run(
        [sys.executable, "-m", "teshis.degerlendirme.metrikler",
         "--model", str(agirlik),
         "--output", str(hedef),
         "--imgsz", satir["imgsz_eval"]],
        cwd=ROOT, capture_output=True, text=True,
    )
    if sonuc.returncode != 0:
        son = (sonuc.stderr or sonuc.stdout or "").strip().splitlines()
        print(f"  {satir['scenario']:<22} HATA: {son[-1][:120] if son else '?'}")
        return False
    # Cikis kodu yeterli kanit degildir; dosya gercekten yazilmis mi?
    if _durum(satir["run_id"]) != "guncel":
        print(f"  {satir['scenario']:<22} HATA: dosya yazilmadi veya eksik")
        return False
    print(f"  {satir['scenario']:<22} tamam (imgsz={satir['imgsz_eval']})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hepsi", action="store_true",
                        help="guncel olanlari da yeniden olc")
    parser.add_argument("--liste", action="store_true", help="kosmadan durumu goster")
    args = parser.parse_args()

    satirlar = kosular()
    durumlar = {s["run_id"]: _durum(s["run_id"]) for s in satirlar}

    if args.liste:
        for s in satirlar:
            print(f"  {durumlar[s['run_id']]:<7} {s['scenario']:<22} "
                  f"imgsz={s['imgsz_eval']}")
        sayim = {d: sum(1 for v in durumlar.values() if v == d)
                 for d in ("guncel", "eski", "yok")}
        print(f"\nguncel {sayim['guncel']} | eski {sayim['eski']} | yok {sayim['yok']}")
        return

    hedefler = [
        s for s in satirlar
        if args.hepsi or durumlar[s["run_id"]] != "guncel"
    ]
    if not hedefler:
        print("Tum kirilim olcumleri guncel.")
        return

    KIRILIM.mkdir(parents=True, exist_ok=True)
    print(f"{len(hedefler)} kosu olculecek\n")
    basarili = sum(olc(s) for s in hedefler)
    print(f"\nbasarili: {basarili}/{len(hedefler)}")

    kalan = [s["scenario"] for s in satirlar if _durum(s["run_id"]) != "guncel"]
    if kalan:
        print(f"Hala guncel olmayan: {kalan}")


if __name__ == "__main__":
    main()

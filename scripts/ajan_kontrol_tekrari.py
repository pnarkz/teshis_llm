"""Ajanin SAF KONTROL kosularini birden cok kez tekrarlar.

Neden tam deneme degil de yalnizca kontroller?
----------------------------------------------
Projenin en zayif kanitli iddiasi su: *"ajan bozulma yokken sorun
uydurmuyor."* Elimizde yalnizca iki saf kontrol var (Baseline ve C2 seed 7)
ve ikisinde de dogru cevap verdi. Wilson %95 araligi **[0.000, 0.658]** —
yani gercek uydurma orani hala %66'ya kadar cikabilir.

Tam denemeyi tekrarlamak bu araligi daraltmanin pahali yoludur: 11 kosu x
~2.5 API istegi = bir gunluk ucretsiz kotanin tamami (20 istek/gun). Uc tam
tekrar uc gun surer.

Ayni kotayla yalnizca iki kontrolu uc kez tekrarlamak, 6 kosu (~15 istek)
eder ve **tek gunde biter**. Kazanc dogrudan zayif noktaya gider: kontrol
gozlemi 2'den 8'e cikar (2 orijinal + 6 tekrar).

Sinir: tekrarlar ayni modelin ayni girdiye verdigi farkli orneklerdir.
Modelin KARARLILIGINI olcer, farkli modellerde genellenebilirligi degil.

Kullanim:
    python scripts/ajan_kontrol_tekrari.py --tekrar 3
    python scripts/ajan_kontrol_tekrari.py --tekrar 3 --devam   # yarim kalirsa
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CIKTI = ROOT / "reports/ajan_denemesi/kontrol_tekrarlari"

# Bozulma icermeyen kosular. kosu_01 saglikli referans, kosu_11 C2 seed 7.
# Ikisinde de dogru cevap "degisim yok"; baska her cevap uydurmadir.
KONTROL_KOSULARI = ["kosu_01", "kosu_11"]


def _dosya(kosu: str, tekrar: int) -> Path:
    return CIKTI / f"{kosu}_tekrar{tekrar}.json"


def kosuyu_calistir(kosu: str, tekrar: int) -> bool:
    """Tek bir kontrol kosusunu calistirir. Kota bittiyse False doner."""
    hedef = _dosya(kosu, tekrar)
    sonuc = subprocess.run(
        [sys.executable, "-m", "teshis.ajan.ajan",
         "--kosu", kosu,
         "--output", str(hedef),
         "--log", str(hedef.with_name(hedef.stem + "_arac.json"))],
        cwd=ROOT, capture_output=True, text=True,
    )
    ciktilar = (sonuc.stdout or "") + (sonuc.stderr or "")
    if sonuc.returncode != 0:
        if "kota" in ciktilar.lower() or "quota" in ciktilar.lower():
            print(f"  {kosu} tekrar {tekrar}: GUNLUK KOTA BITTI, duruluyor.")
            print("  Kota yenilendiginde --devam ile surdurun.")
            return False
        print(f"  {kosu} tekrar {tekrar}: HATA\n{ciktilar[-600:]}")
        return False
    print(f"  {kosu} tekrar {tekrar}: tamam -> {hedef.name}")
    return True


def toplu_sonuc() -> dict:
    """Tekrarlari okuyup uydurma oranini hesaplar."""
    from teshis.ajan.puanlama import teshis_puani

    kayitlar = []
    for dosya in sorted(CIKTI.glob("kosu_*_tekrar*.json")):
        if dosya.stem.endswith("_arac"):
            continue
        icerik = json.loads(dosya.read_text(encoding="utf-8"))
        for cevap in (icerik if isinstance(icerik, list) else [icerik]):
            puan, _ = teshis_puani("anlamli_degisim_yok", cevap)
            kayitlar.append({
                "dosya": dosya.name,
                "kosu": cevap.get("run_id"),
                "teshis": cevap.get("diagnosis"),
                "dogru": puan == 1.0,
            })
    return kayitlar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tekrar", type=int, default=3)
    parser.add_argument("--devam", action="store_true",
                        help="zaten uretilmis tekrarlari atla")
    parser.add_argument("--ozet", action="store_true",
                        help="kosmadan, mevcut tekrarlarin ozetini bas")
    args = parser.parse_args()

    CIKTI.mkdir(parents=True, exist_ok=True)

    if not args.ozet:
        toplam = len(KONTROL_KOSULARI) * args.tekrar
        print(f"{toplam} kosu planlandi (~{round(toplam * 2.5)} API istegi; "
              f"ucretsiz katman 20/gun)\n")
        for tekrar in range(1, args.tekrar + 1):
            for kosu in KONTROL_KOSULARI:
                if args.devam and _dosya(kosu, tekrar).is_file():
                    print(f"  {kosu} tekrar {tekrar}: zaten var, atlandi")
                    continue
                if not kosuyu_calistir(kosu, tekrar):
                    print("\nYarim kaldi. Kota yenilendiginde:")
                    print("  python scripts/ajan_kontrol_tekrari.py "
                          f"--tekrar {args.tekrar} --devam")
                    return

    kayitlar = toplu_sonuc()
    if not kayitlar:
        print("\nHenuz tekrar uretilmemis.")
        return

    from teshis.degerlendirme.bootstrap import wilson_araligi

    uydurma = sum(1 for k in kayitlar if not k["dogru"])
    n = len(kayitlar)
    alt, ust = wilson_araligi(uydurma, n)
    print(f"\n{'dosya':<28} {'kosu':<9} {'teshis':<34} sonuc")
    for k in kayitlar:
        print(f"{k['dosya']:<28} {str(k['kosu']):<9} {str(k['teshis'])[:33]:<34} "
              f"{'dogru' if k['dogru'] else 'UYDURDU'}")
    print(f"\nSaf kontrolde uydurma: {uydurma}/{n} = {uydurma/n:.3f}")
    print(f"Wilson %95 araligi   : [{alt:.3f}, {ust:.3f}]")
    print("\nSinir: tekrarlar ayni modelin ayni girdiye verdigi farkli "
          "orneklerdir;\nmodelin KARARLILIGINI olcer, farkli modellere "
          "genellenebilirligi degil.")


if __name__ == "__main__":
    main()

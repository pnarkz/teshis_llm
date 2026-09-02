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

Ayni kotayla yalnizca kontrol kosularini tekrarlamak cok daha ucuzdur ve
kazanc dogrudan zayif noktaya gider.

Iki tur kontrol var ve degerleri esit degil:

- **Bagimsiz kontroller** (kosu_12, kosu_13): farkli seed'le egitilmis
  BASKA modeller. Her biri yeni bir gozlemdir.
- **Tekrarlar** (kosu_01, kosu_11): ayni girdiye verilen farkli ornekler.
  Modelin KARARLILIGINI olcer, yeni bilgi eklemez.

Bu yuzden once bagimsiz kontroller kosulur. Ilk tekrar turu dort kosudur
(~10 istek) ve kontrol gozlemini 2'den 6'ya cikarir.

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
#
# kosu_12 ve kosu_13 (seed 13 ve 21) TEKRAR DEGIL, bagimsiz kontrollerdir:
# farkli modeller, farkli metrikler. Istatistiksel olarak ayni girdinin
# tekrarindan daha degerlidirler, bu yuzden once onlar kosulur.
KONTROL_KOSULARI = ["kosu_12", "kosu_13", "kosu_01", "kosu_11"]


def _dosya(kosu: str, tekrar: int) -> Path:
    return CIKTI / f"{kosu}_tekrar{tekrar}.json"


# Gecici sunucu hatasi ile gunluk kota bitisi AYRI seylerdir: birincisinde
# hemen tekrar denenir, ikincisinde ertesi gunu beklemek gerekir. Ilk surum
# ikisini de "kota" diye raporladi ve kullaniciya bir gun beklemesini soyledi.
GECICI_HATALAR = ("503", "unavailable", "high demand", "overloaded",
                  "500", "internal error", "deadline exceeded")
KOTA_HATALARI = ("quota", "kota", "resource_exhausted", "429")


def _hata_turu(ciktilar: str) -> str:
    metin = ciktilar.lower()
    if any(k in metin for k in KOTA_HATALARI):
        return "GUNLUK KOTA BITTI"
    if any(k in metin for k in GECICI_HATALAR):
        return "GECICI SUNUCU HATASI (hemen tekrar denenebilir)"
    return "BILINMEYEN HATA"


def _arsivle(hedef: Path) -> Path | None:
    """Var olan bir cevabi ezmeden yana tasir.

    Ilk surum uzerine yaziyordu ve bu gercek bir kanit kaybina yol acti:
    ajanin ilk YANLIS POZITIF cevabi, araclar duzeltilip kosu tekrarlandiginda
    silindi. Once/sonra karsilastirmasi ancak ikisi de elde oldugunda yapilir.
    """
    if not hedef.is_file():
        return None
    i = 1
    while (yedek := hedef.with_name(f"{hedef.stem}_onceki{i}.json")).is_file():
        i += 1
    hedef.replace(yedek)
    return yedek


def kosuyu_calistir(kosu: str, tekrar: int) -> bool:
    """Tek bir kontrol kosusunu calistirir. Basarisizsa False doner."""
    hedef = _dosya(kosu, tekrar)
    if (yedek := _arsivle(hedef)) is not None:
        print(f"  onceki cevap saklandi -> {yedek.name}")
    sonuc = subprocess.run(
        [sys.executable, "-m", "teshis.ajan.ajan",
         "--kosu", kosu,
         "--output", str(hedef),
         "--log", str(hedef.with_name(hedef.stem + "_arac.json"))],
        cwd=ROOT, capture_output=True, text=True,
    )
    ciktilar = (sonuc.stdout or "") + (sonuc.stderr or "")
    if sonuc.returncode != 0:
        tur = _hata_turu(ciktilar)
        print(f"  {kosu} tekrar {tekrar}: {tur}")
        if tur == "BILINMEYEN HATA":
            son_satir = (ciktilar.strip().splitlines() or [""])[-1]
            print(f"    {son_satir[:200]}")
        return False
    # Basarili gorunse bile dosya yazilmis mi diye BAKILIR. Ilk surumde
    # `--kosu` sonucu yalnizca ekrana basiyordu; script "tamam" diyordu ama
    # diskte hicbir sey yoktu ve iki kosunun API maliyeti bosa gitti.
    if not hedef.is_file():
        print(f"  {kosu} tekrar {tekrar}: KOMUT BASARILI AMA DOSYA YAZILMADI")
        print(f"    beklenen: {hedef}")
        return False
    print(f"  {kosu} tekrar {tekrar}: tamam -> {hedef.name}")
    return True


def _bandli_mi(cevap_dosyasi: Path) -> bool | None:
    """Bu cevap, gurultu bandi donduren araclarla mi uretildi?

    Arac kaydinda aracin DONDURDUGU deger de saklanir; band alanlarinin
    varligi, cevabin hangi arac surumune dayandigini kesin olarak soyler.
    Bu olmadan once/sonra karsilastirmasi belirsiz kalir.
    """
    kayit = cevap_dosyasi.with_name(cevap_dosyasi.stem + "_arac.json")
    if kayit.is_file():
        metin = kayit.read_text(encoding="utf-8")
        if '"cevap"' in metin:            # arac cevaplari saklanmis: kesin bilgi
            return "gurultu_bandi" in metin

    # Arac cevaplari saklanmadan once uretilmis kayitlar icin geriye donuk
    # cikarim: ajan cevabinda GURULTU BANDINA atif yapiyorsa bandli araclari
    # gormus olmalidir. Kalip ozgul olmali - "boyut bandi" projede bandli
    # araclardan cok once vardi ve yalnizca "band" aramak onu da yakalardi.
    icerik = cevap_dosyasi.read_text(encoding="utf-8").lower()
    if not icerik:
        return None
    return any(k in icerik for k in ("band orani", "gurultu band", "gürültü band"))


def toplu_sonuc() -> dict:
    """Tekrarlari okuyup uydurma oranini hesaplar."""
    from teshis.ajan.puanlama import teshis_puani

    kayitlar = []
    for dosya in sorted(CIKTI.glob("kosu_*_tekrar*.json")):
        # Arac kayitlari ve arsivlenmis eski cevaplar ozete girmez.
        if dosya.stem.endswith("_arac") or "_onceki" in dosya.stem:
            continue
        icerik = json.loads(dosya.read_text(encoding="utf-8"))
        for cevap in (icerik if isinstance(icerik, list) else [icerik]):
            puan, _ = teshis_puani("anlamli_degisim_yok", cevap)
            kayitlar.append({
                "dosya": dosya.name,
                "kosu": cevap.get("run_id"),
                "teshis": cevap.get("diagnosis"),
                "dogru": puan == 1.0,
                "bandli_arac": _bandli_mi(dosya),
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
    print("")
    print("{:<28} {:<9} {:<30} {:<9} {}".format(
        "dosya", "kosu", "teshis", "arac", "sonuc"))
    for k in kayitlar:
        arac = {True: "bandli", False: "eski", None: "?"}[k["bandli_arac"]]
        print("{:<28} {:<9} {:<30} {:<9} {}".format(
            k["dosya"], str(k["kosu"]), str(k["teshis"])[:29], arac,
            "dogru" if k["dogru"] else "UYDURDU"))
    if len({k["bandli_arac"] for k in kayitlar}) > 1:
        print("")
        print("UYARI: cevaplar FARKLI arac surumleriyle uretilmis; asagidaki")
        print("oran KARISIK bir olcumdur. Karsilastirma icin hepsi ayni")
        print("surumle yeniden kosulmalidir.")
    print(f"\nSaf kontrolde uydurma: {uydurma}/{n} = {uydurma/n:.3f}")
    print(f"Wilson %95 araligi   : [{alt:.3f}, {ust:.3f}]")
    print("\nSinir: tekrarlar ayni modelin ayni girdiye verdigi farkli "
          "orneklerdir;\nmodelin KARARLILIGINI olcer, farkli modellere "
          "genellenebilirligi degil.")


if __name__ == "__main__":
    main()

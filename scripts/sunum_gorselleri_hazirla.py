"""Taze bir klonda konsolun gorselsiz kalmamasi icin kucuk bir set uretir.

Sorun
-----
`reports/` altindaki gorseller 233 MB tutuyor ve `.gitignore` disinda
birakiliyor - hakli olarak: Ultralytics'in urettigi val_batch izgaralari tek
basina 101 MB. Ama sonuc olarak taze bir klonda **Hata Analizi bolumunun
tamami bos kaliyor** ve senaryo sayfalarinda confusion matrix gorunmuyor.
Sunum baska bir makinede yapilacaksa bu kabul edilemez.

Cozum
-----
Kucultulmus bir SUNUM SETI uretilir ve depoyla birlikte gider:

    demo/assets/sunum_gorselleri/<reports altindaki goreli yol>

Ayna duzeni bilerek birebir: `data_loader.gorsel_coz()` orijinal yolu
bulamazsa ayni goreli yolu bu kokte arar, baska bir eslestirme kuralina
gerek kalmaz.

Neler giriyor
-------------
- **Hata galerisi kareleri**: her galeriden, demonun DORT siralama olcutunun
  (skor, yanlis negatif, yanlis pozitif, dusuk IoU) ilk N'sinin BIRLESIMI.
  Yalnizca skora gore secmek yetmezdi: kullanici siralamayi degistirdiginde
  listeye bambaska kareler gelir ve o kareler bulunamazdi.
- **Confusion matrix'ler**: senaryo basina bir tane.

Neler girmiyor
--------------
- `val_batch*.jpg` (101 MB): Ultralytics'in toplu onizleme izgaralari.
  Konsolda "Ornek tahminler" bolumu bunlari kullanir ve dosya yoksa bolum
  sessizce atlanir - metrik veya kanit kaybi olmaz.
- Egri PNG'leri (24 MB): egitim egrisi zaten kosunun kendi `results.csv`
  dosyasindan CIZILIYOR; PNG yalnizca "Diger grafikler" acilirinda duruyor.

Kullanim
--------
    python scripts/sunum_gorselleri_hazirla.py
    python scripts/sunum_gorselleri_hazirla.py --adet 8 --kenar 1100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
RAPORLAR = KOK / "reports"
HEDEF = KOK / "demo/assets/sunum_gorselleri"

def ayna_yolu(kok: Path, goreli: Path) -> Path:
    """Ayna icindeki kisa, deterministik yol.

    GERCEK HATA: ayna once orijinal dosya adini birebir kullaniyordu ve
    `git clone` Windows'ta COKUYORDU:

        error: unable to create file demo/assets/sunum_gorselleri/
        hata_galerisi_C2_seed13/images/030_aaterm__frameoriginal_0280_jpg.
        rf.6b78233d05175da1fe040a835a29e98d.jpg: Filename too long
        fatal: unable to checkout working tree

    Roboflow adlari ~70 karakter; ust dizinlerle birlikte 260 karakterlik
    MAX_PATH sinirini asiyor. Depoyu klonlamak icin kullanicidan
    `core.longpaths` ayarlamasini istemek kabul edilemez - depo kendi
    basina calismali.

    Cozum: dosya adi goreli yolun SHA-1 ozetinden turetilir. Eslestirme
    tablosuna gerek yok; `data_loader.gorsel_coz()` ayni hesabi yapar.
    """
    ozet = hashlib.sha1(goreli.as_posix().encode("utf-8")).hexdigest()[:16]
    return kok / goreli.parts[0] / f"{ozet}{goreli.suffix.lower()}"


# Demonun Hata Analizi bolumundeki siralama olcutleri. Buradaki liste
# demo/bolumler/hata_analizi.py::SIRALAMA ile ayni alanlari kullanir.
SIRALAMA_ALANLARI = (
    ("score", True),
    ("false_negatives", True),
    ("false_positives", True),
    ("mean_iou", False),          # dusuk IoU: kucukten buyuge
)


def _kucult(kaynak: Path, hedef: Path, en_fazla_kenar: int, kalite: int) -> int:
    """Goruntuyu kucultup kaydeder; bayt cinsinden yeni boyutu dondurur.

    Pillow yoksa dosya oldugu gibi kopyalanir - set buyur ama calisir.
    """
    hedef.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ModuleNotFoundError:
        shutil.copy2(kaynak, hedef)
        return hedef.stat().st_size

    with Image.open(kaynak) as gorsel:
        gorsel = gorsel.convert("RGB")
        if max(gorsel.size) > en_fazla_kenar:
            oran = en_fazla_kenar / max(gorsel.size)
            gorsel = gorsel.resize(
                (int(gorsel.width * oran), int(gorsel.height * oran))
            )
        gorsel.save(hedef, "JPEG", quality=kalite, optimize=True)
    return hedef.stat().st_size


def _galeri_kareleri(kayitlar: list[dict], adet: int) -> list[str]:
    """Dort siralama olcutunun ilk `adet` karesinin birlesimi."""
    secilen: list[str] = []
    for alan, ters in SIRALAMA_ALANLARI:
        uygun = [k for k in kayitlar if alan in k and k.get("image")]
        for kayit in sorted(uygun, key=lambda k: k[alan], reverse=ters)[:adet]:
            if kayit["image"] not in secilen:
                secilen.append(kayit["image"])
    return secilen


def hazirla(adet: int = 8, en_fazla_kenar: int = 1100, kalite: int = 82) -> dict:
    if HEDEF.exists():
        shutil.rmtree(HEDEF)

    toplam_bayt = 0
    kare_sayisi = 0
    galeri_sayisi = 0

    # Once her galerinin kendi secimi yapilir.
    secim: dict[Path, list[str]] = {}
    kaynak_adlari: set[str] = set()
    for manifest in sorted(RAPORLAR.glob("hata_galerisi_*/gallery.json")):
        klasor = manifest.parent
        kayitlar = json.loads(manifest.read_text(encoding="utf-8"))
        galeri_sayisi += 1
        secilenler = _galeri_kareleri(kayitlar, adet)
        secim[klasor] = secilenler
        secili = {k["image"] for k in kayitlar if k.get("image") in secilenler}
        kaynak_adlari |= {
            k.get("source") for k in kayitlar if k.get("image") in secili
        }

    # SONRA saglikli referansin galerisinden, baska bir galeride secilmis her
    # kareyi de ekle. Hata Analizi sayfasi ayni kareyi "saglikli model" ve
    # "senaryo modeli" olarak YAN YANA gosterir; secim galeri basina bagimsiz
    # yapilsaydi tam da o esleseme kopardi ve karsilastirma calismazdi.
    saglikli = RAPORLAR / "hata_galerisi_v00_saglikli"
    saglikli_manifest = saglikli / "gallery.json"
    if saglikli_manifest.is_file():
        kayitlar = json.loads(saglikli_manifest.read_text(encoding="utf-8"))
        mevcut = secim.setdefault(saglikli, [])
        for kayit in kayitlar:
            if kayit.get("source") in kaynak_adlari and kayit.get("image"):
                if kayit["image"] not in mevcut:
                    mevcut.append(kayit["image"])

    for klasor, secilenler in secim.items():
        for goreli in secilenler:
            kaynak = klasor / goreli
            if not kaynak.is_file():
                continue
            hedef = ayna_yolu(HEDEF, Path(klasor.name) / goreli)
            toplam_bayt += _kucult(kaynak, hedef, en_fazla_kenar, kalite)
            kare_sayisi += 1

    matris_sayisi = 0
    for kaynak in sorted(RAPORLAR.glob("*/gorseller/confusion_matrix.png")):
        goreli = kaynak.relative_to(RAPORLAR)
        # Confusion matrix bir grafik: JPEG'e cevirmek metni bulaniklastirir,
        # PNG olarak yalnizca kucultulur.
        hedef = ayna_yolu(HEDEF, goreli)
        hedef.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image

            with Image.open(kaynak) as gorsel:
                if max(gorsel.size) > en_fazla_kenar:
                    oran = en_fazla_kenar / max(gorsel.size)
                    gorsel = gorsel.resize(
                        (int(gorsel.width * oran), int(gorsel.height * oran))
                    )
                gorsel.save(hedef, "PNG", optimize=True)
        except ModuleNotFoundError:
            shutil.copy2(kaynak, hedef)
        toplam_bayt += hedef.stat().st_size
        matris_sayisi += 1

    ozet = {
        "galeri": galeri_sayisi,
        "kare": kare_sayisi,
        "confusion_matrix": matris_sayisi,
        "boyut_mb": round(toplam_bayt / 1_048_576, 1),
        "olcut": (f"siralama olcutu basina ilk {adet}, en fazla "
                  f"{en_fazla_kenar} px kenar"),
    }
    (HEDEF / "KAYNAK.json").write_text(
        json.dumps({
            "amac": "Sunum icin tasinabilir gorsel seti",
            "kaynak": "reports/ (Git disi, 233 MB)",
            "not": ("Bu bir ALT KUMEDIR ve KUCULTULMUSTUR. Olcumler hicbir "
                    "zaman bu gorsellerden uretilmez; goruntuler yalnizca "
                    "inceleme icindir. Yeniden uretmek icin: "
                    "python scripts/sunum_gorselleri_hazirla.py"),
            **ozet,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ozet


def main() -> None:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--adet", type=int, default=8,
                             help="siralama olcutu basina kare sayisi")
    ayristirici.add_argument("--kenar", type=int, default=1100,
                             help="en uzun kenar (px)")
    ayristirici.add_argument("--kalite", type=int, default=82,
                             help="JPEG kalitesi")
    args = ayristirici.parse_args()
    print(json.dumps(hazirla(args.adet, args.kenar, args.kalite),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

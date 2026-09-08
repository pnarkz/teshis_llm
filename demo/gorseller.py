"""Etiketli termal ornekler: gorsel bulma ve kutu cizme.

Neden dikkatli olmak gerekiyor
------------------------------
`val_diagnostic/` .gitignore'dadir - kilitli tani seti orijinal veri
setinden turetilir ve depoya girmez. Yani bu makinede calisan galeri, taze
bir klonda BOS kalir. Sunum sirasinda bos bir galeri, "veri yok" gibi
okunur.

Cozum iki katmanli:

1. Once yerel `val_diagnostic/` aranir (tam set, 1056 goruntu).
2. Bulunamazsa `demo/assets/ornekler/` altindaki tasinabilir kucuk set
   kullanilir ve arayuz bunun bir ALT KUME oldugunu soyler.

Ikisi de yoksa sayfa sessizce bos kalmaz; ne eksik oldugunu ve nasil
uretilecegini yazar.

Kutular CALISMA ZAMANINDA cizilir (Pillow). Onceden cizilmis PNG saklamak
yerine etiket dosyasindan cizmek, gosterilen kutunun gercekten olculen
etiket oldugunu garanti eder.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

KOK = Path(__file__).resolve().parents[1]
TANI_KOKU = KOK / "val_diagnostic"
YEDEK_KOKU = Path(__file__).resolve().parent / "assets/ornekler"

SINIF_ADI = {0: "taşıt", 1: "insan", 2: "UAP", 3: "UAI"}
# Sinif renkleri butun konsolda AYNI; stil.py paletiyle uyumlu.
SINIF_RENGI = {
    0: "#4c8dff",   # taşıt  - ana vurgu
    1: "#2dd4bf",   # insan  - camgöbeği
    2: "#e0a33e",   # UAP    - amber
    3: "#e5544b",   # UAI    - kırmızı
}

# Nesne boyut bantlari kirilim olcumuyle AYNI esikleri kullanir
# (teshis/degerlendirme/metrikler.py). Iki yerde farkli esik olursa galeri
# ile metrik tablosu birbirini tutmaz.
BANT_SINIRLARI = ((16, "cok_kucuk_16_alti"), (32, "kucuk_16_32"),
                  (64, "orta_32_64"), (10**9, "buyuk_64_ustu"))
BANT_ADI = {
    "cok_kucuk_16_alti": "çok küçük (<16 px)",
    "kucuk_16_32": "küçük (16-32 px)",
    "orta_32_64": "orta (32-64 px)",
    "buyuk_64_ustu": "büyük (>64 px)",
}


def _bant(genislik: float, yukseklik: float) -> str:
    olcu = (genislik * yukseklik) ** 0.5
    for sinir, ad in BANT_SINIRLARI:
        if olcu < sinir:
            return ad
    return "buyuk_64_ustu"


def kaynak_kokleri() -> tuple[Path | None, str]:
    """(kok dizin, kaynak turu) - 'tam' | 'yedek' | 'yok'."""
    if (TANI_KOKU / "images").is_dir() and any((TANI_KOKU / "images").iterdir()):
        return TANI_KOKU, "tam"
    if (YEDEK_KOKU / "images").is_dir() and any((YEDEK_KOKU / "images").iterdir()):
        return YEDEK_KOKU, "yedek"
    return None, "yok"


def durum() -> dict[str, Any]:
    """Galerinin hangi kaynaktan beslendigi - arayuz bunu acikca yazar."""
    kok, tur = kaynak_kokleri()
    adet = len(list((kok / "images").glob("*"))) if kok else 0
    return {
        "kaynak": tur,
        "dizin": str(kok.relative_to(KOK)) if kok else None,
        "görüntü": adet,
        "açıklama": {
            "tam": "Yerel kilitli tanı seti kullanılıyor (tam set).",
            "yedek": ("Kilitli tanı seti bu makinede yok; depoyla birlikte "
                      "gelen taşınabilir örnek alt kümesi gösteriliyor."),
            "yok": ("Ne yerel tanı seti ne de taşınabilir örnek seti "
                    "bulunabildi."),
        }[tur],
    }


def _kaynak_grubu(dosya_adi: str) -> str:
    """Kaynak grubu, veri katmaninin KENDI kuralindan gelir.

    Burada ayri bir kural yazmak cazipti ("__ oncesi") ama yanlis olurdu:
    `frame_008172_...` dosyalari ayirici tasimaz ve tf2026 grubuna aittir.
    Kirilim olcumleri istatistik.kaynak_adi kullanir; galeri baska bir kural
    kullansa galeri filtreleri metrik tablolariyla tutmazdi.
    """
    from teshis.veri.istatistik import kaynak_adi

    return kaynak_adi(dosya_adi)


@functools.lru_cache(maxsize=1)
def katalog() -> list[dict[str, Any]]:
    """Bulunabilen her ornegin etiket ozeti - filtreler bunun uzerinde calisir."""
    kok, tur = kaynak_kokleri()
    if kok is None:
        return []
    kayitlar = []
    for gorsel in sorted((kok / "images").glob("*")):
        if gorsel.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        etiket = kok / "labels" / f"{gorsel.stem}.txt"
        kutular = _etiket_oku(etiket)
        siniflar = sorted({SINIF_ADI.get(k["sinif"], "?") for k in kutular})
        bantlar = sorted({k["bant"] for k in kutular})
        kayitlar.append({
            "dosya": gorsel.name,
            "yol": gorsel,
            "kaynak": _kaynak_grubu(gorsel.name),
            "nesne": len(kutular),
            "siniflar": siniflar,
            "bantlar": bantlar,
            "kutular": kutular,
        })
    return kayitlar


def _etiket_oku(yol: Path) -> list[dict[str, Any]]:
    """YOLO formati: <sinif> <cx> <cy> <w> <h>, hepsi normalize."""
    if not yol.is_file():
        return []
    kutular = []
    for satir in yol.read_text(encoding="utf-8").splitlines():
        parca = satir.split()
        if len(parca) < 5:
            continue
        try:
            sinif = int(float(parca[0]))
            cx, cy, w, h = (float(p) for p in parca[1:5])
        except ValueError:
            continue
        kutular.append({"sinif": sinif, "cx": cx, "cy": cy, "w": w, "h": h,
                        "bant": "?"})
    return kutular


def kutulu_gorsel(kayit: dict[str, Any], en_fazla_kenar: int = 900):
    """Etiket kutulari cizilmis PIL goruntusu dondurur.

    Pillow yoksa None doner; cagiran taraf ham goruntuyu gosterir. Sunum
    makinesinde bir bagimlilik eksik diye sayfa cokmemelidir.
    """
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError:
        return None

    try:
        gorsel = Image.open(kayit["yol"]).convert("RGB")
    except OSError:
        return None

    G, Y = gorsel.size
    if max(G, Y) > en_fazla_kenar:
        oran = en_fazla_kenar / max(G, Y)
        gorsel = gorsel.resize((int(G * oran), int(Y * oran)))
        G, Y = gorsel.size

    ciz = ImageDraw.Draw(gorsel)
    for k in kayit["kutular"]:
        x1 = (k["cx"] - k["w"] / 2) * G
        y1 = (k["cy"] - k["h"] / 2) * Y
        x2 = (k["cx"] + k["w"] / 2) * G
        y2 = (k["cy"] + k["h"] / 2) * Y
        renk = SINIF_RENGI.get(k["sinif"], "#ffffff")
        ciz.rectangle([x1, y1, x2, y2], outline=renk, width=2)
        etiket = SINIF_ADI.get(k["sinif"], "?")
        # Etiket serit zemini: ince kutularda metin okunmaz kaliyordu.
        ciz.rectangle([x1, max(0, y1 - 13), x1 + 7 * len(etiket) + 6, y1],
                      fill=renk)
        ciz.text((x1 + 3, max(0, y1 - 12)), etiket, fill="#08111f")
    return gorsel


def boyut_bantlarini_doldur(kayitlar: list[dict], varsayilan_kenar: int = 640):
    """Kutulara piksel boyut bandi yazar.

    Bant, kutunun GERCEK piksel olcusune gore hesaplanir; bu yuzden goruntu
    boyutu gerekir. Goruntuyu acmak pahali oldugundan yalnizca gosterilecek
    kayitlar icin cagrilir.
    """
    try:
        from PIL import Image
    except ModuleNotFoundError:
        Image = None
    for kayit in kayitlar:
        if Image is not None:
            try:
                G, Y = Image.open(kayit["yol"]).size
            except OSError:
                G = Y = varsayilan_kenar
        else:
            G = Y = varsayilan_kenar
        for k in kayit["kutular"]:
            k["bant"] = _bant(k["w"] * G, k["h"] * Y)
        kayit["bantlar"] = sorted({k["bant"] for k in kayit["kutular"]})
    return kayitlar


def tasinabilir_set_olustur(adet_sinif_basina: int = 3,
                            hedef: Path = YEDEK_KOKU) -> dict[str, Any]:
    """Yerel tani setinden kucuk, tasinabilir bir ornek seti kopyalar.

    Amac: taze bir klonda galeri bos kalmasin. Nadir siniflar (UAP/UAI) az
    goruntude gectigi icin secim SINIF BASINA yapilir, rastgele degil -
    yoksa 12 goruntunun hepsi insan/tasit cikar ve galeri o iki sinifi
    gosterir.

    Betik olarak calistirilir: python demo/gorseller.py
    """
    import shutil

    kok, tur = kaynak_kokleri()
    if kok is None or tur != "tam":
        return {"durum": "kaynak yok", "kopyalanan": 0}

    secilen: dict[int, list[dict]] = {s: [] for s in SINIF_ADI}
    for kayit in katalog():
        for sinif in {k["sinif"] for k in kayit["kutular"]}:
            if sinif in secilen and len(secilen[sinif]) < adet_sinif_basina:
                secilen[sinif].append(kayit)

    (hedef / "images").mkdir(parents=True, exist_ok=True)
    (hedef / "labels").mkdir(parents=True, exist_ok=True)
    kopyalanan = []
    for kayitlar in secilen.values():
        for kayit in kayitlar:
            ad = kayit["dosya"]
            if ad in kopyalanan:
                continue
            shutil.copy2(kayit["yol"], hedef / "images" / ad)
            etiket = kok / "labels" / f"{Path(ad).stem}.txt"
            if etiket.is_file():
                shutil.copy2(etiket, hedef / "labels" / etiket.name)
            kopyalanan.append(ad)

    (hedef / "KAYNAK.json").write_text(
        json.dumps({
            "amac": "Sunum icin tasinabilir ornek alt kumesi",
            "kaynak": "val_diagnostic (kilitli tani seti)",
            "secim": f"sinif basina en fazla {adet_sinif_basina} goruntu",
            "not": ("Bu bir ALT KUMEDIR; metrikler her zaman tam kilitli set "
                    "uzerinde olculur. Galeri yalnizca gorsel ornekleme icindir."),
            "goruntu": len(kopyalanan),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"durum": "tamam", "kopyalanan": len(kopyalanan), "hedef": str(hedef)}


if __name__ == "__main__":
    print(tasinabilir_set_olustur())

"""Gorsel kanit secimi: hangi kare BU senaryonun bozulmasini gosterir.

Sorun
-----
Hata galerileri (`teshis/degerlendirme/hata_galerisi.py`) kareleri senaryodan
BAGIMSIZ bir zorluk skoruyla siralar:

    score = FN + FP + (1 - ortalama IoU)

Bu skor kalabalik kareleri odullendirir. Sonuc olcuIdu: **26 galerinin
24'unde en ustteki kare ayni** (`hituav__1_130_30_0_03841.jpg`). Konsol her
senaryoda o kareyi gosteriyordu; yani "gorsel kanit" bolumu senaryolar
arasinda hicbir sey ayirt etmiyordu. D3b (tasit/insan karisikligi) icin
gosterilen kare, karisikligi degil yalnizca sahnenin kalabalik oldugunu
gosteriyordu.

Iki ayri sey eksikti ve ikisi de burada kapatilir.

1. IMZA FILTRESI - "bu kare bu bozulmayi GOSTEREBILIR mi?"
   Senaryonun hangi sinifa, hangi boyut bandina veya hangi kaynaga
   dokundugu zaten `senaryolar/*/*.yaml` icinde yaziyor. Burada yeni bir
   tablo YAZILMAZ; imza o dosyalardan turetilir. Elle yazilmis ikinci bir
   senaryo tablosu, projenin tekrarlayan hatasi olurdu: ayni kural iki
   yerde yasarsa biri geride kalir.

2. FARK SIRALAMASI - "bu karede senaryo, referanstan FARKLI davrandi mi?"
   Mutlak hata skoru "bu kare zor" der; kanit icin gereken ise "bu kare
   referansta calisiyordu, bu kosuda bozuldu"dur. Bu yuzden siralama
   kosunun KENDI referansina gore FARKA bakar:

       delta = (FN - FN_ref) + (FP - FP_ref) + (IoU_ref - IoU)

   Iki model de ayni olcude basarisiz oluyorsa delta ~ 0 olur ve o kare
   listenin altina duser. Kalabalik "herkeste ayni" kareler tam olarak
   boyle elenir.

Durustluk notu
--------------
Imza filtresi her zaman uygulanamaz:

- D3'un imzasi UAP/UAI'dir; bu iki sinif kilitli tani setinde yalnizca 15
  ve 17 kutuyla temsil edilir ve o kareler genel zorluk skoruna gore ilk
  50'ye HIC girmez. Yani D3 icin imzaya uyan kare galeride yoktur.
- Taze bir klonda kilitli tani seti bulunmaz; etiket okunamadigi icin imza
  hesaplanamaz.

Bu durumlarda secim sessizce genel siralamaya DUSMEZ: `olcut` alani
`"genel"` olur ve `not` alani nedenini yazar. Arayuz bunu ekranda gosterir.
Uymayan bir kareyi kanit diye sunmak, projenin butun metodolojisiyle
celisirdi.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

KOK = Path(__file__).resolve().parents[1]
SENARYOLAR = KOK / "senaryolar"

# Senaryo YAML'inde imza tasiyan parametre adlari. `sinif_ciftii` bir yazim
# hatasidir ama D3b'nin dosyasinda BOYLE duruyor; ikisi de okunur, boylece
# yazim ileride duzeltilirse filtre sessizce bozulmaz.
SINIF_ANAHTARLARI = ("siniflar", "sinif_cifti", "sinif_ciftii", "sinif_id")
BOYUT_ANAHTARI = "etkin_sqrt_alan_esigi_px"
KAYNAK_ANAHTARI = "izinli_kaynaklar"


@functools.lru_cache(maxsize=1)
def _konfigler() -> dict[str, dict[str, Any]]:
    """Senaryo kodu -> senaryo YAML'i. Yol katalog.yaml'den gelir."""
    import katalog as demo_katalog

    cikti: dict[str, dict[str, Any]] = {}
    for kayit in demo_katalog._ham_katalog():
        yol = SENARYOLAR / str(kayit.get("config", ""))
        if not yol.is_file():
            continue
        cikti[str(kayit["kod"])] = yaml.safe_load(yol.read_text(encoding="utf-8")) or {}
    return cikti


@functools.lru_cache(maxsize=64)
def imza(kod: str) -> dict[str, Any]:
    """Senaryonun gorsel imzasi - tamamen senaryo YAML'inden turetilir.

    Doner: `{"siniflar": [...], "hepsi_gerekli": bool, "boyut_esigi_px": n,
    "disari_kaynaklar": [...], "aciklama": "..."}`. Imza tasimayan
    senaryolarda (D2a, D2b, D6a, D6b, E serisi) bos sozluk doner: bu
    bozulmalar belirli bir sinifa/banda/kaynaga bagli degildir, bu yuzden
    filtrelenecek bir sey de yoktur.
    """
    parametreler = (_konfigler().get(kod) or {}).get("parametreler") or {}

    siniflar: list[int] = []
    for anahtar in SINIF_ANAHTARLARI:
        if anahtar in parametreler:
            deger = parametreler[anahtar]
            siniflar = [int(d) for d in (deger if isinstance(deger, list) else [deger])]
            break

    # Sinif TAKASI iki sinifi birbirine karistirir; karisikligin gorunmesi
    # icin karede her iki sinifin da bulunmasi gerekir. Sinif YETERSIZLIGI
    # (D1) tek sinifi seyreltir; orada tek sinifin varligi yeter.
    hepsi_gerekli = len(siniflar) > 1

    boyut = parametreler.get(BOYUT_ANAHTARI)
    izinli = parametreler.get(KAYNAK_ANAHTARI) or []

    if not (siniflar or boyut or izinli):
        return {}

    from gorseller import SINIF_ADI

    parcalar = []
    if siniflar:
        adlar = " ve ".join(SINIF_ADI.get(s, str(s)) for s in siniflar)
        parcalar.append(
            f"karede {adlar} sınıflarının {'hepsi' if hepsi_gerekli else 'biri'} bulunmalı"
        )
    if boyut:
        parcalar.append(f"karede {boyut} pikselin altında en az bir nesne bulunmalı")
    if izinli:
        parcalar.append(
            "kare, eğitimde tutulan kaynaktan (" + ", ".join(izinli) + ") gelmemeli"
        )

    return {
        "siniflar": siniflar,
        "hepsi_gerekli": hepsi_gerekli,
        "boyut_esigi_px": float(boyut) if boyut else None,
        "izinli_kaynaklar": list(izinli),
        "aciklama": "; ".join(parcalar),
    }


@functools.lru_cache(maxsize=1)
def _kare_ozetleri() -> dict[str, dict[str, Any]]:
    """Dosya adi -> {siniflar, en_kucuk_px, kaynak}. Kilitli tani setinden.

    Boyut, olcum tarafiyla AYNI fonksiyondan (metrikler.etkin_sqrt_alan)
    hesaplanir. Galeri kendi esigini yazsaydi, "16 px altinda" diye
    gosterilen kare ile D4'un egitimden cikardigi kutular ortusmeyebilirdi.
    """
    import gorseller
    from teshis.degerlendirme.metrikler import etkin_sqrt_alan

    try:
        from PIL import Image
    except ModuleNotFoundError:
        Image = None

    ozet: dict[str, dict[str, Any]] = {}
    for kayit in gorseller.katalog():
        boyutlar = []
        if Image is not None and kayit["kutular"]:
            try:
                G, Y = Image.open(kayit["yol"]).size
            except OSError:
                G = Y = 0
            if G and Y:
                boyutlar = [etkin_sqrt_alan(k["w"], k["h"], G, Y)
                            for k in kayit["kutular"]]
        ozet[kayit["dosya"]] = {
            "siniflar": {k["sinif"] for k in kayit["kutular"]},
            "en_kucuk_px": min(boyutlar) if boyutlar else None,
            "kaynak": kayit["kaynak"],
        }
    return ozet


def _imzaya_uyuyor(ozet: dict[str, Any] | None, im: dict[str, Any]) -> bool:
    if not im or ozet is None:
        return False
    siniflar = im.get("siniflar") or []
    if siniflar:
        varlik = ozet["siniflar"] & set(siniflar)
        if im.get("hepsi_gerekli"):
            if len(varlik) < len(set(siniflar)):
                return False
        elif not varlik:
            return False
    esik = im.get("boyut_esigi_px")
    if esik is not None:
        en_kucuk = ozet.get("en_kucuk_px")
        if en_kucuk is None or en_kucuk >= esik:
            return False
    izinli = im.get("izinli_kaynaklar") or []
    if izinli and ozet.get("kaynak") in izinli:
        return False
    return True


def _delta(kayit: dict[str, Any], referans: dict[str, Any]) -> float:
    """Kosunun referansa gore bu karede NE KADAR fazla hata yaptigi."""
    fn = (kayit.get("false_negatives") or 0) - (referans.get("false_negatives") or 0)
    fp = (kayit.get("false_positives") or 0) - (referans.get("false_positives") or 0)
    iou = (referans.get("mean_iou") or 0.0) - (kayit.get("mean_iou") or 0.0)
    return fn + fp + iou


def yon(delta: float) -> str:
    """Farkin yonu - projenin yon ayrimi kuralinin kare duzeyindeki karsiligi.

    Bir metrigin beklenenin TERSINE hareket etmesi bozulma kanidi sayilmaz
    (bkz. `senaryo_ozeti.asan_yone_gore`). Ayni kural burada da gecerlidir:
    kosunun referanstan DAHA AZ hata yaptigi bir kare, o bozulmanin kaniti
    olarak sunulamaz. Gizlenmez de - ayrica soylenir.
    """
    if delta > 1e-9:
        return "bozulma"
    if delta < -1e-9:
        return "iyileşme"
    return "fark yok"


def adaylar(kosu: str) -> dict[str, Any]:
    """Bir kosu icin sirali gorsel kanit adaylari.

    Yalnizca referans galerisinde de bulunan kareler dondurulur; kanit "ayni
    kare, iki model" karsilastirmasidir ve tek tarafli bir kare bunu
    kuramaz.
    """
    from data_loader import error_galleries, referans_galerisi
    import katalog as demo_katalog

    galeriler = error_galleries()
    galeri = galeriler.get(kosu)
    if not galeri:
        return {"kayitlar": [], "olcut": "yok", "not": "", "imza": {},
                "referans_adi": None, "referans_galerisi": {}}

    ref_ad, ref_galeri = referans_galerisi(kosu)
    referanslar = {e.get("source"): e for e in (ref_galeri.get("entries") or [])}

    eslesen = []
    for kayit in galeri["entries"]:
        es = referanslar.get(kayit.get("source"))
        if es is None:
            continue
        fark = _delta(kayit, es)
        eslesen.append({**kayit, "referans": es, "delta": fark, "yon": yon(fark)})

    sonuc = {
        "kayitlar": [],
        "olcut": "genel",
        "not": "",
        "imza": {},
        "referans_adi": ref_ad,
        "referans_galerisi": ref_galeri,
        "galeri": galeri,
        "eslesen": len(eslesen),
        # Referans kendisiyle karsilastirilamaz; her delta sifir cikar ve
        # "bozulma yok" diye okunmasi yanlis olurdu.
        "kendi_referansi": ref_ad == kosu,
    }
    if not eslesen:
        sonuc["olcut"] = "yok"
        return sonuc

    eslesen.sort(key=lambda k: (k["delta"], k.get("score") or 0), reverse=True)

    im = imza(demo_katalog._kod(kosu))
    sonuc["imza"] = im
    if not im:
        sonuc["not"] = (
            "Bu bozulma belirli bir sınıfa, boyut bandına veya kaynağa bağlı "
            "değildir; kareler yalnızca referanstan farka göre sıralanır."
        )
        sonuc["kayitlar"] = eslesen
        return sonuc

    import gorseller

    ozetler = _kare_ozetleri()
    uyan = [k for k in eslesen if _imzaya_uyuyor(ozetler.get(k.get("source")), im)]
    if uyan:
        sonuc["olcut"] = "imza"
        sonuc["kayitlar"] = uyan
        sonuc["not"] = (
            f"Kareler bu senaryonun imzasına göre süzüldü ({im['aciklama']}); "
            f"{len(eslesen)} eşleşen kareden {len(uyan)} tanesi uyuyor."
        )
        return sonuc

    sonuc["kayitlar"] = eslesen
    if gorseller.durum()["kaynak"] != "tam":
        sonuc["not"] = (
            "Kilitli tanı seti bu makinede yok; karelerin etiketi okunamadığı "
            "için senaryo imzası (" + im["aciklama"] + ") uygulanamadı. "
            "Sıralama yalnızca referanstan farka göre yapıldı."
        )
    else:
        sonuc["not"] = (
            "Bu senaryonun imzasına (" + im["aciklama"] + ") uyan kare "
            "galeride yok: galeriler senaryodan bağımsız bir zorluk skoruyla "
            "üretildiği için nadir sınıfların kareleri ilk 50'ye girmiyor. "
            "Gösterilen kare yalnızca referanstan farka göre seçilmiştir ve "
            "bu bozulmanın doğrudan kanıtı <b>değildir</b> — bu senaryonun "
            "kanıtı aşağıdaki karışıklık farkı grafiğidir."
        )
    return sonuc

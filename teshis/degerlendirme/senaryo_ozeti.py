"""Bir senaryonun deney ozetini kaynak dosyalardan toplar.

Demo, senaryo aciklamalarini 24 girdilik elle tutulan bir sozlukte
tasiyordu. Her yeni senaryoda geride kaliyordu ve gercekten geride kaldi:
D6a, D6b, v00n ve D1n eklendiginde demo onlari sessizce eksik gosterdi.

Burada tersi yapilir - ozetin bes bileseninden dordu **turetilir**:

| Soru | Kaynak |
|---|---|
| Ne degistirildi? | senaryolar/<tur>/<kod>_*.yaml (parametreler) |
| Ne sabit tutuldu? | ayni dosya (kaynak_surum, hedef_split) + protokol |
| Beklenen etki neydi? | ayni dosya (beklenen_kanit) |
| Ne gozlendi? | results.csv + reports/ (referansa gore farklar) |
| Kanit ne kadar guclu? | gurultu.py bandi (asiyor mu, kac kosudan) |

Elle yazilan tek alan senaryonun NE OLCTUGU: `senaryolar/anlatim.yaml`.
Bu, turetilemeyecek tek bilgidir - arastirma niyeti veride yazmaz.
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path
from typing import Any

import yaml

KOK = Path(__file__).resolve().parents[2]
SENARYOLAR = KOK / "senaryolar"
RESULTS_CSV = KOK / "results.csv"

GENEL_METRIKLER = ("mAP50", "mAP50_95", "precision", "recall")


@functools.lru_cache(maxsize=1)
def _katalog() -> dict[str, dict]:
    ham = yaml.safe_load((SENARYOLAR / "katalog.yaml").read_text(encoding="utf-8"))
    return {s["kod"]: s for s in ham.get("senaryolar", [])}


@functools.lru_cache(maxsize=1)
def _anlatim() -> dict[str, str]:
    yol = SENARYOLAR / "anlatim.yaml"
    if not yol.is_file():
        return {}
    return yaml.safe_load(yol.read_text(encoding="utf-8")) or {}


@functools.lru_cache(maxsize=1)
def _defter() -> dict[str, dict[str, str]]:
    with RESULTS_CSV.open(encoding="utf-8") as f:
        return {s["scenario"]: s for s in csv.DictReader(f)}


def _kod(senaryo: str) -> str:
    """'D2b final_best' -> 'D2b', 'C2 seed13' -> 'C2', 'E1 last_pt' -> 'E1'."""
    return senaryo.split()[0]


def _konfig(senaryo: str) -> dict[str, Any]:
    kayit = _katalog().get(_kod(senaryo))
    if not kayit:
        return {}
    yol = SENARYOLAR / kayit["config"]
    if not yol.is_file():
        return {}
    return yaml.safe_load(yol.read_text(encoding="utf-8")) or {}


def ne_degisti(senaryo: str) -> dict[str, Any]:
    """Bozulma turu ve uygulanan parametreler."""
    kayit = _katalog().get(_kod(senaryo), {})
    konfig = _konfig(senaryo)
    satir = _defter().get(senaryo, {})
    return {
        "kod": _kod(senaryo),
        "tur": kayit.get("tur"),          # veri | egitim
        "parametreler": konfig.get("parametreler") or {},
        "veri_surumu": satir.get("data_version"),
        "hedef_split": konfig.get("hedef_split"),
    }


def ne_sabit_kaldi(senaryo: str) -> list[str]:
    """Aday ile REFERANSI arasinda sabit tutulan alanlar.

    Bunlari yazmak sekil degil: karsilastirmanin gecerliligi tam olarak bu
    listenin dogru olmasina baglidir. Liste kosunun KENDI degerlerini
    gosterir; referansin ayni degerleri tasidigi karsilastirilabilirlik.py
    tarafindan garanti edilir. Tek istisna, kosunun kasitli olarak farkli
    olan alanidir (D6a'nin degerlendirme kumesi gibi) - o zaman satir bunu
    acikca soyler.
    """
    satir = _defter().get(senaryo, {})
    konfig = _konfig(senaryo)
    # Degerlendirme seti SABIT YAZILAMAZ: D6a kasitli olarak sizintili kume
    # uzerinde olculur. Sabit yazildiginda sayfa kendi kendisiyle celisiyordu -
    # ust kutuda "kilitli set hic degismez", alt kutuda "baska sette olculdu".
    from .karsilastirilabilirlik import karsilastirma, kimlik

    # Referanstan FARKLI olan alan "sabit" diye yazilamaz. E4 512 px'te
    # olculur ve referansi 768 px'tir; liste "Cikarim cozunurlugu: 512 px"
    # deyip bunu sabitler arasinda gosteriyordu - oysa DEGISEN alan tam
    # olarak oydu. Fark, kosunun kendi kimligi ile referansinkinin
    # karsilastirilmasindan turetilir.
    # `referans_senaryo` yalnizca AYNI olcekteki referansi bulur ve eslenik
    # olcumlerde (E4, D6a) None doner - oysa DEGISEN alani isaretlemek tam
    # da o kosularda gerekli. `karsilastirma` her iki durumu da kapsar.
    ref_ad = karsilastirma(senaryo).get("referans")
    ben, ref = kimlik(senaryo), (kimlik(ref_ad) if ref_ad else None)
    if ben is None:
        return []

    def isaret(alan: str, metin: str) -> str:
        if ref is None or getattr(ben, alan) == getattr(ref, alan):
            return metin
        return f"{metin} — **DEĞİŞEN** (referansta: {getattr(ref, alan)})"

    kume = satir.get("evaluation_set", "?")
    sabitler = [
        isaret("degerlendirme_seti",
               f"Değerlendirme seti: {kume}"
               + (" (kilitli)" if kume == "val_diagnostic"
                  else " — kilitli set DEĞİL")),
        isaret("imgsz_eval", f"Çıkarım çözünürlüğü: {satir.get('imgsz_eval', '?')} px"),
        isaret("model", f"Başlangıç modeli: {satir.get('model', '?')}"),
        f"Seed: {satir.get('seed', '?')}",
        isaret("checkpoint",
               f"Checkpoint: {'last.pt' if str(satir.get('weights_path', '')).endswith('last.pt') else 'best.pt'}"),
    ]
    if konfig.get("hedef_split") == "train":
        sabitler.append("Yalnızca train bölümü değiştirildi; val ve test dokunulmadı")
    if _katalog().get(_kod(senaryo), {}).get("tur") == "veri":
        sabitler.append("Eğitim protokolü sabit (senaryolar/egitim_protokolu.yaml)")
    return sabitler


def ne_gozlendi(senaryo: str) -> dict[str, Any]:
    """Kendi olcegindeki referansa gore farklar ve o olcegin gurultu esigi.

    Referans SABIT DEGILDIR. Her kosu, dort kimlik alani (model, degerlendirme
    kumesi, cozunurluk, checkpoint) kendisiyle ayni olan saglikli kosuyla
    karsilastirilir; esik de yalnizca o olcegin kontrol kosularindan gelir.
    Ayrinti ve gerekce: karsilastirilabilirlik.py.
    """
    from .karsilastirilabilirlik import esikler as _grup_esikleri
    from .karsilastirilabilirlik import karsilastirma

    satir = _defter().get(senaryo)
    if satir is None:
        return {}

    karsi = karsilastirma(senaryo)
    ref_ad = karsi["referans"]
    ref = _defter().get(ref_ad) if ref_ad else None
    grup_esik = _grup_esikleri(senaryo)

    metrikler = {}
    for m in GENEL_METRIKLER:
        deger = float(satir[m])
        ref_deger = float(ref[m]) if ref else None
        fark = None if ref_deger is None else deger - ref_deger
        esik = grup_esik.get(m)
        metrikler[m] = {
            "deger": round(deger, 4),
            "referans": None if ref_deger is None else round(ref_deger, 4),
            "fark": None if fark is None else round(fark, 4),
            "gurultu_esigi": round(esik, 4) if esik is not None else None,
            # "asiyor" BUYUKLUK sorusudur: fark, bozulmasiz kosular arasindaki
            # yayilimin disinda mi? Yon ayri tutulur - ikisi karistirilirsa
            # YUKSELEN bir metrik bozulma kaniti gibi okunur.
            "asiyor": (abs(fark) > esik) if (fark is not None and esik) else None,
            "yon": None if not fark else ("dusus" if fark < 0 else "yukselis"),
        }
    asan = [m for m, d in metrikler.items() if d["asiyor"]]
    return {
        "metrikler": metrikler,
        "kontrol_kosu_sayisi": len(karsi["kontroller"]),
        "asan_metrikler": asan,
        # Esigi asan farklar YONE gore ayrilir.
        #
        # Bir metrigin beklenenin TERSINE yukselmesi bozulma kaniti degildir.
        # D1'de mAP50_95 +0.0240 yukselmisti (esik 0.0201) ve bu, sayfanin
        # hukum cumlesinde "mAP50_95 gurultu esigini asiyor" diye, kanit
        # gucunde ise "zayif bulgu" diye gorunuyordu - ikisi de dususmus gibi.
        # Ayni hata once hipotez tablosunda, sonra etki haritasinda cikti;
        # bu, kuralin yasadigi UCUNCU yerdi.
        "asan_dusen": [m for m in asan if metrikler[m]["yon"] == "dusus"],
        "asan_yukselen": [m for m in asan if metrikler[m]["yon"] == "yukselis"],
        "referans_senaryo": ref_ad,
        "karsilastirma_turu": karsi["tur"],
        "karsilastirma_aciklamasi": karsi["aciklama"],
        "kimlik": karsi["kimlik"]._asdict() if karsi["kimlik"] else None,
    }


def kanit_gucu(senaryo: str) -> dict[str, Any]:
    """Bulgunun ne kadar guclu oldugunu tek bakista soyler.

    Onemli: "guclu / zayif / gurultu icinde" derecelendirmesi YALNIZCA kendi
    olceginde bir referansi VE bir gurultu esigi olan kosular icin anlamlidir.
    Digerleri derecelendirilmez - derecelendirilirse saglikli bir kontrol
    kosusu "guclu bozulma kaniti" gorunur. Tam olarak bu olmustu:
    `v00_saglikli last_pt` uc metrikte "esigi asiyor" diye isaretlenmisti,
    oysa o kosuda bozulma yok, yalnizca checkpoint farkli.
    """
    from .karsilastirilabilirlik import bozulmasiz_mi

    gozlem = ne_gozlendi(senaryo)
    if not gozlem:
        return {"seviye": "olcum yok", "aciklama": "Bu koşu defterde bulunamadı."}

    tur = gozlem["karsilastirma_turu"]
    if tur == "yok":
        return {"seviye": "karsilastirilamaz", "aciklama": gozlem["karsilastirma_aciklamasi"]}
    if senaryo == gozlem["referans_senaryo"]:
        return {
            "seviye": "referans",
            "aciklama": (
                "Bu koşu kendi ölçeğinin sağlıklı referansıdır; kendisiyle "
                "karşılaştırılamaz."
            ),
        }
    if tur == "eslenik":
        return {"seviye": "eslenik olcum", "aciklama": gozlem["karsilastirma_aciklamasi"]}

    # Derecelendirme YALNIZCA esigi asan DUSUSLERE bakar. Bir metrigin
    # beklenenin tersine yukselmesi bozulmanin kaniti degildir; ayri
    # anlatilir.
    asan = gozlem["asan_dusen"]
    yukselen = gozlem["asan_yukselen"]
    n = gozlem["kontrol_kosu_sayisi"]

    # Kontrol kosusu OLCUM ARACIDIR, olcum nesnesi degil. Derecelendirilirse
    # kendi kendini "bozulma" sanir: kosu bandin disinda birakildiginda kalan
    # iki gozlemin araligi daralir ve kosu "uc deger" gorunur. C2 seed21 tam
    # boyle "guclu bozulma kaniti" cikmisti - icinde hicbir bozulma yokken.
    if bozulmasiz_mi(senaryo):
        hepsi = gozlem["asan_metrikler"]
        ek = (
            " Aynı ölçütle tartılsaydı şu metriklerde 'eşiği aşıyor' "
            f"çıkardı: {', '.join(hepsi)}. Bu, ölçütün kendisinin ne kadar "
            "oynak olduğunu gösterir." if hepsi else
            " Aynı ölçütle tartıldığında hiçbir metrikte eşiği aşmıyor."
        )
        return {
            "seviye": "kontrol kosusu",
            "aciklama": (
                "Bu koşu hiçbir bozulma içermez; gürültü tabanını ölçmek için "
                "vardır. Kendisi bir bulgu olarak derecelendirilmez." + ek
            ),
        }

    if n == 0:
        return {
            "seviye": "esik yok",
            "aciklama": (
                f"Referans var ({gozlem['referans_senaryo']}) ama bu ölçekte "
                "hiç kontrol koşusu yok. Fark ölçülebiliyor, gürültüden ayırt "
                "edilemiyor: başka bir ölçeğin eşiği ödünç alınamaz."
            ),
        }
    ters_not = (
        f" Ayrıca {', '.join(yukselen)} eşiği aşan bir YÜKSELİŞ gösteriyor: "
        "beklenen yönün tersi, bozulma kanıtı değil." if yukselen else ""
    )
    if not asan:
        return {
            "seviye": "gurultu icinde",
            "aciklama": (
                "Hiçbir genel metrik, bozulmasız koşular arasında gözlenen "
                f"yayılımı aşan bir DÜŞÜŞ göstermiyor ({n} kontrol koşusu). "
                "Bu senaryonun genel metriklerine dayanan bir iddia "
                "kurulamaz." + ters_not
            ),
        }
    if len(asan) == 1:
        return {
            "seviye": "zayif",
            "aciklama": (
                f"Yalnızca {asan[0]} gürültü eşiğini aşan bir düşüş gösteriyor. "
                "Tek metriğe dayanan bir bulgu, kırılım kanıtıyla "
                "desteklenmedikçe zayıftır." + ters_not
            ),
        }
    return {
        "seviye": "guclu",
        "aciklama": (
            f"{len(asan)} genel metrik gürültü eşiğini aşan bir düşüş "
            f"gösteriyor: {', '.join(asan)}." + ters_not
        ),
    }


def sinirlamalar(senaryo: str) -> list[str]:
    """Bu senaryo icin gecerli, VERIDEN turetilen sinirlar."""
    from .bootstrap import VAL_DIAGNOSTIC_BBOX_N

    satir = _defter().get(senaryo, {})
    sinirlar = []

    az = [f"{ad} (n={n})" for ad, n in VAL_DIAGNOSTIC_BBOX_N.items() if n < 30]
    if az:
        sinirlar.append(
            f"Düşük örnekli sınıflar: {', '.join(az)}. Bu sınıflardaki oranlar "
            "genellenemez."
        )
    gozlem = ne_gozlendi(senaryo)
    if gozlem:
        sinirlar.append(gozlem["karsilastirma_aciklamasi"])
        n = gozlem["kontrol_kosu_sayisi"]
        if n:
            sinirlar.append(
                f"Gürültü eşiği {n} kontrol koşusundan hesaplandı; az gözlemle "
                "eşik gerçek yayılımı olduğundan küçük gösterir."
            )
    if satir.get("evaluation_set") != "val_diagnostic":
        sinirlar.append(
            f"Bu koşu kilitli set yerine '{satir.get('evaluation_set')}' üzerinde "
            "ölçüldü; kilitli sette ölçülen koşularla aynı tabloda okunamaz."
        )
    if not satir.get("weights_path", "").endswith("best.pt"):
        sinirlar.append(
            "Bu satır last.pt checkpoint'ine aittir; best.pt satırıyla birlikte "
            "okunmalıdır."
        )
    return sinirlar


def ozet(senaryo: str) -> dict[str, Any]:
    """Senaryo sayfasinin tum icerigi, tek cagriyla."""
    return {
        "senaryo": senaryo,
        "ne_olcuyor": _anlatim().get(_kod(senaryo)),
        "ne_degisti": ne_degisti(senaryo),
        "ne_sabit_kaldi": ne_sabit_kaldi(senaryo),
        "beklenen_etki": _konfig(senaryo).get("beklenen_kanit"),
        "ne_gozlendi": ne_gozlendi(senaryo),
        "kanit_gucu": kanit_gucu(senaryo),
        "sinirlamalar": sinirlamalar(senaryo),
    }

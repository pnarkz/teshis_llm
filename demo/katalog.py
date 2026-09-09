"""SENARYO ile KOSU ayrimi - konsolun bilgi mimarisinin temeli.

Sorun
-----
Konsol 26 kosuyu esit agirlikta kart olarak gosteriyordu. Ama bunlar ayni
seviyede kavramlar degil:

| Kavram | Ornek | Nedir |
|---|---|---|
| **Senaryo** | D4 | Arastirilan bilimsel hipotez |
| **Kosu** | `d4_20260826` | Hipotezi gerceklestiren deney kaydi |
| **Referans** | `v00_saglikli` | Karsilastirma tabani |
| **Kontrol** | `C2 seed13` | Dogal oynakligi olcen kosu |
| **Varyant** | `D4 last_pt`, `E4 imgsz512` | Checkpoint/cozunurluk/seed farki |

Hepsini 26 esit karta donusturmek, veri modelindeki her satiri arayuze
sizdiriyordu: kullanici ilk ekranda D4 ile `C2 seed21`'i ayni onemde
goruyordu. Teknik olarak eksiksiz, iletisim acisindan yanlis.

Bu modul iki liste uretir:

- `senaryolar()` - yalnizca ARASTIRMA SORUSU tasiyan senaryolar (katalog.yaml)
- kalan her sey "kosu defteri"ne aittir (results.csv)

Hicbir sey elle yazilmaz. Ad ve aile `senaryolar/katalog.yaml`, aciklama
`senaryolar/anlatim.yaml`, kosular `results.csv`, referans
`karsilastirilabilirlik.py`, bulgu ise kirilim/kanit olcumlerinden gelir.
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path
from typing import Any

import yaml

KOK = Path(__file__).resolve().parents[1]
KATALOG = KOK / "senaryolar/katalog.yaml"
RESULTS_CSV = KOK / "results.csv"

AILE_ADI = {
    "veri_etiket": "Veri ve etiket",
    "dagilim": "Dağılım",
    "egitim": "Eğitim",
    "cikarim": "Çıkarım",
}


@functools.lru_cache(maxsize=1)
def _ham_katalog() -> list[dict]:
    ham = yaml.safe_load(KATALOG.read_text(encoding="utf-8"))
    return ham.get("senaryolar", [])


@functools.lru_cache(maxsize=1)
def _defter() -> list[dict[str, str]]:
    with RESULTS_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _kod(kosu: str) -> str:
    """Kosu adindan SENARYO kodunu cikarir.

    'D2b final_best' -> 'D2b', 'E3b seed42' -> 'E3b', 'D4 last_pt' -> 'D4'.

    Ayrica MODEL AILESI son eki cozulur: `D1n`, D1 bozulmasinin farkli bir
    baslangic modeli (yolo26n) uzerindeki kaydidir - ayri bir senaryo degil,
    D1'in varyantidir. Cozulmedigi surece D1n hicbir senaryoya baglanmiyor
    ve katalogda oksuz kaliyordu.

    Son ek yalnizca geriye kalan parca GERCEK bir katalog kodu ise soyulur;
    boylece ileride katalogda adi 'D1n' olan bir senaryo tanimlanirsa o
    kazanir.
    """
    taban = kosu.split()[0]
    kodlar = {s["kod"] for s in _ham_katalog()}
    if taban not in kodlar and taban.endswith("n") and taban[:-1] in kodlar:
        return taban[:-1]
    return taban


def kosulari(kod: str) -> dict[str, Any]:
    """Bir senaryonun butun kosulari: ana kosu ve varyantlari.

    ANA KOSU secimi: `best.pt` checkpoint'li, kilitli sette olculmus ve
    protokolun ana baslangic modeliyle egitilmis olan. Boyle birden fazla
    varsa defterdeki ILK satir alinir (defter append-only, yani ilk kayit
    senaryonun asil kosusudur).
    """
    satirlar = [s for s in _defter() if _kod(s["scenario"]) == kod]
    if not satirlar:
        return {"ana": None, "varyantlar": []}

    def ana_mi(s: dict) -> bool:
        return (s["weights_path"].endswith("best.pt")
                and s.get("evaluation_set") == "val_diagnostic"
                and s.get("model") == "main_model.pt")

    anadaylar = [s for s in satirlar if ana_mi(s)]
    ana = (anadaylar or satirlar)[0]
    return {
        "ana": ana["scenario"],
        "ana_run_id": ana["run_id"],
        "varyantlar": [s["scenario"] for s in satirlar
                       if s["scenario"] != ana["scenario"]],
    }


def _ana_etki(kosu: str | None) -> dict[str, Any]:
    """Senaryonun EN BELIRGIN etkisi - GURULTUYE GORE TARTILMIS.

    Ham farki buyuklugune gore secmek yanlisti: her senaryoda `tf2026 recall`
    kazaniyordu, cunku o grup yalnizca 106 bbox tasiyor ve dogal yayilimi
    (band 0.1321) butun gruplarin en genisi. Yani "en buyuk fark" cogu zaman
    en gurultulu grubun gurultusuydu.

    Dogru olcut BAND ORANI: |fark| / o grubun bozulmasiz kosular arasindaki
    yayilimi. Orani 1'in altinda kalan hicbir aday ana etki sayilmaz -
    projenin butun metodolojisi bu cumleye dayaniyor.

    Genel metrikler ve kirilim gruplari ayni olcutle yarisir; boylece D4'te
    <16 px bandi (oran ~48) toplam mAP50'yi (oran ~3.7) hakli olarak geciyor.
    """
    import senaryo_grafikleri as sg
    from teshis.degerlendirme.gurultu import fark_degerlendir
    from teshis.degerlendirme.senaryo_ozeti import ne_gozlendi

    if not kosu:
        return {}
    gozlem = ne_gozlendi(kosu)
    if not gozlem:
        return {}

    # ESLENIK olcumde (E4, D6a) egitim rastgeleligi HIC devrede degil: model
    # dosyasi birebir ayni, degisen tek sey cikarim ayari veya degerlendirme
    # kumesi. Dolayisiyla gurultu esigi uygulanamaz ve fark dogrudan etkidir.
    if gozlem.get("karsilastirma_turu") == "eslenik":
        adaylar = [{"alan": ad, "fark": d["fark"], "oran": None,
                    "kirilim": False, "eslenik": True}
                   for ad, d in gozlem["metrikler"].items()
                   if d["fark"] is not None]
        return (max(adaylar, key=lambda a: abs(a["fark"])) if adaylar else {})

    adaylar = []
    for ad, d in gozlem["metrikler"].items():
        esik = d["gurultu_esigi"]
        if d["fark"] is None or not esik:
            continue
        adaylar.append({"alan": ad, "fark": d["fark"],
                        "oran": abs(d["fark"]) / esik, "kirilim": False})

    from teshis.degerlendirme.karsilastirilabilirlik import _defter as _kd

    run_id = (_kd().get(kosu) or {}).get("run_id")
    for alan, etiket, harita in (
        ("boyut_bandi_recall", "recall", None),
        ("kaynak_recall", "recall", None),
        ("sinif_recall", "recall", None),
    ):
        try:
            veri = sg._kirilim_cifti(kosu, alan, harita)
        except Exception:  # noqa: BLE001 - kirilim yoksa kart yine uretilsin
            veri = None
        if veri is None or veri.empty:
            continue
        for _, satir in veri.iterrows():
            band = fark_degerlendir(alan, satir["grup"], float(satir["fark"]),
                                    haric_run_id=run_id)
            oran = band.get("band_orani")
            if not oran:
                continue
            # Az gozlemli banda dayanan aday ana etki OLAMAZ. n=4 kontrolle
            # bandin kendisi kararsizdir; gurultu.py bu esigi (n<5 ve
            # oran<5) zaten tanimliyor, burada tekrar yazilmaz.
            if band.get("az_gozlem"):
                continue
            adaylar.append({
                "alan": f"{satir['grup']} {etiket}",
                "fark": float(satir["fark"]),
                "oran": oran,
                "kirilim": True,
            })

    # Gurultunun icinde kalan hicbir sey "ana etki" degildir.
    anlamli = [a for a in adaylar if a["oran"] > 1]
    if not anlamli:
        return {}
    return max(anlamli, key=lambda a: a["oran"])


@functools.lru_cache(maxsize=1)
def senaryolar() -> list[dict[str, Any]]:
    """Arastirma senaryolari - referans, kontrol ve varyantlar HARIC.

    Her kayit arayuzun ihtiyaci olan her seyi tasir ve hepsi turetilmistir:
    yeni bir olcum geldiginde kartlar kendiliginden guncellenir.
    """
    from teshis.degerlendirme.senaryo_ozeti import kanit_gucu, ne_gozlendi, ozet

    kayitlar = []
    for s in _ham_katalog():
        kod = s["kod"]
        kosu = kosulari(kod)
        ana = kosu["ana"]
        gozlem = ne_gozlendi(ana) if ana else {}
        o = ozet(ana) if ana else {}
        kayitlar.append({
            "kod": kod,
            "ad": s.get("gorunen_ad") or s["ad"].replace("_", " "),
            "aile": s.get("aile", "veri_etiket"),
            "tur": s.get("tur"),
            "ne_olcuyor": (o.get("ne_olcuyor") or "").strip(),
            "ana_kosu": ana,
            "varyantlar": kosu["varyantlar"],
            "referans": (gozlem or {}).get("referans_senaryo"),
            "kanit": kanit_gucu(ana)["seviye"] if ana else "olcum yok",
            "asan": (gozlem or {}).get("asan_metrikler") or [],
            "ana_etki": _ana_etki(ana),
            "olculdu": bool(ana),
        })
    return kayitlar


def senaryo(kod: str) -> dict[str, Any] | None:
    return next((s for s in senaryolar() if s["kod"] == kod), None)


def kosu_defteri_disinda_mi(kosu: str) -> bool:
    """Bu kosu bir senaryonun ANA kosusu mu?

    Kosu defteri butun satirlari gosterir; senaryo katalogu yalnizca ana
    kosulari. Ayrim buradan tek kaynakla okunur.
    """
    return any(s["ana_kosu"] == kosu for s in senaryolar())


def olculemeyen() -> list[dict[str, Any]]:
    """Defterde satiri olmayan senaryolar.

    E3 boyle: ogrenme orani 100 kat yukseltilince egitim epoch 1'de
    iraksadi (butun kayiplar nan) ve degerlendirilebilir bir model
    uretilmedi. Bu bir eksiklik degil NEGATIF BULGUDUR; katalogdan
    dusurulmez, "ölçülemedi" rozetiyle gosterilir.
    """
    return [s for s in senaryolar() if not s["olculdu"]]

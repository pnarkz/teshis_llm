"""Alt grup gurultu bandinin TEK kaynagi.

Neden var
---------
Bir alt gruptaki farkin anlamli olup olmadigi, farkin BUYUKLUGUNDEN degil,
o grubun saglikli kosular arasindaki DOGAL YAYILIMINDAN okunur.

Bu, somut bir yanlis pozitifle ogrenildi. Ajan, hicbir bozulma icermeyen bir
kontrol kosusunda su teshisi koydu:

    "kaynak_d grubunda recall 0.9906'dan 0.8585'e duserek -0.1321 oraninda
     ciddi bir kayip yasamistir"  (guven: yuksek)

Rakamlar dogruydu ve iki ayri istatistik testi de farki anlamli buluyordu
(kutu birimli z=-3.64, p=2.7e-04; goruntu birimli tabakali bootstrap
araliklari ortusmuyordu). Ama o grupta bozulma YOKTU: dort saglikli kosuda
ayni degerin yayilimi zaten 0.1321'di.

Sorun testlerde degil, karsilastirma tabanindaydi. Tek bir referans kosusu,
yayilimi genis bir dagilimdan cekilmis TEK bir gozlemdir.

Onemli gozlem: bu yalnizca "kucuk orneklem" sorunu degildir. `termal` grubu
858 bbox tasir ama bandi 0.1306'dir - `hituav`in (2165 bbox) bandinin on
katindan fazla. Bazi gruplar gercekten oynaktir.
"""

from __future__ import annotations

import csv
import functools
import json
import statistics
from pathlib import Path
from typing import Any

KOK = Path(__file__).resolve().parents[2]
RESULTS_CSV = KOK / "results.csv"

# Kirilim dosyalarindaki recall alanlari.
KIRILIM_ALANLARI = ("kaynak_recall", "boyut_bandi_recall", "sinif_recall")


def _saglikli_kosular() -> list[dict[str, str]]:
    """Hicbir bozulma icermeyen kosular: saglikli referans + kontrol kosullari.

    Kontrol kosullari adlandirma kurali geregi `C` + rakam ile baslar
    (docs/MIMARI.md). Yeni bir seed eklendiginde liste kendiliginden buyur.
    """
    with RESULTS_CSV.open(encoding="utf-8") as f:
        satirlar = list(csv.DictReader(f))
    return [
        s for s in satirlar
        if (s["scenario"] == "v00_saglikli"
            or (s["scenario"].startswith("C") and s["scenario"][1:2].isdigit()))
        and s["weights_path"].endswith("best.pt")
    ]


@functools.lru_cache(maxsize=8)
def alt_grup_bandi(haric_run_id: str | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """Her kirilim grubu icin saglikli kosular arasindaki yayilimi dondurur.

    Cikti: ``{alan: {grup: {"band", "std", "n_kosu", "bbox_n", "recallar"}}}``

    ``haric_run_id`` verilirse o kosu banda KATILMAZ. Bu, degerlendirilen
    gozlemin kendi bandini tanimlamasini onler: kontrol kosularindan biri
    incelenirken o kosu banda dahil olursa, farki her zaman bandin tam
    sinirinda (oran = 1.00) gorunur ve olcut anlamini yitirir. Senaryo
    kosulari zaten banda girmedigi icin onlarda fark etmez.

    Iki kosudan az olcum kalirsa band hesaplanamaz ve o grup atlanir - band
    yerine uydurma bir sayi dondurmek, kapatmaya calistigi hatanin aynisini
    uretirdi.
    """
    olcumler = []
    for satir in _saglikli_kosular():
        if haric_run_id and satir["run_id"] == haric_run_id:
            continue
        yol = KOK / f"reports/kirilim/{satir['run_id']}.json"
        if yol.is_file():
            olcumler.append(json.loads(yol.read_text(encoding="utf-8")))

    band: dict[str, dict[str, dict[str, Any]]] = {alan: {} for alan in KIRILIM_ALANLARI}
    if len(olcumler) < 2:
        return band

    for alan in KIRILIM_ALANLARI:
        gruplar = {g for o in olcumler for g in o.get(alan, {})}
        for grup in gruplar:
            recallar = [
                o[alan][grup]["recall"] for o in olcumler
                if grup in o.get(alan, {}) and o[alan][grup]["recall"] is not None
            ]
            if len(recallar) < 2:
                continue
            band[alan][grup] = {
                "band": round(max(recallar) - min(recallar), 4),
                "std": round(statistics.stdev(recallar), 4),
                "n_kosu": len(recallar),
                "bbox_n": olcumler[0].get(alan, {}).get(grup, {}).get("gercek_kutu"),
                "recallar": [round(r, 4) for r in recallar],
            }
    return band


def fark_degerlendir(
    alan: str, grup: str, fark: float | None, haric_run_id: str | None = None
) -> dict[str, Any]:
    """Bir alt grup farkini o grubun gurultu bandiyla birlikte yorumlar.

    ``band_orani`` 1'den kucukse fark, saglikli kosular arasinda zaten
    gorulen yayilimin icindedir ve bozulma kaniti sayilamaz.
    """
    if fark is None:
        return {"band": None, "band_orani": None, "yorum": "fark hesaplanamadi"}

    bilgi = alt_grup_bandi(haric_run_id).get(alan, {}).get(grup)
    if bilgi is None:
        return {
            "band": None,
            "band_orani": None,
            "yorum": (
                "bu grup icin gurultu bandi yok (en az iki saglikli kosu olcumu "
                "gerekir); fark tek basina yorumlanmamalidir"
            ),
        }

    band, n = bilgi["band"], bilgi["n_kosu"]
    saglikli = bilgi["recallar"]
    oran = None if band == 0 else round(abs(fark) / band, 2)

    # n=3-4'te max-min kararsiz bir tahmincidir: uc gozlemi cikarinca band
    # cokuyor ve kalan gozlem "asiri" gorunuyor. Bu belirsizlik GIZLENMEZ,
    # yorumda soylenir. Ajanin ilk yanlis pozitifi, tam da bu belirsizligin
    # gorunmemesinden dogdu.
    # Cok buyuk bir oranda (>5x) az gozlem uyarisi anlamini yitirir:
    # bandin birkac kati fark, band tahmininin kararsizligiyla aciklanamaz.
    az_gozlem = n < 5 and (oran is not None and oran < 5)
    if oran is None:
        yorum = ("bu grup bozulmasiz kosularda hic degismiyor (band 0); fark "
                 "dikkat cekici olabilir ama band tek degerden hesaplanamadi")
    elif oran < 1:
        yorum = (
            "GURULTU ICINDE: bu buyuklukteki bir fark, hicbir bozulma icermeyen "
            "kosular arasinda da goruluyor. Tek basina bozulma kaniti degildir."
        )
    elif az_gozlem:
        yorum = (
            f"Bandin uzerinde, ANCAK band yalnizca {n} bozulmasiz kosudan "
            f"hesaplandi (gozlenen degerler: {saglikli}). Bu kadar az gozlemle "
            "'uc bir rastgelelik cekilisi' ile 'gercek etki' ayirt edilemez. "
            "Bu grubu tek basina teshise dayanak yapmayin; baska kirilimlarda "
            "da destek arayin."
        )
    elif oran < 2:
        yorum = "bandin hemen uzerinde; zayif kanit, tek basina yeterli degil"
    else:
        yorum = "bandin belirgin uzerinde"

    return {
        "band": band,
        "band_orani": oran,
        "band_kosu_sayisi": n,
        "bozulmasiz_kosu_degerleri": saglikli,
        "yorum": yorum,
    }


BAND_ACIKLAMASI = (
    "Her grup icin 'gurultu_bandi', hicbir bozulma icermeyen kosular arasinda "
    "o grupta gozlenen en buyuk yayilimi verir; 'bozulmasiz_kosu_degerleri' "
    "o kosularda olculen degerlerdir. 'band_orani' = |fark| / band. "
    "Orani 1'in ALTINDA olan bir fark, saf rastgelelikten ayirt edilemez ve "
    "bozulma kaniti olarak kullanilmamalidir - farkin mutlak buyuklugu ne "
    "olursa olsun. Orani 1'in uzerinde olsa bile, band az sayida kosudan "
    "hesaplandiysa ('band_kosu_sayisi') tek basina yeterli kanit degildir; "
    "'band_yorumu' bunu belirtir."
)

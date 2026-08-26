"""Ajan cevabini gizli cevap anahtarina karsi puanlayan pilot rubrik.

Bu, bir bilimsel karsilastirma degil (README'de de boyle belirtiliyor);
diagnosis, evidence ve limitations alanlarini esit agirlikla, anahtar
kelime/regex eslesmesiyle puanlar. Mantik onceden yalnizca
``scripts/score_llm_trial.py`` icinde tekrarlanmaz bir sekilde duruyordu;
bu modul onu tek kaynak haline getirir.

Not: Dosyanin ilk halinde bahsedilen T0/T1/T2 katmanli degerlendirme
semasi hicbir yerde tanimlanmamisti; burada icerilmiyor. Ilerde
katmanli bir rubrik gerekiyorsa once o katmanlarin ne olcecegi
netlestirilmelidir.
"""

from __future__ import annotations

import re
from typing import Any

# Senaryo kodu -> cevap anahtarindaki "expected" etiketi. prepare_llm_trial.py
# cevap anahtarini bu tablodan uretir; puanlama da ayni tabloyu okur, boylece
# paket ile puanlama birbirinden ayrisamaz.
SENARYO_BEKLENEN: dict[str, str] = {
    "Baseline": "saglikli_referans",
    # v00: temiz veriyle, senaryolarla ayni protokolde egitilmis saglikli
    # referans. Ajan icin de saglikli referans olarak puanlanir.
    "v00_saglikli": "saglikli_referans",
    "D3b": "tasit_insan_sinif_karisikligi",
    "D4": "kucuk_nesne_sinyal_kaybi",
    "D5": "kaynak_alani_kaymasi",
    "D1": "sinif_yetersizligi",
    "D2a": "lokalizasyon_etiket_gurultusu",
    "D2b": "eksik_etiket",
    # Ayni eksik-etiket bozulmasi, farkli baslangic modeli: beklenen teshis aynidir.
    "D2b final_best": "eksik_etiket",
    "D3": "uap_uai_sinif_karisikligi",
}

# beklenen etiket -> (diagnosis alaninda aranan regex, okunabilir etiket).
#
# Anahtar olarak kosu_NN degil beklenen etiket kullanilir: kosu_NN numaralari
# results.csv satir sirasina bagli oldugu icin araya yeni bir kosu eklenirse
# kayar; beklenen etiket ise senaryonun kendisine baglidir ve kaymaz.
# Yeni bir senaryo eklendiginde SENARYO_BEKLENEN'e ve buraya birlikte satir
# eklenmelidir; eksikse teshis_puani (0.0, None) doner, yani "degerlendirilemedi"
# olarak ayirt edilir ve sessizce yanlis puanlanmaz.
ANAHTAR_KALIPLAR: dict[str, tuple[str, str]] = {
    "saglikli_referans": (r"baseline|saglikli|referans|optimal|dengeli|en yuksek", "saglikli referans"),
    "sinif_yetersizligi": (r"sinif|yetersiz|insan.*(az|dus|kayb)|temsil", "sinif yetersizligi"),
    "lokalizasyon_etiket_gurultusu": (r"konum|lokal|box|iou|yerlestir|konumlandirma|gurultu", "lokalizasyon gurultusu"),
    "eksik_etiket": (r"eksik|etiket|anot|annotation|yanlis pozitif|false positive", "eksik etiket"),
    "uap_uai_sinif_karisikligi": (r"sinif|karis|uap|uai|id hatas|capraz|confusion", "UAP/UAI sinif karisikligi"),
    "tasit_insan_sinif_karisikligi": (r"sinif|karis|tasit|insan|id hatas|capraz|confusion", "tasit/insan sinif karisikligi"),
    "kucuk_nesne_sinyal_kaybi": (r"kucuk|boyut|small|olcek|piksel|cozunurluk|minik", "kucuk nesne sinyal kaybi"),
    "kaynak_alani_kaymasi": (r"kaynak|alan|domain|dagilim|kayma|shift|sensor|cesitlilik", "kaynak/alan kaymasi"),
}


def metin_ozeti(cevap: dict[str, Any]) -> str:
    """Puanlama icin cevabin metin alanlarini kucuk harfli tek dizeye indirger."""
    parcalar = [cevap.get("diagnosis", ""), *cevap.get("evidence", []), *cevap.get("limitations", [])]
    parcalar.append(cevap.get("next_measurement", ""))
    return " ".join(str(parca).lower() for parca in parcalar)


def teshis_puani(beklenen_etiket: str, cevap: dict[str, Any]) -> tuple[float, str | None]:
    """Diagnosis metninin beklenen teshise uyup uymadigini 0/0.5/1 olarak puanlar.

    beklenen_etiket, cevap anahtarindaki "expected" degeridir. ANAHTAR_KALIPLAR'da
    kayit yoksa (henuz kalip yazilmamis yeni bir senaryo) puan hesaplanamaz;
    (0.0, None) doner ve cagiran taraf bunu "degerlendirilemedi" olarak ayirt eder.
    """
    if beklenen_etiket not in ANAHTAR_KALIPLAR:
        return 0.0, None
    kalip, etiket = ANAHTAR_KALIPLAR[beklenen_etiket]
    metin = str(cevap.get("diagnosis", "")).lower()
    if re.search(kalip, metin):
        return 1.0, etiket
    # Eksik etikette model nedeni adlandiramasa bile precision/recall dengesizligini
    # dogru tarif etmisse kismi puan alir.
    if beklenen_etiket == "eksik_etiket" and "precision" in metin and "recall" in metin:
        return 0.5, etiket
    return 0.0, etiket


def kaniti_puanla(cevap: dict[str, Any]) -> float:
    """En az iki kanit ve en az birinde bir rakam var mi diye kontrol eder."""
    kanitlar = cevap.get("evidence", [])
    if not isinstance(kanitlar, list) or len(kanitlar) < 2:
        return 0.0
    return 1.0 if any(re.search(r"\d", str(deger)) for deger in kanitlar) else 0.0


def siniri_puanla(cevap: dict[str, Any]) -> float:
    """UAP/UAI bbox sayisinin (15, 17) sinirlamalar alaninda anildigini kontrol eder."""
    metin = " ".join(str(oge) for oge in cevap.get("limitations", [])).lower()
    return 1.0 if "uap" in metin and "uai" in metin and ("15" in metin or "17" in metin) else 0.0


def kosuyu_puanla(kosu_id: str, beklenen: dict[str, Any], cevap: dict[str, Any]) -> dict[str, Any]:
    """Tek bir kosu icin diagnosis/evidence/limitation alt puanlarini ve toplami dondurur."""
    teshis, beklenen_etiket = teshis_puani(str(beklenen.get("expected", "")), cevap)
    kanit = kaniti_puanla(cevap)
    sinir = siniri_puanla(cevap)
    return {
        "run_id": kosu_id,
        "expected": beklenen.get("expected"),
        "expected_label": beklenen_etiket,
        "diagnosis_score": teshis,
        "evidence_score": kanit,
        "limitation_score": sinir,
        "total": round((teshis + kanit + sinir) / 3, 3),
        "model_diagnosis": cevap.get("diagnosis", "missing"),
    }


def paketi_puanla(cevaplar: list[dict[str, Any]], cevap_anahtari: dict[str, Any]) -> dict[str, Any]:
    """Tum kor deneme paketini puanlar; cevap_anahtari={run_id: {"expected": ...}}."""
    id_ile_cevap = {oge.get("run_id"): oge for oge in cevaplar}
    satirlar = [
        kosuyu_puanla(kosu_id, beklenen, id_ile_cevap.get(kosu_id, {}))
        for kosu_id, beklenen in cevap_anahtari.items()
    ]
    return {
        "metric_definition": (
            "diagnosis, evidence and limitations each score 0..1; "
            "this is a pilot rubric, not a scientific benchmark"
        ),
        "runs": satirlar,
        "mean_score": round(sum(satir["total"] for satir in satirlar) / len(satirlar), 3) if satirlar else 0.0,
    }

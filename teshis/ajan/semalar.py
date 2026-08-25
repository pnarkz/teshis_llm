"""Ajanin JSON sozlesmeleri: arac (function-calling) bildirimleri ve teshis cikti semasi.

Bu modul iki seyi tek yerden tanimlar:

1. ``TESHIS_SEMASI`` / ``teshis_dogrula``: ajanin uretmesi gereken nihai JSON
   cevabinin sekli ve bunu programatik dogrulayan fonksiyon. Onceden
   ``scripts/run_gemini_trial.py`` Gemini'den gelen JSON'u hic dogrulamadan
   dogrudan diske yaziyordu; eksik alan veya gecersiz "confidence" degeri
   sessizce gecebiliyordu.
2. ``ARAC_BILDIRIMLERI``: Gemini function-calling icin arac tanimlari. Adlar
   ``teshis/ajan/araclar.py`` icindeki fonksiyon adlariyla birebir eslesir.
"""

from __future__ import annotations

from typing import Any

GECERLI_GUVEN_DEGERLERI = ("dusuk", "orta", "yuksek")

TESHIS_SEMASI: dict[str, Any] = {
    "type": "object",
    "required": ["diagnosis", "evidence", "confidence", "limitations", "next_measurement"],
    "properties": {
        "diagnosis": {"type": "string", "description": "Teshis; kanit yetersizse 'yetersiz_kanit'."},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 2, "description": "En az iki sayisal kanit."},
        "confidence": {"type": "string", "enum": list(GECERLI_GUVEN_DEGERLERI)},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "next_measurement": {"type": "string"},
    },
}


def teshis_dogrula(cevap: dict[str, Any]) -> list[str]:
    """TESHIS_SEMASI'na uymayan alanlari kod listesi olarak dondurur.

    Bos liste = gecerli. Cagiran taraf (ajan.py, testler) hatalari bulup
    cevabi reddedebilir veya kullaniciya raporlayabilir.
    """
    hatalar: list[str] = []
    for alan in TESHIS_SEMASI["required"]:
        if alan not in cevap:
            hatalar.append(f"eksik_alan:{alan}")

    if "diagnosis" in cevap and not isinstance(cevap["diagnosis"], str):
        hatalar.append("diagnosis_string_olmali")

    if "evidence" in cevap:
        kanitlar = cevap["evidence"]
        if not isinstance(kanitlar, list) or len(kanitlar) < 2:
            hatalar.append("evidence_en_az_iki_ogeli_liste_olmali")
        elif not any(any(karakter.isdigit() for karakter in str(oge)) for oge in kanitlar):
            hatalar.append("evidence_en_az_bir_sayisal_kanit_icermeli")

    if "confidence" in cevap and cevap["confidence"] not in GECERLI_GUVEN_DEGERLERI:
        hatalar.append("confidence_gecersiz_deger")

    if "limitations" in cevap and not isinstance(cevap["limitations"], list):
        hatalar.append("limitations_liste_olmali")

    if "next_measurement" in cevap and not isinstance(cevap["next_measurement"], str):
        hatalar.append("next_measurement_string_olmali")

    return hatalar


# Gemini function-calling icin arac bildirimleri. "parameters" alani
# google-genai types.FunctionDeclaration(**bildirim) ile dogrudan kullanilir.
ARAC_BILDIRIMLERI: list[dict[str, Any]] = [
    {
        "name": "kosu_listesini_getir",
        "description": (
            "Incelenebilecek tum anonim kosu kimliklerini dondurur. "
            "kosu_01 her zaman saglikli referanstir."
        ),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "baseline_metriklerini_getir",
        "description": (
            "Saglikli referans (kosu_01) kosusunun mAP50, mAP50_95, precision, "
            "recall ve sinif bazli AP50/AP50_95 degerlerini dondurur."
        ),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "kosu_metriklerini_getir",
        "description": (
            "Verilen anonim kosu_id icin mAP50, mAP50_95, precision, recall ve "
            "sinif bazli AP50 degerlerini dondurur. Senaryo adi icermez."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {"kosu_id": {"type": "STRING", "description": "Orn. kosu_02"}},
            "required": ["kosu_id"],
        },
    },
    {
        "name": "baseline_farkini_getir",
        "description": (
            "Verilen kosu_id'nin baseline'a (kosu_01) gore mAP50/mAP50_95/"
            "precision/recall farkini dondurur."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {"kosu_id": {"type": "STRING", "description": "Orn. kosu_02"}},
            "required": ["kosu_id"],
        },
    },
    {
        "name": "bbox_sayilarini_getir",
        "description": (
            "val_diagnostic tanı setinde sinif basina bbox sayisini dondurur. "
            "UAP/UAI gibi az orneli siniflarda genelleme sinirini degerlendirmek "
            "icin kullanilir."
        ),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
]

"""LLM tabanli teshis ajani giris noktasi.

Onceki durum: ``scripts/run_gemini_trial.py`` Gemini'ye tum kosularin
metriklerini tek bir buyuk JSON blogu halinde, tek seferlik bir prompt
icinde veriyordu. Model hicbir arac cagirmiyordu; bu bir "ajan" degil, tek
seferlik bir metin-ozetleme cagrisiydi.

Bu modul gercek function-calling kullanir: model once ``araclar.py``
icindeki fonksiyonlari (kosu listesi, metrikler, baseline farki, bbox
sayilari) cagirir, sonra ``semalar.TESHIS_SEMASI``'na uygun bir JSON teshis
uretir. Cikti ``semalar.teshis_dogrula`` ile programatik olarak dogrulanir.

Onemli sinirlama: ``teshis_uret`` ve ``kor_deneme_calistir`` gercekten
Gemini API'sine internet uzerinden baglanir ve ``GEMINI_API_KEY`` gerektirir.
Bu kod tabaninda ucdan uca (canli API ile) test edilmemistir; google-genai
SDK'sinin function-calling akisi (Content/Part/FunctionResponse rolleri)
kullanmadan once gercek bir API anahtariyla dogrulanmalidir. Buna karsin
``araclar.py`` ve ``semalar.py`` tamamen yerel/offline calisir ve
tests/test_ajan_araclar.py ile test edilir.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from . import araclar, semalar

ROOT = Path(__file__).resolve().parents[2]
VARSAYILAN_CIKTI = ROOT / "reports/llm_trial/gemini_response.json"

# Arac adi -> gercek Python fonksiyonu. semalar.ARAC_BILDIRIMLERI ile birebir eslesmelidir.
ARAC_UYGULAMALARI: dict[str, Callable[..., Any]] = {
    "kosu_listesini_getir": araclar.kosu_listesini_getir,
    "baseline_metriklerini_getir": araclar.baseline_metriklerini_getir,
    "kosu_metriklerini_getir": araclar.kosu_metriklerini_getir,
    "baseline_farkini_getir": araclar.baseline_farkini_getir,
    "bbox_sayilarini_getir": araclar.bbox_sayilarini_getir,
    "boyut_bazli_recall_getir": araclar.boyut_bazli_recall_getir,
    "kaynak_bazli_recall_getir": araclar.kaynak_bazli_recall_getir,
    "sinif_karisikligini_getir": araclar.sinif_karisikligini_getir,
}

SISTEM_TALIMATI = """
Sen termal drone YOLO nesne tespit modelinin teshis ajanisin. Sana veri
dogrudan verilmez; yalnizca sana taninan araclari cagirarak anonim kosu
metriklerini okuyabilirsin.

Kurallar:
- kosu_01, kosu_02, ... gibi kimliklerden hangi senaryoya ait olduklarini
  TAHMIN ETMEYE CALISMA; yalnizca metrik farklarina dayan.
- Her iddiayi en az iki sayisal kanitla destekle.
- bbox_sayilarini_getir ile bir sinifin ornek sayisi cok dusukse (orn. 15-20
  bbox) o sinif icin kesin genelleme yapma; bunu limitations alaninda belirt.
- Kanit yetersizse diagnosis alaninda tam olarak "yetersiz_kanit" yaz.
- Analizini bitirdiginde SADECE asagidaki JSON semasina uygun tek bir nesne
  dondur; baska aciklama metni ekleme:

{sema}
""".strip()


def _arac_tanimlari():
    """google-genai Tool nesnesini olusturur (yalnizca cagrildiginda import edilir)."""
    from google.genai import types

    return types.Tool(
        function_declarations=[types.FunctionDeclaration(**bildirim) for bildirim in semalar.ARAC_BILDIRIMLERI]
    )


def _arac_cagrisini_calistir(ad: str, argumanlar: dict[str, Any]) -> dict[str, Any]:
    fonksiyon = ARAC_UYGULAMALARI.get(ad)
    if fonksiyon is None:
        return {"hata": f"bilinmeyen_arac:{ad}"}
    try:
        return fonksiyon(**argumanlar)
    except Exception as hata:  # noqa: BLE001 - arac hatasi modele geri bildirilir, ajani dusurmez
        return {"hata": str(hata)}


def teshis_uret(kosu_id: str, model: str = "gemini-3.6-flash", max_tur: int = 6) -> dict[str, Any]:
    """Tek bir anonim kosu icin Gemini'ye arac erisimi vererek teshis uretir ve dogrular."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY tanimli degil. API anahtarini sohbete yazmadan "
            "PowerShell'de ortam degiskeni olarak ayarlayin."
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    sema_metni = json.dumps(semalar.TESHIS_SEMASI["properties"], ensure_ascii=False, indent=2)
    talimat = SISTEM_TALIMATI.format(sema=sema_metni)
    config = types.GenerateContentConfig(tools=[_arac_tanimlari()], system_instruction=talimat)

    contents: list[Any] = [
        types.Content(role="user", parts=[types.Part(text=f"Incelenecek kosu: {kosu_id}")]),
    ]

    for _ in range(max_tur):
        response = client.models.generate_content(model=model, contents=contents, config=config)
        aday = response.candidates[0]
        contents.append(aday.content)

        cagrilar = [parca.function_call for parca in aday.content.parts if getattr(parca, "function_call", None)]
        if not cagrilar:
            metin = (response.text or "").strip()
            cevap = json.loads(metin)
            hatalar = semalar.teshis_dogrula(cevap)
            if hatalar:
                raise ValueError(f"{kosu_id}: ajan ciktisi semaya uymuyor: {hatalar}")
            cevap["run_id"] = kosu_id
            return cevap

        yanit_parcalari = []
        for cagri in cagrilar:
            sonuc = _arac_cagrisini_calistir(cagri.name, dict(cagri.args or {}))
            yanit_parcalari.append(
                types.Part(function_response=types.FunctionResponse(name=cagri.name, response=sonuc))
            )
        contents.append(types.Content(role="tool", parts=yanit_parcalari))

    raise RuntimeError(f"{kosu_id}: {max_tur} turda semaya uygun bir JSON teshis alinamadi")


def kor_deneme_calistir(model: str = "gemini-3.6-flash", cikti: Path = VARSAYILAN_CIKTI) -> list[dict[str, Any]]:
    """Tum anonim kosular icin teshis uretir; mevcut demo/puanlama formatinda yazar."""
    sonuclar = [teshis_uret(kosu_id, model=model) for kosu_id in araclar.kosu_listesini_getir()]
    cikti.parent.mkdir(parents=True, exist_ok=True)
    cikti.write_text(json.dumps(sonuclar, indent=2, ensure_ascii=False), encoding="utf-8")
    return sonuclar


def main() -> None:
    parser = argparse.ArgumentParser(description="Function-calling tabanli LLM teshis ajani")
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--output", type=Path, default=VARSAYILAN_CIKTI)
    args = parser.parse_args()
    sonuclar = kor_deneme_calistir(model=args.model, cikti=args.output)
    print(json.dumps(sonuclar, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

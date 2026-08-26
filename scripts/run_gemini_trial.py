"""Anonim teshis denemesini Gemini ile calistirir (tek atislik, arac kullanmadan).

Bu script, paketi tek seferde modele verir ve JSON cevap bekler. Ajanin
function-calling surumu icin teshis/ajan/ajan.py kullanilir; oradaki soru
"hangi kirilima bakmasi gerektigini bilebiliyor mu" olurken burada tum kanit
onceden verilir ve yalnizca yorumlama olculur.

Cevap, yazilmadan once teshis/ajan/semalar.py::teshis_dogrula ile denetlenir;
gecersiz alanlar uyari olarak listelenir ancak cevap yine de kaydedilir
(puanlama tarafinda eksik alan zaten sifir puan alir).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teshis.ajan.semalar import GECERLI_GUVEN_DEGERLERI, teshis_dogrula  # noqa: E402


def prompt_uret(packet: dict) -> str:
    """Paketten, kosu sayisini ve kimliklerini iceren prompt uretir.

    Kosu sayisi paketten turetilir; onceki surumde "dort kosu" diye sabit
    yazilmisti ve paket dokuz kosuya cikinca prompt bayat kaldi.
    """
    kosu_kimlikleri = list(packet["runs"])
    return f"""
Bu JSON'da {len(kosu_kimlikleri)} anonim egitim kosusu var: {", ".join(kosu_kimlikleri)}.
kosu_01 saglikli referanstir; digerlerinde bir veri/etiket sorunu olabilir.

Her kosu icin bir teshis uret. Kosu adlarindan senaryo tahmin etmeye calisma;
yalnizca verilen sayilara dayan.

DIKKAT: Toplam mAP/precision/recall bazi bozulmalari GIZLEYEBILIR. Genel
metrikler az degismisken su alanlarda buyuk fark olabilir:
  - boyut_bandi_recall : nesne boyutuna gore recall (kucuk nesne kaybi)
  - kaynak_grubu_recall: veri kaynagina gore recall (alan/kaynak kaymasi)
  - sinif_karisikligi  : gercek sinif -> tahmin edilen sinif sayimlari
Genel metrikler durgunken bu kirilimlarin birinde belirgin fark varsa
teshisi o kirilime dayandir.

sinif_karisikligi icinde "bulunamadi", o gercek kutunun hicbir tahminle
eslesmedigi anlamina gelir; baska bir sinif adi ise yanlis siniflandirmadir.
Bir kirilimda bbox_n kucukse (orn. 20 altinda) oradan kesin sonuc cikarma,
bunu limitations alaninda belirt.

Cikti, tam olarak {len(kosu_kimlikleri)} nesneden olusan bir JSON DIZISI olsun.
Her nesne su alanlari icermeli:
{{
  "run_id": "kosu_NN",
  "diagnosis": "string",
  "evidence": ["en az iki sayisal kanit"],
  "confidence": "{'|'.join(GECERLI_GUVEN_DEGERLERI)}",
  "limitations": ["string"],
  "next_measurement": "string"
}}

Kanit yetersizse diagnosis alaninda "yetersiz_kanit" kullan.
Yalnizca gecerli JSON dizisi dondur, baska metin yazma.

Veri:
{json.dumps(packet, ensure_ascii=False)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--input", type=Path, default=ROOT / "reports/llm_trial/llm_input.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/llm_trial/gemini_response.json")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY tanimli degil. API anahtarini sohbete yazmadan "
            "PowerShell'de ortam degiskeni olarak ayarlayin:\n"
            '  $env:GEMINI_API_KEY = "..."'
        )
    if not args.input.is_file():
        raise SystemExit(
            f"Paket bulunamadi: {args.input}\n"
            "Once: python scripts/prepare_llm_trial.py --force"
        )

    from google import genai

    packet = json.loads(args.input.read_text(encoding="utf-8"))
    beklenen = list(packet["runs"])
    print(f"paket: {len(beklenen)} kosu ({args.input.name}), model={args.model}")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=args.model,
        contents=prompt_uret(packet),
        config={"response_mime_type": "application/json"},
    )

    ham = (response.text or "").strip()
    if not ham:
        raise SystemExit("Model bos cevap dondu; tekrar deneyin veya --model degistirin.")
    try:
        result = json.loads(ham)
    except json.JSONDecodeError as hata:
        bozuk = args.output.with_suffix(".ham.txt")
        bozuk.parent.mkdir(parents=True, exist_ok=True)
        bozuk.write_text(ham, encoding="utf-8")
        raise SystemExit(f"Model gecerli JSON dondurmedi ({hata}). Ham cevap: {bozuk}")

    if isinstance(result, dict):
        result = [result]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved={args.output.resolve()}")

    # --- kalite ozeti (puanlama degil) ---
    gelen = [str(oge.get("run_id", "")) for oge in result]
    eksik = [k for k in beklenen if k not in gelen]
    fazla = [k for k in gelen if k not in beklenen]
    print(f"\ncevap: {len(result)} kayit")
    if eksik:
        print(f"  UYARI eksik kosu : {eksik}")
    if fazla:
        print(f"  UYARI taninmayan : {fazla}")
    for oge in result:
        hatalar = teshis_dogrula(oge)
        if hatalar:
            print(f"  UYARI {oge.get('run_id', '?')}: sema hatalari {hatalar}")
    if not eksik and not fazla and all(not teshis_dogrula(o) for o in result):
        print("  tum kayitlar sema ile uyumlu")
    print("\nsonraki adim: python scripts/score_llm_trial.py")


if __name__ == "__main__":
    main()

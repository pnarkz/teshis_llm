"""Run the anonymized diagnostic trial with Gemini."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "reports/llm_trial/llm_input.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/llm_trial/gemini_response.json",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY tanimli degil. API anahtarini sohbete yazmadan "
            "PowerShell'de ortam degiskeni olarak ayarlayin."
        )

    from google import genai

    packet = json.loads(args.input.read_text(encoding="utf-8"))
    prompt = f"""
Bu JSON'daki dört anonim kosuyu analiz et.

Senaryo isimlerini kosu adlarindan tahmin etmeye calisma.
Her kosu icin su JSON semasina uygun cikti uret:
{{
  "diagnosis": "string",
  "evidence": ["en az iki sayisal kanit"],
  "confidence": "dusuk|orta|yuksek",
  "limitations": ["string"],
  "next_measurement": "string"
}}

UAP ve UAI bbox sayilarinin dusuk oldugunu dikkate al.
Kanıt yetersizse diagnosis alaninda "yetersiz_kanit" kullan.
Yalnizca gecerli JSON dondur.

Veri:
{json.dumps(packet, ensure_ascii=False)}
"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=args.model,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    result = json.loads(response.text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()

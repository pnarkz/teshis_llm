"""Score the anonymized LLM trial against the local hidden answer key.

Puanlama mantigi artik teshis/ajan/puanlama.py icinde tek kaynak olarak
tutulur (bkz. tests/test_ajan_puanlama.py); bu script sadece ince bir CLI
sarmalayicisidir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teshis.ajan.puanlama import paketi_puanla  # noqa: E402


def eslesme_denetle(response: list, key: dict) -> list[str]:
    """Cevaplarin cevap anahtariyla eslesip eslesmedigini denetler.

    Puanlama run_id uzerinden esler; run_id yoksa veya tutmuyorsa her kosu
    sessizce 0 alir ve "modelin performansi kotu" gibi gorunur. Bu fonksiyon
    o durumu ayirt edip acik uyari uretir.
    """
    uyarilar: list[str] = []
    if not isinstance(response, list):
        return ["Cevap bir JSON dizisi degil."]
    idsiz = [i for i, oge in enumerate(response) if not str(oge.get("run_id", "")).strip()]
    if idsiz:
        uyarilar.append(
            f"{len(idsiz)}/{len(response)} kayitta run_id alani yok. Puanlama run_id ile "
            "eslestigi icin hepsi 0 alir. scripts/ajan_tek_atislik_calistir.py denemeyi "
            "run_id isteyecek sekilde yeniden calistirir."
        )
    gelen = {str(oge.get("run_id", "")) for oge in response}
    eksik = sorted(set(key) - gelen)
    fazla = sorted(gelen - set(key) - {""})
    if eksik:
        uyarilar.append(f"Cevapta bulunmayan kosular: {eksik}")
    if fazla:
        uyarilar.append(f"Cevap anahtarinda olmayan run_id'ler: {fazla}")
    return uyarilar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--response", type=Path, default=ROOT / "reports/ajan_denemesi/gemini_response.json")
    parser.add_argument("--answer-key", type=Path, default=ROOT / "reports/ajan_denemesi/answer_key.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/ajan_denemesi/llm_score.json")
    parser.add_argument(
        "--yine-de-yaz",
        action="store_true",
        help="Eslesme uyarilarina ragmen skoru diske yaz (varsayilan: yazma).",
    )
    args = parser.parse_args()
    for yol in (args.response, args.answer_key):
        if not yol.is_file():
            raise SystemExit(f"Dosya bulunamadi: {yol}")

    response = json.loads(args.response.read_text(encoding="utf-8"))
    key = json.loads(args.answer_key.read_text(encoding="utf-8"))

    uyarilar = eslesme_denetle(response, key)
    if uyarilar:
        print("!!! ESLESME SORUNU — skor anlamli olmayabilir:")
        for uyari in uyarilar:
            print(f"  - {uyari}")
        print()

    result = paketi_puanla(response, key)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if uyarilar and not args.yine_de_yaz:
        raise SystemExit(
            "\nSkor diske YAZILMADI: yukaridaki esleme sorunu giderilmeden kaydetmek "
            "yaniltici bir kayit birakir. Bilerek kaydetmek icin --yine-de-yaz verin."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved={args.output.resolve()}")


if __name__ == "__main__":
    main()

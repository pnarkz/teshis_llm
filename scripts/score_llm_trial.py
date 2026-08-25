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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", type=Path, default=ROOT / "reports/llm_trial/gemini_response.json")
    parser.add_argument("--answer-key", type=Path, default=ROOT / "reports/llm_trial/answer_key.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/llm_trial/llm_score.json")
    args = parser.parse_args()
    response = json.loads(args.response.read_text(encoding="utf-8"))
    key = json.loads(args.answer_key.read_text(encoding="utf-8"))
    result = paketi_puanla(response, key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

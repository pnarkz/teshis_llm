"""Score the anonymized LLM trial against the local hidden answer key."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def text_of(item: dict) -> str:
    parts = [item.get("diagnosis", ""), *item.get("evidence", []), *item.get("limitations", [])]
    parts.append(item.get("next_measurement", ""))
    return " ".join(str(part).lower() for part in parts)


def diagnosis_score(run_id: str, response: dict) -> tuple[float, str]:
    # Causal diagnosis must be present in the diagnosis field itself. Evidence
    # and next_measurement are scored separately and must not leak into it.
    text = str(response.get("diagnosis", "")).lower()
    patterns = {
        "kosu_01": (r"baseline|saglikli|referans|optimal|dengeli|en yuksek", "saglikli referans"),
        "kosu_02": (r"sinif|yetersiz|insan.*(az|dus|kayb)|temsil", "sinif yetersizligi"),
        "kosu_03": (r"konum|lokal|box|iou|yerlestir|konumlandirma|gurultu", "lokalizasyon gurultusu"),
        "kosu_04": (r"eksik|etiket|anot|annotation|yanlis pozitif|false positive", "eksik etiket"),
    }
    pattern, label = patterns[run_id]
    if re.search(pattern, text):
        return 1.0, label
    if run_id == "kosu_04" and "precision" in text and "recall" in text:
        return 0.5, label
    return 0.0, label


def score_response(response: list[dict], key: dict) -> dict:
    by_id = {item.get("run_id"): item for item in response}
    rows = []
    for run_id, expected in key.items():
        item = by_id.get(run_id, {})
        diagnosis, expected_label = diagnosis_score(run_id, item)
        evidence = item.get("evidence", [])
        evidence_score = 1.0 if len(evidence) >= 2 and any(re.search(r"\d", str(value)) for value in evidence) else 0.0
        limitations_text = " ".join(item.get("limitations", [])).lower()
        limitation_score = 1.0 if "uap" in limitations_text and "uai" in limitations_text and ("15" in limitations_text or "17" in limitations_text) else 0.0
        rows.append({
            "run_id": run_id,
            "expected": expected["expected"],
            "expected_label": expected_label,
            "diagnosis_score": diagnosis,
            "evidence_score": evidence_score,
            "limitation_score": limitation_score,
            "total": round((diagnosis + evidence_score + limitation_score) / 3, 3),
            "model_diagnosis": item.get("diagnosis", "missing"),
        })
    return {
        "metric_definition": "diagnosis, evidence and limitations each score 0..1; this is a pilot rubric, not a scientific benchmark",
        "runs": rows,
        "mean_score": round(sum(row["total"] for row in rows) / len(rows), 3) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", type=Path, default=ROOT / "reports/llm_trial/gemini_response.json")
    parser.add_argument("--answer-key", type=Path, default=ROOT / "reports/llm_trial/answer_key.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/llm_trial/llm_score.json")
    args = parser.parse_args()
    response = json.loads(args.response.read_text(encoding="utf-8"))
    key = json.loads(args.answer_key.read_text(encoding="utf-8"))
    result = score_response(response, key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

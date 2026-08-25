"""Prepare an anonymized LLM diagnostic trial from completed evaluations.

Paket ve gizli cevap anahtari, ajanin calisma zamaninda kullandigi ayni
kaynaktan (teshis/ajan/araclar.py) turetilir. Boylece "LLM'ye verilen paket"
ile "ajanin araclarla gordugu veri" birbirinden ayrisamaz; yeni bir senaryo
results.csv'ye eklendiginde pakete de otomatik girer.

Onceki surum dort kosuyu (baseline, D1, D2a, D2b) dosya yollariyla hardcode
ediyordu; D2b final_best ve D3 eklendiginde paket sessizce eksik kaldi.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from teshis.ajan import araclar  # noqa: E402
from teshis.ajan.puanlama import SENARYO_BEKLENEN  # noqa: E402


def build_packet() -> tuple[dict, dict]:
    """Anonim LLM paketini ve gizli cevap anahtarini birlikte uretir."""
    frame = pd.read_csv(araclar.RESULTS_CSV).set_index("run_id")
    mapping = araclar.anonim_kosu_haritasi()
    bbox_counts = araclar.bbox_sayilarini_getir()

    runs: dict[str, dict] = {}
    answer_key: dict[str, dict] = {}

    for kosu_id in araclar.kosu_listesini_getir():
        metrics = araclar.kosu_metriklerini_getir(kosu_id)
        run = {
            "mAP50": metrics["mAP50"],
            "mAP50_95": metrics["mAP50_95"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "class_AP50": metrics["class_AP50"],
            "validation_images": 1056,
            "bbox_counts": bbox_counts,
            "delta_vs_kosu_01": araclar.baseline_farkini_getir(kosu_id),
        }
        reference = araclar.baseline_metriklerini_getir()["class_AP50"]
        run["class_AP50_delta_vs_kosu_01"] = {
            name: round(metrics["class_AP50"][name] - reference[name], 6)
            for name in metrics["class_AP50"]
        }
        runs[kosu_id] = run

        # Cevap anahtari yalnizca yerelde tutulur; pakete girmez.
        scenario = "Baseline" if kosu_id == "kosu_01" else str(frame.loc[mapping[kosu_id], "scenario"])
        expected = SENARYO_BEKLENEN.get(scenario)
        if expected is None:
            raise KeyError(
                f"{scenario} icin beklenen teshis tanimli degil. "
                "teshis/ajan/puanlama.py::SENARYO_BEKLENEN ve ANAHTAR_KALIPLAR'a ekleyin."
            )
        answer_key[kosu_id] = {"hidden_role": scenario, "expected": expected}

    packet = {
        "task": "Termal drone YOLO diagnostigi",
        "classes": list(araclar.SINIFLAR),
        "metric_definitions": {
            "mAP50": "IoU esigi 0.50 altinda ortalama tespit basarisi",
            "mAP50_95": "IoU 0.50-0.95 araliginda daha kati tespit basarisi",
            "precision": "Tahminlerin ne kadarinin dogru oldugu",
            "recall": "Gercek kutularin ne kadarinin yakalandigi",
        },
        "rules": [
            "Kosu adlarindan senaryo tahmini yapma.",
            "Her iddiayi en az iki sayisal kanitla destekle.",
            f"UAP/UAI bbox sayisi {bbox_counts['UAP']} ve {bbox_counts['UAI']} oldugu icin "
            "bu siniflarda kesin genelleme yapma.",
            "delta_vs_kosu_01 alanlarini degisim kaniti olarak kullan; tek basina nedensellik kaniti sayma.",
            "Kanıt yetersizse yetersiz_kanit de.",
        ],
        "required_output": {
            "diagnosis": "string",
            "evidence": ["string"],
            "confidence": "dusuk|orta|yuksek",
            "limitations": ["string"],
            "next_measurement": "string",
        },
        "runs": runs,
    }
    return packet, answer_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "reports/llm_trial")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Mevcut llm_input.json/answer_key.json dosyalarinin uzerine yaz.",
    )
    args = parser.parse_args()

    packet, answer_key = build_packet()
    args.output.mkdir(parents=True, exist_ok=True)
    input_path = args.output / "llm_input.json"
    key_path = args.output / "answer_key.json"

    if not args.force and (input_path.exists() or key_path.exists()):
        existing = json.loads(input_path.read_text(encoding="utf-8")).get("runs", {}) if input_path.exists() else {}
        parser.error(
            f"{args.output} icinde paket zaten var ({len(existing)} kosu) ve yeni paket "
            f"{len(packet['runs'])} kosu iceriyor. Uzerine yazmak LLM cevabini (gemini_response.json) "
            "kayitli paketten kopardigi icin varsayilan olarak engellendi. Bilerek yeniliyorsaniz "
            "--force verin ve ardindan LLM denemesini yeniden calistirin."
        )

    input_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
    key_path.write_text(json.dumps(answer_key, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "README.md").write_text(
        "# LLM Deneme Paketi\n\n"
        "LLM'ye sadece `llm_input.json` verilir. `answer_key.json` gizli tutulur ve\n"
        "LLM cevabini sonradan puanlamak icin kullanilir. Beklenen cikti,\n"
        "`required_output` semasina uygun JSON olmalidir.\n\n"
        "Bu dosyalar `scripts/prepare_llm_trial.py` tarafindan `results.csv` ve\n"
        "`teshis/ajan/araclar.py` uzerinden uretilir; elle duzenlenmemelidir.\n",
        encoding="utf-8",
    )
    print(f"created={args.output} runs={len(packet['runs'])}")


if __name__ == "__main__":
    main()

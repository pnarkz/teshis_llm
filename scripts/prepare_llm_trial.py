"""Prepare an anonymized LLM diagnostic trial from completed evaluations."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "llm_trial"


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compact(metrics: dict, bbox_counts: dict[str, int]) -> dict:
    if "siniflar" in metrics:
        class_names = list(metrics["siniflar"])
        class_ap50 = [metrics["siniflar"][name]["mAP50"] for name in class_names]
        class_ap50_95 = [metrics["siniflar"][name]["mAP50_95"] for name in class_names]
    else:
        class_names = metrics["class_names"]
        class_ap50 = metrics["class_ap50"]
        class_ap50_95 = metrics["class_ap50_95"]
    return {
        "mAP50": metrics["mAP50"],
        "mAP50_95": metrics["mAP50_95"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "class_AP50": dict(zip(class_names, class_ap50)),
        "class_AP50_95": dict(zip(class_names, class_ap50_95)),
        "validation_images": 1056,
        "bbox_counts": bbox_counts,
    }


def main() -> None:
    baseline_source = load_metrics(
        ROOT / "reports/model_karsilastirma_fair/model_karsilastirma.json"
    )["aday"]
    sources = [
        ("kosu_01", baseline_source, {"tasit": 1264, "insan": 2718, "UAP": 15, "UAI": 17}),
        ("kosu_02", load_metrics(ROOT / "reports/d1_sonuc/d1_metrics.json"), {"tasit": 1264, "insan": 2718, "UAP": 15, "UAI": 17}),
        ("kosu_03", load_metrics(ROOT / "reports/d2a_sonuc/d1_metrics.json"), {"tasit": 1264, "insan": 2718, "UAP": 15, "UAI": 17}),
        ("kosu_04", load_metrics(ROOT / "reports/d2b_sonuc/d1_metrics.json"), {"tasit": 1264, "insan": 2718, "UAP": 15, "UAI": 17}),
    ]
    runs = {
        name: compact(metrics, bbox_counts)
        for name, metrics, bbox_counts in sources
    }
    reference = runs["kosu_01"]
    for run in runs.values():
        run["delta_vs_kosu_01"] = {
            metric: round(run[metric] - reference[metric], 6)
            for metric in ("mAP50", "mAP50_95", "precision", "recall")
        }
        run["class_AP50_delta_vs_kosu_01"] = {
            class_name: round(
                run["class_AP50"][class_name] - reference["class_AP50"][class_name],
                6,
            )
            for class_name in run["class_AP50"]
        }
    packet = {
        "task": "Termal drone YOLO diagnostigi",
        "classes": ["tasit", "insan", "UAP", "UAI"],
        "metric_definitions": {
            "mAP50": "IoU esigi 0.50 altinda ortalama tespit basarisi",
            "mAP50_95": "IoU 0.50-0.95 araliginda daha kati tespit basarisi",
            "precision": "Tahminlerin ne kadarinin dogru oldugu",
            "recall": "Gercek kutularin ne kadarinin yakalandigi",
        },
        "rules": [
            "Kosu adlarindan senaryo tahmini yapma.",
            "Her iddiayi en az iki sayisal kanitla destekle.",
            "UAP/UAI bbox sayisi 15 ve 17 oldugu icin bu siniflerde kesin genelleme yapma.",
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
    answer_key = {
        "kosu_01": {"hidden_role": "baseline", "expected": "saglikli_referans"},
        "kosu_02": {"hidden_role": "D1", "expected": "sinif_yetersizligi"},
        "kosu_03": {"hidden_role": "D2a", "expected": "lokalizasyon_etiket_gurultusu"},
        "kosu_04": {"hidden_role": "D2b", "expected": "eksik_etiket"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "llm_input.json").write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "answer_key.json").write_text(json.dumps(answer_key, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# LLM Deneme Paketi\n\n"
        "LLM'ye sadece `llm_input.json` verilir. `answer_key.json` gizli tutulur ve\n"
        "LLM cevabini sonradan puanlamak icin kullanilir. Beklenen cikti,\n"
        "`required_output` semasina uygun JSON olmalidir.\n",
        encoding="utf-8",
    )
    print(f"created={OUT}")


if __name__ == "__main__":
    main()

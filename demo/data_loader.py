"""Load completed experiment evidence for the presentation demo."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_results() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "results.csv")
    baseline = read_json(ROOT / "reports/model_karsilastirma_fair/model_karsilastirma.json")["aday"]
    baseline_row = {
        "run_id": "baseline_768",
        "scenario": "Baseline",
        "data_version": "kilitli_referans",
        "seed": 42,
        "model": "main_model.pt",
        "imgsz_train": 768,
        "imgsz_eval": 768,
        "epochs": "-",
        "batch": "-",
        "lr0": "-",
        "evaluation_set": "val_diagnostic",
        "mAP50": baseline["mAP50"],
        "mAP50_95": baseline["mAP50_95"],
        "precision": baseline["precision"],
        "recall": baseline["recall"],
        "AP_tasit": baseline["siniflar"]["tasit"]["mAP50"],
        "AP_insan": baseline["siniflar"]["insan"]["mAP50"],
        "AP_UAP": baseline["siniflar"]["UAP"]["mAP50"],
        "AP_UAI": baseline["siniflar"]["UAI"]["mAP50"],
        "duration_min": "-",
        "weights_path": "main_model.pt",
    }
    combined = pd.concat([pd.DataFrame([baseline_row]), frame], ignore_index=True)
    # Two D2b runs use different starting models; keep their dashboard keys unique.
    combined.loc[combined["run_id"] == "d2b_20260820_final", "scenario"] = "D2b final_best"
    return combined


def evidence_for(scenario: str) -> dict:
    paths = {
        "D1": ROOT / "reports/d1_sonuc/d1_metrics.json",
        "D2a": ROOT / "reports/d2a_sonuc/d1_metrics.json",
        "D2b": ROOT / "reports/d2b_sonuc/d1_metrics.json",
        "D2b final_best": ROOT / "reports/d2b_final_best_sonuc/d1_metrics.json",
    }
    if scenario == "Baseline":
        return read_json(ROOT / "reports/model_karsilastirma_fair/model_karsilastirma.json").get("aday", {})
    return read_json(paths.get(scenario, Path("")))


def images_for(scenario: str) -> list[Path]:
    folders = {
        "D1": ROOT / "reports/d1_sonuc/d1_val_diagnostic",
        "D2a": ROOT / "reports/d2a_sonuc/d2a_val_diagnostic",
        "D2b": ROOT / "reports/d2b_sonuc/d2b_val_diagnostic",
        "D2b final_best": ROOT / "reports/d2b_final_best_sonuc/d2b_final_best_val_diagnostic",
    }
    folder = folders.get(scenario)
    if not folder or not folder.is_dir():
        return []
    return [folder / name for name in ("confusion_matrix.png", "confusion_matrix_normalized.png") if (folder / name).is_file()]


def examples_for(scenario: str) -> list[Path]:
    folders = {
        "D1": ROOT / "reports/d1_sonuc/d1_val_diagnostic",
        "D2a": ROOT / "reports/d2a_sonuc/d2a_val_diagnostic",
        "D2b": ROOT / "reports/d2b_sonuc/d2b_val_diagnostic",
        "D2b final_best": ROOT / "reports/d2b_final_best_sonuc/d2b_final_best_val_diagnostic",
    }
    folder = folders.get(scenario)
    if not folder or not folder.is_dir():
        return []
    names = ("val_batch0_labels.jpg", "val_batch0_pred.jpg", "val_batch1_pred.jpg")
    return [folder / name for name in names if (folder / name).is_file()]


def llm_response() -> list | dict:
    return read_json(ROOT / "reports/llm_trial/gemini_response.json")


def llm_score() -> dict:
    return read_json(ROOT / "reports/llm_trial/llm_score.json")

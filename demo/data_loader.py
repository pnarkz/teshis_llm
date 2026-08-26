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


EVIDENCE_JSON = {
    "v00_saglikli": ROOT / "reports/v00_sonuc/d1_metrics.json",
    "D1": ROOT / "reports/d1_v2_sonuc/d1_metrics.json",
    "D2a": ROOT / "reports/d2a_sonuc/d1_metrics.json",
    "D2b": ROOT / "reports/d2b_sonuc/d1_metrics.json",
    "D2b final_best": ROOT / "reports/d2b_final_best_sonuc/d1_metrics.json",
    "D3": ROOT / "reports/d3_v2_sonuc/d1_metrics.json",
    "D3b": ROOT / "reports/d3b_sonuc/d1_metrics.json",
    "D4": ROOT / "reports/d4_best_sonuc/d1_metrics.json",
    "D5": ROOT / "reports/d5_best_sonuc/d1_metrics.json",
}

EVIDENCE_FOLDERS = {
    "v00_saglikli": ROOT / "reports/v00_sonuc/v00_val_diagnostic",
    "D1": ROOT / "reports/d1_v2_sonuc/d1_v2_val_diagnostic",
    "D2a": ROOT / "reports/d2a_sonuc/d2a_val_diagnostic",
    "D2b": ROOT / "reports/d2b_sonuc/d2b_val_diagnostic",
    "D2b final_best": ROOT / "reports/d2b_final_best_sonuc/d2b_final_best_val_diagnostic",
    "D3": ROOT / "reports/d3_v2_sonuc/d3_v2_val_diagnostic",
    "D3b": ROOT / "reports/d3b_sonuc/d3b_val_diagnostic",
    "D4": ROOT / "reports/d4_best_sonuc/d4_best_val_diagnostic",
    "D5": ROOT / "reports/d5_best_sonuc/d5_best_val_diagnostic",
}


def evidence_for(scenario: str) -> dict:
    if scenario == "Baseline":
        return read_json(ROOT / "reports/model_karsilastirma_fair/model_karsilastirma.json").get("aday", {})
    return read_json(EVIDENCE_JSON.get(scenario, Path("")))


def images_for(scenario: str) -> list[Path]:
    folder = EVIDENCE_FOLDERS.get(scenario)
    if not folder or not folder.is_dir():
        return []
    return [folder / name for name in ("confusion_matrix.png", "confusion_matrix_normalized.png") if (folder / name).is_file()]


def examples_for(scenario: str) -> list[Path]:
    """val_batch etiket/tahmin ciftlerini (etiket, tahmin) sirasinda dondurur."""
    folder = EVIDENCE_FOLDERS.get(scenario)
    if not folder or not folder.is_dir():
        return []
    names = (
        "val_batch0_labels.jpg", "val_batch0_pred.jpg",
        "val_batch1_labels.jpg", "val_batch1_pred.jpg",
        "val_batch2_labels.jpg", "val_batch2_pred.jpg",
    )
    return [folder / name for name in names if (folder / name).is_file()]


CURVE_FILES = (
    ("BoxPR_curve.png", "Precision-Recall"),
    ("BoxF1_curve.png", "F1 / guven esigi"),
    ("BoxP_curve.png", "Precision / guven esigi"),
    ("BoxR_curve.png", "Recall / guven esigi"),
)


def curves_for(scenario: str) -> list[tuple[Path, str]]:
    """Diagnostic degerlendirmenin PR/F1/P/R egri gorsellerini dondurur."""
    folder = EVIDENCE_FOLDERS.get(scenario)
    if not folder or not folder.is_dir():
        return []
    return [(folder / name, label) for name, label in CURVE_FILES if (folder / name).is_file()]


def run_dir_for(row: pd.Series) -> Path | None:
    """results.csv satirindaki weights_path'ten kosu klasorunu turetir.

    Boylece senaryo -> kosu klasoru eslemesi ayrica hardcode edilmez; yeni bir
    senaryo results.csv'ye eklendiginde egitim egrisi otomatik gelir.
    """
    weights = str(row.get("weights_path", ""))
    if "/weights/" not in weights.replace("\\", "/"):
        return None
    run_dir = ROOT / weights.replace("\\", "/").split("/weights/")[0]
    return run_dir if run_dir.is_dir() else None


def training_curve(row: pd.Series) -> pd.DataFrame:
    """Kosunun epoch bazli Ultralytics results.csv dosyasini okur."""
    run_dir = run_dir_for(row)
    if run_dir is None:
        return pd.DataFrame()
    path = run_dir / "results.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def train_batch_images(row: pd.Series) -> list[Path]:
    """Egitim sirasinda kaydedilen train_batch onizlemeleri (bozulmus veriyi gosterir)."""
    run_dir = run_dir_for(row)
    if run_dir is None:
        return []
    return sorted(run_dir.glob("train_batch*.jpg"))


def label_distribution_image(row: pd.Series) -> Path | None:
    """Kosunun labels.jpg sinif/bbox dagilim grafigini dondurur."""
    run_dir = run_dir_for(row)
    if run_dir is None:
        return None
    path = run_dir / "labels.jpg"
    return path if path.is_file() else None


def error_galleries() -> dict[str, dict]:
    """reports/ altindaki *_hata_galerisi klasorlerini senaryo koduna gore dondurur."""
    galleries: dict[str, dict] = {}
    reports = ROOT / "reports"
    if not reports.is_dir():
        return galleries
    for folder in sorted(reports.glob("*_hata_galerisi")):
        manifest = folder / "gallery.json"
        if not manifest.is_file():
            continue
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        scenario = folder.name.replace("_hata_galerisi", "").upper().replace("D2A", "D2a").replace("D2B", "D2b")
        galleries[scenario] = {"folder": folder, "entries": entries}
    return galleries


def sparkline(values: list[float], blocks: str = "▁▂▃▄▅▆▇█") -> str:
    """Sayi dizisini tek satirlik unicode blok grafigine cevirir."""
    numbers = [float(value) for value in values if pd.notna(value)]
    if not numbers:
        return ""
    low, high = min(numbers), max(numbers)
    if high - low < 1e-12:
        return blocks[len(blocks) // 2] * len(numbers)
    scale = len(blocks) - 1
    return "".join(blocks[round((number - low) / (high - low) * scale)] for number in numbers)


def llm_response() -> list | dict:
    return read_json(ROOT / "reports/llm_trial/gemini_response.json")


def llm_score() -> dict:
    return read_json(ROOT / "reports/llm_trial/llm_score.json")

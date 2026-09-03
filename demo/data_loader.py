"""Load completed experiment evidence for the presentation demo."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teshis.degerlendirme.raporlar import rapor_klasoru as _rapor_klasoru


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_results() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "results.csv")
    baseline = read_json(ROOT / "reports/model_secimi/model_karsilastirma.json")["aday"]
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
    return combined


# Rapor klasoru senaryo adindan KONVANSIYONLA turetilir; elle tutulan bir
# harita degildir. Onceki surum sabit kodlu bir sozlukdu ve her yeni senaryoda
# geride kaliyordu: D6a, D6b, v00n ve D1n eklendiginde demo onlari sessizce
# "kanit yok" gosteriyordu. Adlandirma kurali docs/MIMARI.md'de tanimlidir.
# Rapor klasoru kurali tek kaynaktan gelir; burada kopyasi TUTULMAZ.
# Iki kopya daha once birbirinden ayrilmisti (C2 kontrol kosusu demo'da
# bulunuyor, kanit uretiminde bulunmuyordu).
def rapor_klasoru(scenario: str) -> Path | None:
    return _rapor_klasoru(scenario, ROOT)


def evidence_for(scenario: str) -> dict:
    if scenario == "Baseline":
        return read_json(ROOT / "reports/model_secimi/model_karsilastirma.json").get("aday", {})
    klasor = rapor_klasoru(scenario)
    return read_json(klasor / "d1_metrics.json") if klasor else {}


def images_for(scenario: str) -> list[Path]:
    klasor = rapor_klasoru(scenario)
    folder = klasor / "gorseller" if klasor else None
    if not folder or not folder.is_dir():
        return []
    return [folder / name for name in ("confusion_matrix.png", "confusion_matrix_normalized.png") if (folder / name).is_file()]


def examples_for(scenario: str) -> list[Path]:
    """val_batch etiket/tahmin ciftlerini (etiket, tahmin) sirasinda dondurur."""
    klasor = rapor_klasoru(scenario)
    folder = klasor / "gorseller" if klasor else None
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
    klasor = rapor_klasoru(scenario)
    folder = klasor / "gorseller" if klasor else None
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
    """Hata galerilerini SENARYO ADINA gore dondurur.

    Klasor adindaki alt cizgiyi sokup senaryo adi saymak yanlisti: "C2 seed13"
    senaryosunun klasoru `hata_galerisi_C2_seed13` oldugu icin galeri
    `C2_seed13` anahtariyla dururdu ve defterdeki adla eslesmezdi. Sonuc,
    demonun galeri seciciside klasor adlarinin gorunmesiydi.

    Dogru yon tersidir: defterdeki her senaryo icin klasor adi TEK KAYNAKTAN
    (raporlar.galeri_adi) uretilir ve galeri o senaryo adiyla anahtarlanir.
    """
    from teshis.degerlendirme.raporlar import galeri_adi

    galleries: dict[str, dict] = {}
    for scenario in load_results()["scenario"]:
        folder = ROOT / "reports" / galeri_adi(str(scenario))
        manifest = folder / "gallery.json"
        if not manifest.is_file():
            continue
        galleries[str(scenario)] = {
            "folder": folder,
            "entries": json.loads(manifest.read_text(encoding="utf-8")),
        }
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
    """TEK ATISLIK denemenin cevaplari (tum kanit onceden prompt'a konur)."""
    return read_json(ROOT / "reports/ajan_denemesi/gemini_response.json")


def ajan_cevaplari() -> list:
    """FONKSIYON CAGIRMA denemesinin cevaplari (ajan kaniti kendi secer).

    Iki deneme ayri dosyalarda durur ve karistirilmamalidir; demo ajan
    sayfasi fonksiyon cagirma denemesini gosterir, karsilastirma sayfasi
    ikisini yan yana koyar.
    """
    d = read_json(ROOT / "reports/ajan_denemesi/ajan_response.json")
    return d if isinstance(d, list) else []


def llm_score() -> dict:
    return read_json(ROOT / "reports/ajan_denemesi/llm_score.json")


# --- Ajan katmani -----------------------------------------------------------
#
# Ana denemenin arac kaydinda yalnizca cagri ADLARI var; arac cevaplarini
# saklama ozelligi sonradan eklendi. Demo bu bosluğu API harcamadan kapatir:
# araclar yerel ve deterministiktir, ayni kosu icin ayni cevabi verir. Ekranda
# "yerel olarak yeniden calistirildi" diye etiketlenir.

def ajan_kosu_haritasi() -> dict[str, str]:
    """kosu_NN -> gercek senaryo adi. YALNIZCA sunucu gorunumu icin."""
    import csv

    from teshis.ajan import araclar

    with open(araclar.RESULTS_CSV, encoding="utf-8") as f:
        senaryo = {s["run_id"]: s["scenario"] for s in csv.DictReader(f)}
    harita = {"kosu_01": araclar.REFERANS_SENARYO}
    harita.update({k: senaryo.get(v, v) for k, v in araclar.anonim_kosu_haritasi().items()})
    return harita


def ajan_kaydi() -> dict:
    """Tamamlanmis denemenin cevaplari, arac kaydi ve puanlari."""
    cevaplar = ajan_cevaplari()
    kayit = read_json(ROOT / "reports/ajan_denemesi/ajan_arac_kaydi.json")
    puan = llm_score()
    return {
        "cevaplar": {c.get("run_id"): c for c in cevaplar},
        "arac_kaydi": kayit,
        "puanlar": {r["run_id"]: r for r in puan.get("runs", [])},
        "ozet": {k: v for k, v in puan.items() if k != "runs"},
    }


def ajan_araclarini_calistir(kosu_id: str) -> dict:
    """Ajanin gordugu kaniti yerel olarak yeniden uretir (API harcamaz)."""
    from teshis.ajan import araclar

    cagrilabilir = {
        "baseline_metriklerini_getir": lambda: araclar.baseline_metriklerini_getir(),
        "kosu_metriklerini_getir": lambda: araclar.kosu_metriklerini_getir(kosu_id),
        "baseline_farkini_getir": lambda: araclar.baseline_farkini_getir(kosu_id),
        "bbox_sayilarini_getir": lambda: araclar.bbox_sayilarini_getir(),
        "boyut_bazli_recall_getir": lambda: araclar.boyut_bazli_recall_getir(kosu_id),
        "kaynak_bazli_recall_getir": lambda: araclar.kaynak_bazli_recall_getir(kosu_id),
        "sinif_karisikligini_getir": lambda: araclar.sinif_karisikligini_getir(kosu_id),
    }
    sonuc = {}
    for ad, fn in cagrilabilir.items():
        try:
            sonuc[ad] = fn()
        except Exception as hata:  # noqa: BLE001 - biri duserse digerleri gorunsun
            sonuc[ad] = {"hata": f"{type(hata).__name__}: {hata}"}
    return sonuc


def ajana_gizlenenler() -> dict[str, str]:
    """Korlestirme paneli: neyin ajana gitmedigi, gerekcesiyle."""
    from teshis.ajan import araclar

    return {
        "Senaryo adi": "Gonderilmiyor - kosular kosu_NN olarak sunulur",
        "Veri surumu / manifest": "Gonderilmiyor",
        "Bozulma parametresi": "Gonderilmiyor",
        "Kaynak grubu adlari": "Takma adla (kaynak_a, kaynak_b ...)",
        "Dosya yollari": "Gonderilmiyor",
        "Cevap anahtari": "Gonderilmiyor - puanlama cevap uretildikten SONRA yerelde yapilir",
        "Anonim metrikler": "Gonderiliyor",
        "Kirilim araclari": f"Kullanilabilir ({len(araclar.ARAC_ADLARI) if hasattr(araclar, 'ARAC_ADLARI') else 7} arac)",
    }

"""Ajanin function-calling arac katmani.

katalog.yaml kurali geregi ``manifest_ajana_verilmez: true`` ve senaryo
adlari kor testte ajana verilmez. Bu modul bu kurali kod seviyesinde
uygular: hicbir fonksiyon veri surumu manifestini, senaryo adini veya
"D1"/"D2a" gibi kodlari disariya vermez; yalnizca results.csv ve
reports/ altindaki sayisal metrikleri, anonim ``kosu_NN`` kimlikleriyle
sunar. Gercek run_id/senaryo eslemesi yalnizca ``anonim_kosu_haritasi``
icinde tutulur ve bu fonksiyon ajanin arac listesine dahil edilmez;
sadece puanlama/rapor tarafinda (yerelde) kullanilir.

Her fonksiyon teshis/ajan/semalar.py::ARAC_BILDIRIMLERI icindeki bir arac
tanimina karsilik gelir; isim ve parametreler birebir eslesmelidir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SINIFLAR = ("tasit", "insan", "UAP", "UAI")

# val_diagnostic kilitli oldugu icin sabittir (PROJECT_STRUCTURE.md / README).
VAL_DIAGNOSTIC_BBOX_N: dict[str, int] = {"tasit": 1264, "insan": 2718, "UAP": 15, "UAI": 17}

RESULTS_CSV = ROOT / "results.csv"
BASELINE_JSON = ROOT / "reports/model_karsilastirma_fair/model_karsilastirma.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_rows() -> pd.DataFrame:
    if not RESULTS_CSV.is_file():
        raise FileNotFoundError(f"results.csv bulunamadi: {RESULTS_CSV}")
    return pd.read_csv(RESULTS_CSV)


def anonim_kosu_haritasi() -> dict[str, str]:
    """kosu_NN -> gercek run_id eslemesi. YALNIZCA yerel puanlama/rapor icin; ajana verilmez.

    Numaralandirma results.csv satir sirasina (yani kosunun eklendigi
    kronolojik siraya) dayanir, boylece yeni bir senaryo eklendiginde eski
    kosu_NN kimlikleri degismez. kosu_01 baseline'a ayrildigi icin senaryo
    kosulari kosu_02'den baslar.
    """
    frame = _scenario_rows()
    return {f"kosu_{index + 2:02d}": str(run_id) for index, run_id in enumerate(frame["run_id"])}


def kosu_listesini_getir() -> list[str]:
    """Ajanin gorebilecegi tum anonim kosu kimliklerini dondurur (kosu_01 = baseline dahil)."""
    return ["kosu_01", *sorted(anonim_kosu_haritasi())]


def baseline_metriklerini_getir() -> dict[str, Any]:
    """Saglikli referans (kosu_01) metriklerini dondurur."""
    veri = _read_json(BASELINE_JSON).get("aday")
    if not veri:
        raise FileNotFoundError(f"Baseline metrikleri bulunamadi: {BASELINE_JSON}")
    return {
        "mAP50": veri["mAP50"],
        "mAP50_95": veri["mAP50_95"],
        "precision": veri["precision"],
        "recall": veri["recall"],
        "class_AP50": {isim: veri["siniflar"][isim]["mAP50"] for isim in SINIFLAR},
        "class_AP50_95": {isim: veri["siniflar"][isim]["mAP50_95"] for isim in SINIFLAR},
    }


def kosu_metriklerini_getir(kosu_id: str) -> dict[str, Any]:
    """Verilen anonim kosu kimligi icin metrikleri dondurur. Senaryo adi icermez."""
    if kosu_id == "kosu_01":
        return baseline_metriklerini_getir()
    harita = anonim_kosu_haritasi()
    if kosu_id not in harita:
        raise KeyError(f"Bilinmeyen kosu_id: {kosu_id}")
    frame = _scenario_rows().set_index("run_id")
    row = frame.loc[harita[kosu_id]]
    return {
        "mAP50": float(row["mAP50"]),
        "mAP50_95": float(row["mAP50_95"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "class_AP50": {isim: float(row[f"AP_{isim}"]) for isim in SINIFLAR},
    }


def baseline_farkini_getir(kosu_id: str) -> dict[str, float]:
    """Secili kosunun baseline'a gore mAP50/mAP50_95/precision/recall farkini dondurur."""
    if kosu_id == "kosu_01":
        return {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}
    kosu = kosu_metriklerini_getir(kosu_id)
    taban = baseline_metriklerini_getir()
    return {alan: round(kosu[alan] - taban[alan], 6) for alan in ("mAP50", "mAP50_95", "precision", "recall")}


def bbox_sayilarini_getir() -> dict[str, int]:
    """val_diagnostic uzerindeki sinif basina bbox sayisini dondurur (belirsizlik notu icin)."""
    return dict(VAL_DIAGNOSTIC_BBOX_N)

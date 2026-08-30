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
BASELINE_JSON = ROOT / "reports/model_secimi/model_karsilastirma.json"
# teshis/degerlendirme/metrikler.py ciktilari: run_id basina bir dosya.
KIRILIM_DIR = ROOT / "reports/kirilim"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_rows() -> pd.DataFrame:
    if not RESULTS_CSV.is_file():
        raise FileNotFoundError(f"results.csv bulunamadi: {RESULTS_CSV}")
    return pd.read_csv(RESULTS_CSV)


REFERANS_SENARYO = "v00_saglikli"

# Ajana senaryo olarak SUNULMAYAN kosular ve nedenleri.
#
# Ajan tum kosulari kosu_01 (v00, main_model tabanli saglikli referans) ile
# karsilastirir. Baska bir taban modelden gelen kosular bu referansla
# kiyaslanamaz: fark, bozulmadan degil model kapasitesinden gelir.
#
# yolo26n cifti (v00n / D1n) kendi icinde gecerli bir kontrollu deneydir ve
# kendi referansina (v00n) sahiptir; README'de ayri bir bolumde raporlanir.
AJANA_VERILMEYEN: dict[str, str] = {
    REFERANS_SENARYO: "karsilastirma tabani (kosu_01 olarak sunulur)",
    "v00n": "farkli taban model (yolo26n); kendi ciftinin referansi",
    "D1n": "farkli taban model (yolo26n); v00n ile karsilastirilir",
}
# Ajan yalnizca AYNI kilitli kumede olculmus kosulari karsilastirabilir.
# Farkli bir degerlendirme kumesinde olculen bir kosu (orn. D6a'nin sizintili
# kumesi) ayni tabloya konursa, ajan elmayla armudu kiyaslar ve fark
# "bozulma" gibi gorunur. Bu sabit, o karistirmayi yapisal olarak engeller.
KILITLI_DEGERLENDIRME_SETI = "val_diagnostic"


def _referans_satiri() -> "pd.Series":
    """Saglikli referans kosusunun (v00) results.csv satiri."""
    frame = _scenario_rows()
    satir = frame[frame["scenario"] == REFERANS_SENARYO]
    if satir.empty:
        raise FileNotFoundError(
            f"Saglikli referans ({REFERANS_SENARYO}) results.csv'de bulunamadi"
        )
    return satir.iloc[0]


def anonim_kosu_haritasi() -> dict[str, str]:
    """kosu_NN -> gercek run_id eslemesi. YALNIZCA yerel puanlama/rapor icin; ajana verilmez.

    Numaralandirma results.csv satir sirasina (yani kosunun eklendigi
    kronolojik siraya) dayanir, boylece yeni bir senaryo eklendiginde eski
    kosu_NN kimlikleri degismez.

    kosu_01 saglikli referansa (v00) ayrildigi icin senaryo kosulari
    kosu_02'den baslar ve v00'in kendisi bu haritaya DAHIL EDILMEZ: ajanin
    teshis etmesi gereken bir senaryo degil, karsilastirma tabanidir.

    Kilitli tanı setinden BASKA bir kumede olculmus kosular ve
    AJANA_VERILMEYEN'de listelenenler disarida birakilir; ajan yalnizca ayni
    tabanda olculmus ve ayni taban modelden gelen kosulari karsilastirabilir.
    """
    frame = _scenario_rows()
    senaryolar = frame[
        (~frame["scenario"].isin(AJANA_VERILMEYEN))
        & (frame["evaluation_set"] == KILITLI_DEGERLENDIRME_SETI)
    ]
    return {
        f"kosu_{index + 2:02d}": str(run_id)
        for index, run_id in enumerate(senaryolar["run_id"])
    }


def kosu_listesini_getir() -> list[str]:
    """Ajanin gorebilecegi tum anonim kosu kimliklerini dondurur (kosu_01 = baseline dahil)."""
    return ["kosu_01", *sorted(anonim_kosu_haritasi())]


def baseline_metriklerini_getir() -> dict[str, Any]:
    """Saglikli referans (kosu_01) metriklerini dondurur.

    Referans, veri hic bozulmadan senaryolarla **ayni protokolde** egitilmis
    v00 kosusudur. Onceden burada fine-tune edilmemis main_model.pt
    kullaniliyordu; o durumda her senaryo farki (bozulma etkisi) +
    (fine-tune etkisi) toplami oluyordu ve ajan yanlis tabana gore
    karsilastirma yapiyordu. Ayrinti: README "v00 Saglikli Referans".
    """
    row = _referans_satiri()
    return {
        "mAP50": float(row["mAP50"]),
        "mAP50_95": float(row["mAP50_95"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "class_AP50": {isim: float(row[f"AP_{isim}"]) for isim in SINIFLAR},
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


def _kirilim_oku(kosu_id: str) -> dict[str, Any]:
    """Bir kosunun kirilim analizini okur; dosya adi anonim kimlikten turetilmez.

    Kirilim dosyalari gercek run_id ile isimlendirildigi icin esleme burada
    yapilir ve ajana yalnizca sayilar doner; dosya yolu veya run_id disari
    sizmaz.
    """
    if kosu_id == "kosu_01":
        # kosu_01 saglikli referansin kendisidir; kendi kirilimi dondurulur
        # (bu durumda "fark" alanlari sifir cikar).
        return _referans_kirilim()
    harita = anonim_kosu_haritasi()
    if kosu_id not in harita:
        raise KeyError(f"Bilinmeyen kosu_id: {kosu_id}")
    yol = KIRILIM_DIR / f"{harita[kosu_id]}.json"
    if not yol.is_file():
        raise FileNotFoundError(f"{kosu_id} icin kirilim analizi henuz uretilmemis")
    return json.loads(yol.read_text(encoding="utf-8"))


def _referans_kirilim() -> dict[str, Any]:
    """Saglikli referans (v00) kirilim analizini okur."""
    yol = KIRILIM_DIR / f"{_referans_satiri()['run_id']}.json"
    if not yol.is_file():
        raise FileNotFoundError("Saglikli referans icin kirilim analizi uretilmemis")
    return json.loads(yol.read_text(encoding="utf-8"))


def boyut_bazli_recall_getir(kosu_id: str) -> dict[str, Any]:
    """Nesne boyutu bandi basina recall'i, saglikli referansla birlikte dondurur.

    Toplam mAP kucuk nesne kaybini gizleyebilir; bu kirilim onu gorunur kilar.
    """
    kosu, referans = _kirilim_oku(kosu_id), _referans_kirilim()
    sonuc: dict[str, Any] = {}
    for bant, deger in kosu["boyut_bandi_recall"].items():
        taban = referans["boyut_bandi_recall"].get(bant, {})
        sonuc[bant] = {
            "bbox_n": deger["gercek_kutu"],
            "recall": deger["recall"],
            "referans_recall": taban.get("recall"),
            "fark": (
                round(deger["recall"] - taban["recall"], 4)
                if deger["recall"] is not None and taban.get("recall") is not None
                else None
            ),
        }
    return {"bant_tanimi_px": kosu.get("bant_tanimi", {}), "bantlar": sonuc}


def kaynak_bazli_recall_getir(kosu_id: str) -> dict[str, Any]:
    """Veri kaynagi grubu basina recall'i, saglikli referansla birlikte dondurur.

    Kaynak gruplari anonimdir (kaynak_a, kaynak_b ...): gercek kaynak adlari
    (aaterm, hituav ...) ajana verilmez, cunku senaryo tahminine yardim
    edebilir. Gruplama sirasi bbox sayisina gore sabittir.
    """
    kosu, referans = _kirilim_oku(kosu_id), _referans_kirilim()
    sirali = sorted(
        referans["kaynak_recall"].items(), key=lambda oge: -oge[1]["gercek_kutu"]
    )
    takma = {gercek: f"kaynak_{chr(97 + i)}" for i, (gercek, _) in enumerate(sirali)}
    sonuc: dict[str, Any] = {}
    for gercek_ad, taban in sirali:
        deger = kosu["kaynak_recall"].get(gercek_ad)
        if deger is None:
            continue
        sonuc[takma[gercek_ad]] = {
            "bbox_n": deger["gercek_kutu"],
            "recall": deger["recall"],
            "referans_recall": taban["recall"],
            "fark": (
                round(deger["recall"] - taban["recall"], 4)
                if deger["recall"] is not None and taban["recall"] is not None
                else None
            ),
        }
    return {
        "aciklama": "Kaynak gruplari anonimlestirilmistir; sira bbox sayisina gore sabittir.",
        "kaynaklar": sonuc,
    }


def sinif_karisikligini_getir(kosu_id: str) -> dict[str, Any]:
    """Gercek sinif basina, modelin hangi sinifi tahmin ettigini dondurur.

    Sinif bazli AP ve recall, "dogru yerde bulundu ama yanlis sinif verildi"
    durumunu gizleyebilir; bu matris onu gorunur kilar. "bulunamadi", o gercek
    kutunun hicbir tahminle eslesmedigi anlamina gelir.
    """
    kosu, referans = _kirilim_oku(kosu_id), _referans_kirilim()
    sonuc: dict[str, Any] = {}
    for gercek_sinif, sayimlar in kosu["karisiklik_matrisi"].items():
        taban = referans["karisiklik_matrisi"].get(gercek_sinif, {})
        toplam = sum(sayimlar.values())
        sonuc[gercek_sinif] = {
            "toplam_gercek_kutu": toplam,
            "tahminler": dict(sorted(sayimlar.items(), key=lambda oge: -oge[1])),
            "referans_tahminler": dict(
                sorted(taban.items(), key=lambda oge: -oge[1])
            ),
        }
    return sonuc

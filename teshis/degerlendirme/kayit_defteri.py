"""results.csv'ye satir eklemenin TEK yolu.

Neden gerekli
-------------
`results.csv` bugune kadar elle tutuldu: her yeni kosu icin ayri bir tek
seferlik script yazildi. Bunun uc somut bedeli oldu:

1. **Senaryo adi belirsizligi.** Iki farkli kosu (`d2b_20260820_main` ve
   `d2b_20260820_final`) ayni `scenario` degerini tasidi; demo bunu run_id'ye
   gore elle yamalamak zorunda kaldi ve kanit ureticisi yanlis rapor
   klasorunu buldu.
2. **Tutarsiz `last_pt` kaydi.** E1 hem `best` hem `last` satiri aldi;
   D4/D5/D6b'nin `last.pt` olcumleri uretildi ama defterde satiri olmadi.
   Demo ve ajan defterden beslendigi icin bu olcumler gorunmez kaldi.
3. **Sessiz alan kaymasi.** Satirlar v00 satiri kopyalanip uzerine yazilarak
   uretildi; guncellenmeyi unutulan bir alan (orn. `weights_path`) yanlis
   kosuya isaret etti.

Bu modul satiri OLCUM DOSYASINDAN uretir, benzersizligi dogrular ve
adlandirma kuralini uygular.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

KOK = Path(__file__).resolve().parents[2]
RESULTS_CSV = KOK / "results.csv"
SINIFLAR = ["tasit", "insan", "UAP", "UAI"]

SUTUNLAR = [
    "run_id", "scenario", "data_version", "seed", "model",
    "imgsz_train", "imgsz_eval", "epochs", "batch", "lr0",
    "evaluation_set", "mAP50", "mAP50_95", "precision", "recall",
    "AP_tasit", "AP_insan", "AP_UAP", "AP_UAI", "duration_min", "weights_path",
]


def oku() -> list[dict[str, str]]:
    with RESULTS_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def satir_uret(
    *,
    run_id: str,
    scenario: str,
    metrik_json: Path,
    weights_path: str,
    data_version: str,
    seed: int,
    model: str = "main_model.pt",
    imgsz_train: int = 768,
    imgsz_eval: int = 768,
    epochs: int,
    batch: int = 8,
    lr0: float = 0.001,
    evaluation_set: str = "val_diagnostic",
    duration_min: float = 0.0,
) -> dict[str, Any]:
    """Olcum dosyasindan bir defter satiri uretir.

    Metrikler ELLE verilmez; `metrik_json` dosyasindan okunur. Boylece
    defterdeki sayi ile raporun sayisi ayrisamaz.
    """
    yol = Path(metrik_json)
    if not yol.is_absolute():
        yol = KOK / yol
    if not yol.is_file():
        raise FileNotFoundError(f"Olcum dosyasi yok: {yol}")
    m = json.loads(yol.read_text(encoding="utf-8"))

    eksik = [a for a in ("mAP50", "mAP50_95", "precision", "recall",
                         "class_ap50") if a not in m]
    if eksik:
        raise ValueError(f"{yol} bu alanlari icermiyor: {eksik}")
    if len(m["class_ap50"]) != len(SINIFLAR):
        raise ValueError(
            f"{yol}: class_ap50 {len(SINIFLAR)} sinif icermeli, "
            f"{len(m['class_ap50'])} var"
        )

    return {
        "run_id": run_id,
        "scenario": scenario,
        "data_version": data_version,
        "seed": str(seed),
        "model": model,
        "imgsz_train": str(imgsz_train),
        "imgsz_eval": str(imgsz_eval),
        "epochs": str(epochs),
        "batch": str(batch),
        "lr0": str(lr0),
        "evaluation_set": evaluation_set,
        "mAP50": f"{m['mAP50']:.7f}",
        "mAP50_95": f"{m['mAP50_95']:.7f}",
        "precision": f"{m['precision']:.7f}",
        "recall": f"{m['recall']:.7f}",
        **{f"AP_{ad}": f"{m['class_ap50'][i]:.7f}" for i, ad in enumerate(SINIFLAR)},
        "duration_min": f"{duration_min:g}",
        "weights_path": weights_path,
    }


def dogrula(satir: dict[str, Any], mevcut: list[dict[str, str]]) -> None:
    """Defterin degismezlerini kontrol eder; ihlalde ekleme yapilmaz."""
    if satir["run_id"] in {s["run_id"] for s in mevcut}:
        raise ValueError(f"run_id zaten defterde: {satir['run_id']}")
    if satir["scenario"] in {s["scenario"] for s in mevcut}:
        raise ValueError(
            f"scenario zaten defterde: '{satir['scenario']}'. Senaryo adlari "
            "BENZERSIZ olmalidir - iki kosu ayni adi tasidiginda demo ve kanit "
            "ureticisi hangisinin hangisi oldugunu ayirt edemez "
            "(d2b_main / d2b_final bu yuzden karisti)."
        )
    if eksik := [a for a in SUTUNLAR if a not in satir]:
        raise ValueError(f"Satirda eksik sutun: {eksik}")
    if fazla := [a for a in satir if a not in SUTUNLAR]:
        raise ValueError(f"Satirda tanimsiz sutun: {fazla}")


def ekle(satir: dict[str, Any]) -> None:
    """Dogrulanmis satiri defterin SONUNA ekler.

    Sona eklemek zorunludur: ajanin `kosu_NN` numaralari satir sirasina
    dayanir, ortaya eklenen bir satir tamamlanmis denemeleri gecersiz kilar.
    """
    mevcut = oku()
    dogrula(satir, mevcut)
    with RESULTS_CSV.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, SUTUNLAR).writerow(satir)


# DEFTERE GIRME KURALI
#
# Defter, KOSULAR ARASI karsilastirmada kullanilan her olcumu kaydeder.
# Yalnizca tek bir analizin ICINDE anlamli olan olcumler defterde degil, o
# analizin kendi raporunda durur. Aksi halde ana karsilastirma tablosu ayni
# modelin tekrarlariyla dolar ve okunmaz hale gelir.
#
# Kural disi birakilan her klasor burada GEREKCESIYLE yazilir; sessizce
# atlanmaz.
DEFTER_DISI: dict[str, str] = {
    **{
        f"senaryo_E4_imgsz{r}": (
            "E4 cozunurluk taramasinin ara noktasi. Tarama tek bir analizdir "
            "ve egrisi reports/senaryo_E4/e4_cozunurluk_taramasi.json icinde "
            "durur; bes noktayi ayri kosu gibi deftere yazmak ana tabloyu ayni "
            "modelin tekrarlariyla doldururdu. Mansete giren 512 noktasinin "
            "satiri vardir."
        )
        for r in (640, 768, 1024, 1280)
    },
}


def kayitsiz_olcumler() -> list[str]:
    """Olcumu uretilmis ama defterde satiri olmayan rapor klasorleri.

    `eski_` onekli klasorler superseded kayitlardir; DEFTER_DISI ise gerekcesi
    yazilmis bilincli istisnalardir. Geriye kalan her sey bir EKSIKTIR:
    uretilmis ama demo ve ajanin goremedigi bir olcum.
    """
    from .raporlar import klasor_adi

    kayitli = {klasor_adi(s["scenario"]) for s in oku()}
    var = {
        p.name for p in (KOK / "reports").iterdir()
        if p.is_dir() and (p / "d1_metrics.json").is_file()
        and not p.name.startswith("eski_")
        and p.name not in DEFTER_DISI
    }
    return sorted(var - kayitli)

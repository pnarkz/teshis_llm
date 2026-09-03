"""Bir senaryonun deney ozetini kaynak dosyalardan toplar.

Demo, senaryo aciklamalarini 24 girdilik elle tutulan bir sozlukte
tasiyordu. Her yeni senaryoda geride kaliyordu ve gercekten geride kaldi:
D6a, D6b, v00n ve D1n eklendiginde demo onlari sessizce eksik gosterdi.

Burada tersi yapilir - ozetin bes bileseninden dordu **turetilir**:

| Soru | Kaynak |
|---|---|
| Ne degistirildi? | senaryolar/<tur>/<kod>_*.yaml (parametreler) |
| Ne sabit tutuldu? | ayni dosya (kaynak_surum, hedef_split) + protokol |
| Beklenen etki neydi? | ayni dosya (beklenen_kanit) |
| Ne gozlendi? | results.csv + reports/ (referansa gore farklar) |
| Kanit ne kadar guclu? | gurultu.py bandi (asiyor mu, kac kosudan) |

Elle yazilan tek alan senaryonun NE OLCTUGU: `senaryolar/anlatim.yaml`.
Bu, turetilemeyecek tek bilgidir - arastirma niyeti veride yazmaz.
"""

from __future__ import annotations

import csv
import functools
import json
from pathlib import Path
from typing import Any

import yaml

KOK = Path(__file__).resolve().parents[2]
SENARYOLAR = KOK / "senaryolar"
RESULTS_CSV = KOK / "results.csv"

GENEL_METRIKLER = ("mAP50", "mAP50_95", "precision", "recall")


@functools.lru_cache(maxsize=1)
def _katalog() -> dict[str, dict]:
    ham = yaml.safe_load((SENARYOLAR / "katalog.yaml").read_text(encoding="utf-8"))
    return {s["kod"]: s for s in ham.get("senaryolar", [])}


@functools.lru_cache(maxsize=1)
def _anlatim() -> dict[str, str]:
    yol = SENARYOLAR / "anlatim.yaml"
    if not yol.is_file():
        return {}
    return yaml.safe_load(yol.read_text(encoding="utf-8")) or {}


@functools.lru_cache(maxsize=1)
def _defter() -> dict[str, dict[str, str]]:
    with RESULTS_CSV.open(encoding="utf-8") as f:
        return {s["scenario"]: s for s in csv.DictReader(f)}


def _kod(senaryo: str) -> str:
    """'D2b final_best' -> 'D2b', 'C2 seed13' -> 'C2', 'E1 last_pt' -> 'E1'."""
    return senaryo.split()[0]


def _konfig(senaryo: str) -> dict[str, Any]:
    kayit = _katalog().get(_kod(senaryo))
    if not kayit:
        return {}
    yol = SENARYOLAR / kayit["config"]
    if not yol.is_file():
        return {}
    return yaml.safe_load(yol.read_text(encoding="utf-8")) or {}


def _referans() -> dict[str, Any]:
    return json.loads(
        (KOK / "reports/referans_v00/d1_metrics.json").read_text(encoding="utf-8")
    )


def ne_degisti(senaryo: str) -> dict[str, Any]:
    """Bozulma turu ve uygulanan parametreler."""
    kayit = _katalog().get(_kod(senaryo), {})
    konfig = _konfig(senaryo)
    satir = _defter().get(senaryo, {})
    return {
        "kod": _kod(senaryo),
        "tur": kayit.get("tur"),          # veri | egitim
        "parametreler": konfig.get("parametreler") or {},
        "veri_surumu": satir.get("data_version"),
        "hedef_split": konfig.get("hedef_split"),
    }


def ne_sabit_kaldi(senaryo: str) -> list[str]:
    """Kontrollu deneyin degismeyen tarafi.

    Bunlari yazmak sekil degil: karsilastirmanin gecerliligi tam olarak bu
    listenin dogru olmasina baglidir.
    """
    satir = _defter().get(senaryo, {})
    konfig = _konfig(senaryo)
    sabitler = [
        "Degerlendirme seti: val_diagnostic (kilitli, hic degismez)",
        f"Cikarim cozunurlugu: {satir.get('imgsz_eval', '?')} px",
        f"Baslangic modeli: {satir.get('model', '?')}",
        f"Seed: {satir.get('seed', '?')}",
    ]
    if konfig.get("hedef_split") == "train":
        sabitler.append("Yalnizca train bolumu degistirildi; val ve test dokunulmadi")
    if _katalog().get(_kod(senaryo), {}).get("tur") == "veri":
        sabitler.append("Egitim protokolu sabit (senaryolar/egitim_protokolu.yaml)")
    return sabitler


def ne_gozlendi(senaryo: str) -> dict[str, Any]:
    """Referansa gore genel metrik farklari ve gurultu bandi degerlendirmesi."""
    from .gurultu import alt_grup_bandi  # dairesel import olmasin diye burada

    satir = _defter().get(senaryo)
    if satir is None:
        return {}
    ref = _referans()

    kontroller = [
        s for ad, s in _defter().items()
        if (ad.startswith("C") and ad[1:2].isdigit())
        and s["weights_path"].endswith("best.pt")
    ]
    esikler = {
        m: max((abs(float(k[m]) - ref[m]) for k in kontroller), default=None)
        for m in GENEL_METRIKLER
    }

    metrikler = {}
    for m in GENEL_METRIKLER:
        deger = float(satir[m])
        fark = deger - ref[m]
        esik = esikler[m]
        metrikler[m] = {
            "deger": round(deger, 4),
            "referans": round(ref[m], 4),
            "fark": round(fark, 4),
            "gurultu_esigi": round(esik, 4) if esik is not None else None,
            "asiyor": (abs(fark) > esik) if esik is not None else None,
        }
    return {
        "metrikler": metrikler,
        "kontrol_kosu_sayisi": len(kontroller),
        "asan_metrikler": [m for m, d in metrikler.items() if d["asiyor"]],
    }


def kanit_gucu(senaryo: str) -> dict[str, Any]:
    """Bulgunun ne kadar guclu oldugunu tek bakista soyler."""
    gozlem = ne_gozlendi(senaryo)
    if not gozlem:
        return {"seviye": "olcum yok", "aciklama": "Bu kosu defterde bulunamadi."}

    asan = gozlem["asan_metrikler"]
    n = gozlem["kontrol_kosu_sayisi"]
    if not asan:
        return {
            "seviye": "gurultu icinde",
            "aciklama": (
                "Hicbir genel metrik, bozulmasiz kosular arasinda gozlenen "
                f"yayilimi asmiyor ({n} kontrol kosusu). Bu senaryonun genel "
                "metriklerine dayanan bir iddia kurulamaz."
            ),
        }
    if len(asan) == 1:
        return {
            "seviye": "zayif",
            "aciklama": (
                f"Yalnizca {asan[0]} gurultu esigini asiyor. Tek metrige dayanan "
                "bir bulgu, kirilim kanitiyla desteklenmedikce zayiftir."
            ),
        }
    return {
        "seviye": "guclu",
        "aciklama": (
            f"{len(asan)} genel metrik gurultu esigini asiyor: {', '.join(asan)}."
        ),
    }


def sinirlamalar(senaryo: str) -> list[str]:
    """Bu senaryo icin gecerli, VERIDEN turetilen sinirlar."""
    from .bootstrap import VAL_DIAGNOSTIC_BBOX_N

    satir = _defter().get(senaryo, {})
    sinirlar = []

    az = [f"{ad} (n={n})" for ad, n in VAL_DIAGNOSTIC_BBOX_N.items() if n < 30]
    if az:
        sinirlar.append(
            f"Dusuk ornekli siniflar: {', '.join(az)}. Bu siniflardaki oranlar "
            "genellenemez."
        )
    gozlem = ne_gozlendi(senaryo)
    if gozlem:
        sinirlar.append(
            f"Gurultu esigi {gozlem['kontrol_kosu_sayisi']} kontrol kosusundan "
            "hesaplandi; az gozlemle esik gercek yayilimi oldugundan kucuk gosterir."
        )
    if satir.get("evaluation_set") != "val_diagnostic":
        sinirlar.append(
            f"Bu kosu kilitli set yerine '{satir.get('evaluation_set')}' uzerinde "
            "olculdu; digerleriyle dogrudan karsilastirilamaz."
        )
    if not satir.get("weights_path", "").endswith("best.pt"):
        sinirlar.append(
            "Bu satir last.pt checkpoint'ine aittir; best.pt satiriyla birlikte "
            "okunmalidir."
        )
    return sinirlar


def ozet(senaryo: str) -> dict[str, Any]:
    """Senaryo sayfasinin tum icerigi, tek cagriyla."""
    return {
        "senaryo": senaryo,
        "ne_olcuyor": _anlatim().get(_kod(senaryo)),
        "ne_degisti": ne_degisti(senaryo),
        "ne_sabit_kaldi": ne_sabit_kaldi(senaryo),
        "beklenen_etki": _konfig(senaryo).get("beklenen_kanit"),
        "ne_gozlendi": ne_gozlendi(senaryo),
        "kanit_gucu": kanit_gucu(senaryo),
        "sinirlamalar": sinirlamalar(senaryo),
    }

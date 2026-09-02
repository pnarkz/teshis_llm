"""C2: seed kaynakli dalgalanmayi olcer ve D serisi etkilerini ona karsi tartar.

Sartname bolum 8, C2'yi soyle tanimliyor: *"Ayni yapilandirma, seed 7 — Ajan
seed kaynakli dalgalanmayi 'sorun' saniyor mu?"*

Neden gerekli
-------------
Projedeki her kosu seed 42 ile egitilmisti. Bu, merkezi iddiayi sinayan
NEGATIF KONTROLUN eksik olmasi demekti ve daha temel bir sorun yaratiyordu:
"bozulma etkisi" diye raporlanan her seyin altinda duran gurultu tabani
bilinmiyordu.

Kontrol kosulari v00 ile BIREBIR ayni protokolu kullanir; tek degisken
seed'dir.

Tek cift yetmez
---------------
Ilk surumde yalnizca bir kontrol kosusu (seed 7) vardi ve "gurultu esigi"
o tek gozlemden aliniyordu. Bunun iki kusuru vardi:

1. Esik, kendisini kalibre etmesi gereken tek gozlemden turetildigi icin o
   gozlem trivially esige oturuyordu (C2'nin kendisi "esigi asiyor"
   gorunuyordu).
2. Tek fark, degiskenligin NOKTA TAHMINIDIR; yayilim hakkinda bilgi vermez.

Bu surum tum kontrol kosularini kullanir ve esigi **gozlenen en buyuk sapma**
olarak alir. Hala bir hipotez testi degildir - eleme suzgecidir - ama artik
birden fazla gozleme dayanir ve kac gozlemden geldigi raporda yazilidir.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SINIFLAR = ["tasit", "insan", "UAP", "UAI"]
GENEL = ["mAP50", "mAP50_95", "precision", "recall"]

REFERANS = ROOT / "reports/referans_v00/d1_metrics.json"
CIKTI = ROOT / "reports/kontrol_C2_seed7"

# E serisi disarida: onlar protokolu bozar. Bu tablo yalnizca
# "protokol sabit, veri degisti" kosularini tartar.
D_SERISI = ["D1", "D2a", "D2b", "D2b final_best", "D3", "D3b", "D4", "D5", "D6b"]


def _oku(yol: Path) -> dict:
    if not yol.is_file():
        raise FileNotFoundError(f"{yol} yok. Once kontrol kosusu degerlendirilmeli.")
    return json.loads(yol.read_text(encoding="utf-8"))


def _defter() -> dict[str, dict[str, str]]:
    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        return {s["scenario"]: s for s in csv.DictReader(f)}


def kontrol_kosulari() -> list[dict]:
    """v00 ile ayni protokolde, yalnizca seed'i farkli olan tum kosular.

    Adlandirma kurali geregi kontrol kosullari `C` ile baslar; defterdeki
    her boyle satir buraya girer. Yeni bir seed eklendiginde bu fonksiyon
    onu kendiliginden bulur - liste elle guncellenmez.
    """
    from teshis.degerlendirme.raporlar import rapor_klasoru

    kosular = []
    for senaryo, satir in _defter().items():
        if not senaryo.startswith("C") or not senaryo[1:2].isdigit():
            continue
        if not satir["weights_path"].endswith("best.pt"):
            continue
        klasor = rapor_klasoru(senaryo)
        if klasor is None:
            continue
        kosular.append({
            "senaryo": senaryo,
            "seed": int(satir["seed"]),
            "epoch": int(satir["epochs"]),
            "metrikler": _oku(klasor / "d1_metrics.json"),
        })
    return sorted(kosular, key=lambda k: k["seed"])


def gurultu_tabani(kontroller: list[dict], v00: dict) -> dict:
    """Her metrik icin gozlenen en buyuk sapmayi ve yayilimi dondurur."""
    taban: dict[str, dict] = {}
    for alan in GENEL:
        farklar = [k["metrikler"][alan] - v00[alan] for k in kontroller]
        taban[alan] = {
            "esik": round(max(abs(f) for f in farklar), 4),
            "farklar": [round(f, 4) for f in farklar],
            "std": round(statistics.stdev(farklar), 4) if len(farklar) > 1 else None,
        }
    for i, s in enumerate(SINIFLAR):
        farklar = [k["metrikler"]["class_ap50"][i] - v00["class_ap50"][i]
                   for k in kontroller]
        taban[f"AP_{s}"] = {
            "esik": round(max(abs(f) for f in farklar), 4),
            "farklar": [round(f, 4) for f in farklar],
            "std": round(statistics.stdev(farklar), 4) if len(farklar) > 1 else None,
        }
    return taban


def karsilastirma() -> dict:
    v00 = _oku(REFERANS)
    kontroller = kontrol_kosulari()
    if not kontroller:
        raise RuntimeError("Hicbir kontrol kosusu bulunamadi (senaryo adi C ile baslamali)")
    taban = gurultu_tabani(kontroller, v00)
    defter = _defter()

    senaryolar = []
    for ad in D_SERISI:
        s = defter.get(ad)
        if not s:
            continue
        farklar = {m: float(s[m]) - v00[m] for m in GENEL}
        asanlar = [m for m in GENEL if abs(farklar[m]) > taban[m]["esik"]]
        senaryolar.append({
            "senaryo": ad,
            "farklar": {m: round(v, 4) for m, v in farklar.items()},
            "gurultuyu_asan_metrikler": asanlar,
            "hicbiri_asmiyor": not asanlar,
        })

    epochlar = [k["epoch"] for k in kontroller] + [11]  # 11 = v00'un durdugu epoch
    return {
        "kontrol": "C2 (sartname bolum 8) - coklu seed",
        "tanim": "v00 ile birebir ayni protokol; tek degisken seed",
        "referans": {"seed": 42, "epoch": 11},
        "kontrol_kosulari": [
            {"senaryo": k["senaryo"], "seed": k["seed"], "epoch": k["epoch"]}
            for k in kontroller
        ],
        "gozlem_sayisi": len(kontroller),
        "not_erken_durdurma": (
            f"Kontrol kosulari {min(epochlar)} ile {max(epochlar)} epoch arasinda "
            "durdu. Erken durdurma noktasi bile seed'e baglidir; gurultu yalnizca "
            "son metrikte degil egitim suresinde de gorunur."
        ),
        "sinir": (
            f"n={len(kontroller)} kontrol kosusu. Esik, GOZLENEN EN BUYUK sapmadir; "
            "bir hipotez testi degil, eleme suzgecidir. Esigin altinda kalan bir "
            "etki 'seed degisiminden ayirt edilemez' demektir, 'etki yoktur' "
            "demek degildir."
        ),
        "gurultu_tabani": taban,
        "d_serisi": senaryolar,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yazma", action="store_true", help="yalnizca ekrana bas")
    args = parser.parse_args()

    sonuc = karsilastirma()
    taban = sonuc["gurultu_tabani"]

    print(f"=== Kontrol kosulari (n={sonuc['gozlem_sayisi']}) ===")
    for k in sonuc["kontrol_kosulari"]:
        print(f"  {k['senaryo']:<14} seed {k['seed']:<4} {k['epoch']} epoch")
    print(f"\n{sonuc['not_erken_durdurma']}\n")

    print("=== Seed gurultusu: v00'a gore farklar ===")
    print(f"{'metrik':<12} {'farklar':<32} {'esik':>8} {'std':>8}")
    for alan in GENEL + [f"AP_{s}" for s in SINIFLAR]:
        d = taban[alan]
        farklar = " ".join(f"{f:+.4f}" for f in d["farklar"])
        std = f"{d['std']:.4f}" if d["std"] is not None else "-"
        print(f"{alan:<12} {farklar:<32} {d['esik']:>8.4f} {std:>8}")

    print("\n=== D serisi etkileri bu gurultuyu asiyor mu? ===")
    print(f"{'senaryo':<16} {'dmAP50':>9} {'dprec':>9} {'drecall':>9}   asan metrikler")
    print(f"{'esik':<16} {taban['mAP50']['esik']:>9.4f} "
          f"{taban['precision']['esik']:>9.4f} {taban['recall']['esik']:>9.4f}")
    for d in sonuc["d_serisi"]:
        f = d["farklar"]
        asan = ", ".join(d["gurultuyu_asan_metrikler"]) or ">>> HICBIRI <<<"
        print(f"{d['senaryo']:<16} {f['mAP50']:>+9.4f} {f['precision']:>+9.4f} "
              f"{f['recall']:>+9.4f}   {asan}")

    print(f"\nSINIR: {sonuc['sinir']}")

    if not args.yazma:
        CIKTI.mkdir(parents=True, exist_ok=True)
        hedef = CIKTI / "c2_seed_gurultusu.json"
        hedef.write_text(json.dumps(sonuc, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
        print(f"\nRapor: {hedef.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""C2: seed kaynakli dalgalanmayi olcer ve D serisi etkilerini ona karsi tartar.

Sartname bolum 8, C2'yi soyle tanimliyor: *"Ayni yapilandirma, seed 7 — Ajan
seed kaynakli dalgalanmayi 'sorun' saniyor mu?"*

Neden gerekli
-------------
Projedeki her kosu seed 42 ile egitilmisti. Bu, merkezi iddiayi
("ajan bozulmayi metrik imzasindan teshis edebilir") sinayan NEGATIF
KONTROLUN eksik olmasi demekti: hicbir bozulma icermeyen, yalnizca farkli
seed ile egitilmis bir kosuda ajanin sorun uydurup uydurmadigini
bilmiyorduk. Bir teshis sisteminin yanlis pozitif orani olculmeden dogruluk
iddiasi eksik kalir.

C2, v00 ile BIREBIR ayni protokolu kullanir; tek degisken seed'dir. Aradaki
fark, "bozulma etkisi" diye raporlanan her seyin altinda duran gurultu
tabanidir.

Onemli sinir: n=1
-----------------
Tek bir seed cifti, degiskenligin NOKTA TAHMINIDIR; dagilimi degil. Bu
scriptin urettigi "gurultu esigi" bir hipotez testi degil, bir ELEME
suzgecidir: esigin altinda kalan bir etki, seed degisiminden ayirt
edilemez demektir - "etki yoktur" demek degildir. Kesin konusmak icin
birkac seed daha gerekir (v00 seed 7, 13, 21... ile tekrarlanmali).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SINIFLAR = ["tasit", "insan", "UAP", "UAI"]
GENEL = ["mAP50", "mAP50_95", "precision", "recall"]

REFERANS = ROOT / "reports/referans_v00/d1_metrics.json"
KONTROL = ROOT / "reports/kontrol_C2_seed7/d1_metrics.json"
CIKTI = ROOT / "reports/kontrol_C2_seed7"

# results.csv'deki senaryo adlari. E serisi disarida: onlar protokolu bozar,
# bu tablo yalnizca "protokol sabit, veri degisti" kosularini tartar.
D_SERISI = ["D1", "D2a", "D2b", "D2b final_best", "D3", "D3b", "D4", "D5", "D6b"]


def _oku(yol: Path) -> dict:
    if not yol.is_file():
        raise FileNotFoundError(f"{yol} yok. Once C2 kosusu degerlendirilmeli.")
    return json.loads(yol.read_text(encoding="utf-8"))


def gurultu_tabani() -> dict[str, float]:
    """v00 ile C2 arasindaki mutlak farklar: saf seed degiskenligi."""
    v00, c2 = _oku(REFERANS), _oku(KONTROL)
    taban = {ad: abs(c2[ad] - v00[ad]) for ad in GENEL}
    for i, s in enumerate(SINIFLAR):
        taban[f"AP_{s}"] = abs(c2["class_ap50"][i] - v00["class_ap50"][i])
    return taban


def karsilastirma() -> dict:
    v00, c2 = _oku(REFERANS), _oku(KONTROL)
    taban = gurultu_tabani()

    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        satirlar = {s["scenario"]: s for s in csv.DictReader(f)}

    senaryolar = []
    for ad in D_SERISI:
        s = satirlar.get(ad)
        if not s:
            continue
        farklar = {m: float(s[m.replace("mAP50_95", "mAP50_95")]) - v00[m] for m in GENEL}
        asanlar = [m for m in GENEL if abs(farklar[m]) > taban[m]]
        senaryolar.append({
            "senaryo": ad,
            "farklar": {m: round(v, 4) for m, v in farklar.items()},
            "gurultuyu_asan_metrikler": asanlar,
            "hicbiri_asmiyor": not asanlar,
        })

    return {
        "kontrol": "C2 (sartname bolum 8)",
        "tanim": "v00 ile birebir ayni protokol; tek degisken seed (42 -> 7)",
        "referans_kosu": {"seed": 42, "epoch": 11, "kaynak": str(REFERANS.relative_to(ROOT))},
        "kontrol_kosusu": {"seed": 7, "epoch": 19, "kaynak": str(KONTROL.relative_to(ROOT))},
        "not_erken_durdurma": (
            "Iki kosu farkli epoch'ta durdu (11 vs 19). Erken durdurma noktasi "
            "bile seed'e baglidir; bu, gurultunun yalnizca son metrikte degil "
            "egitim suresinde de gorundugunu gosterir."
        ),
        "sinir": (
            "n=1. Tek seed cifti degiskenligin nokta tahminidir, dagilimi degil. "
            "Asagidaki esik bir hipotez testi DEGIL, eleme suzgecidir: esigin "
            "altinda kalan etki seed degisiminden ayirt edilemez demektir, "
            "'etki yoktur' demek degildir."
        ),
        "gurultu_tabani": {k: round(v, 4) for k, v in taban.items()},
        "genel_farklar": {m: round(c2[m] - v00[m], 4) for m in GENEL},
        "sinif_ap_farklari": {
            s: round(c2["class_ap50"][i] - v00["class_ap50"][i], 4)
            for i, s in enumerate(SINIFLAR)
        },
        "d_serisi": senaryolar,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yazma", action="store_true", help="yalnizca ekrana bas")
    args = parser.parse_args()

    sonuc = karsilastirma()
    taban = sonuc["gurultu_tabani"]

    print("=== C2 (seed 7) vs v00 (seed 42): saf seed gurultusu ===")
    for m in GENEL:
        print(f"  {m:<12} {sonuc['genel_farklar'][m]:>+8.4f}")
    for s, v in sonuc["sinif_ap_farklari"].items():
        print(f"  AP {s:<9} {v:>+8.4f}")

    print("\n=== D serisi etkileri bu gurultuyu asiyor mu? ===")
    print(f"{'senaryo':<16} {'dmAP50':>9} {'dprec':>9} {'drecall':>9}   asan metrikler")
    print(f"{'esik (C2)':<16} {taban['mAP50']:>9.4f} {taban['precision']:>9.4f} "
          f"{taban['recall']:>9.4f}")
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

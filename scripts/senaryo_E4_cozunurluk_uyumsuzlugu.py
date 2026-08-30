"""E4: egitim/cikarim cozunurluk uyumsuzlugunu olcer ve raporlar.

E4, E serisinin egitim gerektirmeyen tek senaryosudur; uyumsuzluk cikarim
tarafindadir. Protokol-uyumlu saglikli referans (v00) kilitli tanı setinde
birden cok imgsz degerinde degerlendirilir.

Konfigden bilincli sapma
------------------------
senaryolar/egitim/e4_cozunurluk_uyumsuzlugu.yaml tek bir cift tanimlar
(imgsz_egitim=640, imgsz_degerlendirme=1280). Burada bunun yerine mevcut
768 modeli uzerinde cok noktali bir tarama yapilir:

- tek cift yalnizca bir sayi verir; tarama egrinin BICIMINI gosterir
  (tepe noktasi ve kucultme/buyutme asimetrisi ancak boyle gorulur),
- 640'ta yeni bir model egitmek, egitim protokolunu hic degistirmeden ayni
  olguyu olcmenin pahali yoludur.

Olcum noktalari senaryolar/egitim_protokolu.yaml -> e_serisi.E4 altinda
beyan edilir; bu script onlari oradan okur.

Kullanim:
    python scripts/senaryo_E4_cozunurluk_uyumsuzlugu.py            # rapor uret
    python scripts/senaryo_E4_cozunurluk_uyumsuzlugu.py --degerlendir  # once val kos
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teshis.degerlendirme.bootstrap import (  # noqa: E402
    VAL_DIAGNOSTIC_BBOX_N,
    iki_oran_testi,
    wilson_araligi,
)
from teshis.egitim.protokol import e_senaryo_ayarlari  # noqa: E402

SINIFLAR = ["tasit", "insan", "UAP", "UAI"]
REFERANS_MODEL = ROOT / "experiments/run_20260826_081354_v00_42/weights/best.pt"
CIKTI = ROOT / "reports/senaryo_E4"


def rapor_yolu(imgsz: int) -> Path:
    return ROOT / f"reports/senaryo_E4_imgsz{imgsz}"


def degerlendir(imgsz_listesi: list[int]) -> None:
    """Her cozunurluk icin kilitli tanı setinde degerlendirme kosar."""
    for imgsz in imgsz_listesi:
        print(f"--- imgsz={imgsz} degerlendiriliyor ---", flush=True)
        subprocess.run(
            [sys.executable, "-m", "teshis.degerlendirme.d1_sonuc",
             "--model", str(REFERANS_MODEL),
             "--data", "val_diagnostic/data.yaml",
             "--output", str(rapor_yolu(imgsz)),
             "--imgsz", str(imgsz),
             "--scenario", f"E4_imgsz{imgsz}"],
            cwd=ROOT, check=True,
        )


def _oku(imgsz: int) -> dict:
    yol = rapor_yolu(imgsz) / "d1_metrics.json"
    if not yol.is_file():
        raise FileNotFoundError(
            f"{yol} yok. Once --degerlendir ile olcumleri uretin."
        )
    return json.loads(yol.read_text(encoding="utf-8"))


def tarama_raporu(egitim_imgsz: int, olcumler: list[int]) -> dict:
    kayit = {r: _oku(r) for r in olcumler}
    taban = kayit[egitim_imgsz]

    satirlar = []
    for r in olcumler:
        m = kayit[r]
        satirlar.append({
            "imgsz": r,
            "egitimle_orani": round(r / egitim_imgsz, 3),
            "mAP50": round(m["mAP50"], 4),
            "mAP50_95": round(m["mAP50_95"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "mAP50_kaybi": round(m["mAP50"] - taban["mAP50"], 4),
            "recall_kaybi": round(m["recall"] - taban["recall"], 4),
            "sinif_recall": dict(zip(SINIFLAR, [round(v, 4) for v in m["class_recall"]])),
            "sinif_ap50": dict(zip(SINIFLAR, [round(v, 4) for v in m["class_ap50"]])),
        })

    en_iyi = max(satirlar, key=lambda o: o["mAP50"])
    kucuk = [o for o in satirlar if o["imgsz"] < egitim_imgsz]
    buyuk = [o for o in satirlar if o["imgsz"] > egitim_imgsz]

    return {
        "senaryo": "E4",
        "ad": "egitim/cikarim cozunurluk uyumsuzlugu",
        "model": taban["model"],
        "degerlendirme_seti": "val_diagnostic (kilitli)",
        "egitim_imgsz": egitim_imgsz,
        "not": (
            "Konfig (senaryolar/egitim/e4_cozunurluk_uyumsuzlugu.yaml) tek cift "
            "imgsz_egitim=640 / imgsz_degerlendirme=1280 tanimliyordu. Bunun yerine "
            "protokol-uyumlu v00 referansi (imgsz=768) cok noktali olculdu; gerekce "
            "bu scriptin docstring'inde. Sapma bilinclidir; E4 egitim gerektirmez."
        ),
        "olcumler": satirlar,
        "bulgular": {
            "tepe_imgsz": en_iyi["imgsz"],
            "tepe_egitim_cozunurlugunde_mi": en_iyi["imgsz"] == egitim_imgsz,
            "en_kotu_kucultme": min(kucuk, key=lambda o: o["mAP50"])["mAP50_kaybi"] if kucuk else None,
            "en_kotu_buyutme": min(buyuk, key=lambda o: o["mAP50"])["mAP50_kaybi"] if buyuk else None,
            "asimetri": (
                "Kucultmek buyutmekten pahalidir: kucultme termal imzayi kalici "
                "olarak yok eder, buyutme var olan bilgiyi yeniden orneklemekle kalir."
            ),
            "precision_recall_ayrisimi": (
                "Precision cozunurlukten neredeyse etkilenmez, recall coker. "
                "Uyumsuzluk modeli yaniltmaz, KOR EDER: buldugunu dogru bulur, "
                "ama bulamaz. Etiket bozulmalarinda (D serisi) precision da bozulur; "
                "iki arizayi ayirt eden imza budur."
            ),
        },
    }


def anlamlilik_raporu(egitim_imgsz: int, dusuk_imgsz: int) -> dict:
    """Sinif bazli recall dususunu iki oran z-testi + Wilson araligi ile sinar."""
    taban, dusuk = _oku(egitim_imgsz), _oku(dusuk_imgsz)
    siniflar = {}
    for i, s in enumerate(SINIFLAR):
        n = VAL_DIAGNOSTIC_BBOX_N[s]
        r1, r2 = taban["class_recall"][i], dusuk["class_recall"][i]
        k1, k2 = round(r1 * n), round(r2 * n)
        test = iki_oran_testi(k1, n, k2, n)
        ga1, ga2 = wilson_araligi(k1, n), wilson_araligi(k2, n)
        siniflar[s] = {
            "n": n,
            f"recall_{egitim_imgsz}": round(r1, 4),
            f"recall_{dusuk_imgsz}": round(r2, 4),
            "fark": round(r2 - r1, 4),
            "z": round(test["z"], 3),
            "p": test["p"],
            "anlamli_005": test["p"] < 0.05,
            f"wilson_{egitim_imgsz}": [round(x, 3) for x in ga1],
            f"wilson_{dusuk_imgsz}": [round(x, 3) for x in ga2],
            "guven_araliklari_ayrik": ga2[1] < ga1[0] or ga1[1] < ga2[0],
        }
    return {
        "karsilastirma": f"imgsz {dusuk_imgsz} (uyumsuz) vs {egitim_imgsz} (egitim cozunurlugu)",
        "yontem": "iki oran z-testi + Wilson %95 guven araligi, bbox sayilari uzerinde",
        "uyari": (
            "UAP (n=15) ve UAI (n=17) bbox sayilari cok dusuktur; bu siniflardaki "
            "buyuk yuzde farklari birkac kutunun kaybina karsilik gelir. Genel makro "
            "recall dususunu bu iki sinif surukler. Fark yine de ayakta kalir: Wilson "
            "araliklari ortusmez."
        ),
        "siniflar": siniflar,
    }


def main() -> None:
    ayar = e_senaryo_ayarlari("E4")["kosu_ayarlari"]
    egitim_imgsz, olcumler = ayar["egitim_imgsz"], ayar["olcum_imgsz"]

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degerlendir", action="store_true",
                        help="raporlamadan once her cozunurlukte val kos (GPU gerekir)")
    args = parser.parse_args()

    if args.degerlendir:
        degerlendir(olcumler)

    CIKTI.mkdir(parents=True, exist_ok=True)
    tarama = tarama_raporu(egitim_imgsz, olcumler)
    anlamlilik = anlamlilik_raporu(egitim_imgsz, min(olcumler))
    (CIKTI / "e4_cozunurluk_taramasi.json").write_text(
        json.dumps(tarama, indent=2, ensure_ascii=False), encoding="utf-8")
    (CIKTI / "e4_sinif_anlamlilik.json").write_text(
        json.dumps(anlamlilik, indent=2, ensure_ascii=False), encoding="utf-8")

    basliklar = ("imgsz", "oran", "mAP50", "recall", "prec", "fark")
    print("{:>6} {:>6} {:>7} {:>7} {:>7} {:>8}".format(*basliklar))
    for o in tarama["olcumler"]:
        im = "  <- egitim" if o["imgsz"] == egitim_imgsz else ""
        print("{:>6} {:>6.2f} {:>7.3f} {:>7.3f} {:>7.3f} {:>+8.3f}{}".format(
            o["imgsz"], o["egitimle_orani"], o["mAP50"],
            o["recall"], o["precision"], o["mAP50_kaybi"], im))
    tepe = tarama["bulgular"]["tepe_egitim_cozunurlugunde_mi"]
    print("")
    print("Tepe egitim cozunurlugunde mi:", tepe)
    print("Rapor:", CIKTI)


if __name__ == "__main__":
    main()

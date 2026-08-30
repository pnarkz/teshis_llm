"""Oran karsilastirmalari icin guven araligi ve anlamlilik hesaplari.

Bu modul bir tekrarlanabilirlik bosluğunu kapatir: README'deki z degerleri ve
%95 guven araliklari (orn. "UAI recall -0.6123, z=-3,74" veya "%95 GA =
[0.413, 0.827]") tek seferlik scriptlerde hesaplanmisti ve repoda karsiligi
yoktu. Kimse dogrulayamaz veya yeniden uretemezdi. Artik tum bu sayilar
buradaki fonksiyonlardan uretilir ve testlidir.

Uc olcum saglanir:

1. ``wilson_araligi``  : bir oranin %95 guven araligi. Normal yaklasima gore
   kucuk orneklemde cok daha guvenilirdir; UAP/UAI gibi n=15-17 siniflarda
   fark kritiktir.
2. ``iki_oran_testi``  : iki orani karsilastiran z testi. Kutulari bagimsiz
   sayar; hizli ve seffaftir ama asagidaki sinirlamaya bakin.
3. ``goruntu_bootstrap``: GORUNTU birimli bootstrap. Ayni goruntudeki kutular
   bagimsiz degildir (ayni sahne, ayni sensor, ayni kosullar); kutu birimli
   testler bu yuzden guven araligini oldugundan DAR gosterir. Bootstrap,
   yeniden ornekleme birimi olarak goruntuyu alarak bu bagimliligi hesaba
   katar ve daha durustur.

4. ``tabakali_goruntu_bootstrap``: sartnamedeki C3 protokolunun tam
   karsiligi. Goruntu birimlidir VE kaynak grubuna gore tabakalidir, yani
   her kaynagin payi her turda korunur. B=1000, yuzdelik yontemiyle %95.
5. ``yontem_karsilastir``: ayni veri icin Wilson ve tabakali bootstrap
   araliklarini yan yana koyar; ``genislik_orani`` 1'den ne kadar buyukse
   kutu birimli aralik o kadar dar kalmistir.

Hangisi kullanilmali: yon ve kaba anlamlilik icin ``iki_oran_testi`` yeterli;
bir sayiyi rapora manset olarak koyacaksaniz ``tabakali_goruntu_bootstrap``
tercih edilmelidir.

OLCULEN FARK (tests/test_bootstrap.py). Kutulari kareler icinde kumelendikce
Wilson araligi gercege gore daralir:

    kare basina 1 kutu (kumelenme yok) ... bootstrap / Wilson = 1.02x
    kare basina 3-4 kutu (gercekci)  ..... 1.50x
    kare basina 5 kutu  .................. 2.19x
    kare basina 10 kutu .................. 2.91x

DIKKAT: docs/BULGULAR.md'de yayimlanmis guven araliklarinin buyuk kismi
Wilson ile hesaplandi; yani muhtemelen oldugundan DARDIR. Farkin sonuca
etkisi, ``goruntu_kayitlari`` iceren yeni kirilim olcumleri uretildikce
kirilim bazinda yeniden degerlendirilmelidir. Aralari genis olan bulgular
(orn. E4'te UAP ve UAI) bundan etkilenmez; sinirda olanlar (orn. E4'te
tasit) yeniden kontrol edilmelidir.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

VARSAYILAN_Z = 1.96  # %95


def wilson_araligi(basari: int, deneme: int, z: float = VARSAYILAN_Z) -> tuple[float, float]:
    """Bir oranin Wilson %95 guven araligini dondurur.

    Normal (Wald) yaklasimi kucuk orneklemde arali sinirlarin disina tasirir
    ve 0/n ya da n/n durumlarinda sifir genislik verir; Wilson bu iki sorunu
    da yasamaz. UAP (n=15) ve UAI (n=17) gibi siniflarda bu fark belirleyicidir.
    """
    if deneme <= 0:
        raise ValueError("deneme sayisi pozitif olmalidir")
    if not 0 <= basari <= deneme:
        raise ValueError(f"basari 0..{deneme} araliginda olmalidir, verilen: {basari}")
    oran = basari / deneme
    payda = 1 + z * z / deneme
    merkez = (oran + z * z / (2 * deneme)) / payda
    yariCap = z * math.sqrt(oran * (1 - oran) / deneme + z * z / (4 * deneme * deneme)) / payda
    return max(0.0, merkez - yariCap), min(1.0, merkez + yariCap)


def iki_oran_testi(
    basari_a: int, deneme_a: int, basari_b: int, deneme_b: int
) -> dict[str, float]:
    """Iki orani karsilastiran z testi; z, p ve fark dondurur.

    Kutulari bagimsiz varsayar. Ayni goruntudeki kutular korele oldugu icin
    bu varsayim iyimserdir; kesin bir manset sayi icin goruntu_bootstrap
    kullanin.
    """
    if deneme_a <= 0 or deneme_b <= 0:
        raise ValueError("deneme sayilari pozitif olmalidir")
    oran_a, oran_b = basari_a / deneme_a, basari_b / deneme_b
    havuz = (basari_a + basari_b) / (deneme_a + deneme_b)
    standart_hata = math.sqrt(havuz * (1 - havuz) * (1 / deneme_a + 1 / deneme_b))
    z = (oran_a - oran_b) / standart_hata if standart_hata > 0 else 0.0
    # Iki yonlu p degeri; math.erfc ile normal kuyruk.
    p = math.erfc(abs(z) / math.sqrt(2))
    return {
        "oran_a": oran_a,
        "oran_b": oran_b,
        "fark": oran_a - oran_b,
        "z": z,
        "p": p,
        "anlamli_005": p < 0.05,
    }


def goruntu_bootstrap(
    goruntu_kayitlari: Sequence[tuple[int, int]],
    tekrar: int = 10000,
    seed: int = 42,
    z_yuzde: float = 95.0,
) -> dict[str, float]:
    """Goruntu birimli bootstrap ile bir oranin guven araligini hesaplar.

    goruntu_kayitlari: her goruntu icin ``(yakalanan, toplam)`` cifti.

    Yeniden ornekleme birimi GORUNTUDUR: her turda goruntuler yerine koyarak
    secilir ve oran, secilen goruntulerin toplamlarindan hesaplanir. Boylece
    ayni goruntudeki kutularin korelasyonu korunur. Kutu birimli bir bootstrap
    bu korelasyonu yok sayar ve araligi oldugundan dar gosterir.
    """
    kayitlar = [(int(y), int(t)) for y, t in goruntu_kayitlari if t > 0]
    if not kayitlar:
        raise ValueError("en az bir goruntu kaydi gerekir")
    if tekrar < 1:
        raise ValueError("tekrar pozitif olmalidir")

    toplam_yakalanan = sum(y for y, _ in kayitlar)
    toplam_kutu = sum(t for _, t in kayitlar)
    nokta = toplam_yakalanan / toplam_kutu

    rng = random.Random(seed)
    n = len(kayitlar)
    oranlar: list[float] = []
    for _ in range(tekrar):
        y_top = t_top = 0
        for _ in range(n):
            y, t = kayitlar[rng.randrange(n)]
            y_top += y
            t_top += t
        if t_top:
            oranlar.append(y_top / t_top)
    oranlar.sort()
    alt_yuzde = (100 - z_yuzde) / 2
    alt = oranlar[max(0, int(len(oranlar) * alt_yuzde / 100) - 1)]
    ust = oranlar[min(len(oranlar) - 1, int(len(oranlar) * (100 - alt_yuzde) / 100))]
    return {
        "oran": nokta,
        "alt": alt,
        "ust": ust,
        "genislik": ust - alt,
        "goruntu_sayisi": n,
        "kutu_sayisi": toplam_kutu,
        "tekrar": tekrar,
    }


def kirilim_karsilastir(
    kosu_json: Path, referans_json: Path, alan: str = "sinif_recall"
) -> list[dict[str, Any]]:
    """Iki kirilim analizini karsilastirip her grup icin istatistik uretir.

    metrikler.py ciktilarindaki ``gercek_kutu``/``yakalanan`` sayimlarini
    kullanir; boylece README'deki z degerleri ve guven araliklari koddan
    yeniden uretilebilir.
    """
    kosu = json.loads(Path(kosu_json).read_text(encoding="utf-8"))
    referans = json.loads(Path(referans_json).read_text(encoding="utf-8"))
    if alan not in kosu or alan not in referans:
        raise KeyError(f"'{alan}' iki dosyada da bulunmali. Mevcut: {sorted(kosu)}")

    satirlar: list[dict[str, Any]] = []
    for grup, deger in kosu[alan].items():
        taban = referans[alan].get(grup)
        if taban is None or not deger["gercek_kutu"]:
            continue
        test = iki_oran_testi(
            deger["yakalanan"], deger["gercek_kutu"],
            taban["yakalanan"], taban["gercek_kutu"],
        )
        alt, ust = wilson_araligi(deger["yakalanan"], deger["gercek_kutu"])
        satirlar.append({
            "grup": grup,
            "bbox_n": deger["gercek_kutu"],
            "recall": deger["recall"],
            "referans_recall": taban["recall"],
            "fark": round(test["fark"], 4),
            "z": round(test["z"], 2),
            "p": round(test["p"], 5),
            "anlamli_005": test["anlamli_005"],
            "wilson_alt": round(alt, 4),
            "wilson_ust": round(ust, 4),
            "wilson_genislik": round(ust - alt, 4),
        })
    return sorted(satirlar, key=lambda s: -s["bbox_n"])


# Kilitli tanı setindeki sinif basina bbox sayisi (README / PROJECT_STRUCTURE).
VAL_DIAGNOSTIC_BBOX_N: dict[str, int] = {"tasit": 1264, "insan": 2718, "UAP": 15, "UAI": 17}


def sinif_metrigi_karsilastir(
    kosu_json: Path, referans_json: Path, alan: str = "class_recall"
) -> list[dict[str, Any]]:
    """Iki ``degerlendirme/d1_sonuc.py`` ciktisi olan metrik dosyasindaki sinif oranlarini karsilastirir.

    ``kirilim_karsilastir`` metrikler.py'nin kendi eslestirmesini kullanir;
    bu fonksiyon ise Ultralytics'in raporladigi sinif recall/precision
    degerlerini alir. README tablolarindaki z degerleri bu yolla uretilmistir,
    bu yuzden ikisi ayri ayri saglanir ve karistirilmamalidir.

    Oranlar sayim degil yuzde olarak geldigi icin, bbox sayisi kilitli tanı
    setinin sabit degerlerinden alinir ve basari sayisi geri hesaplanir.

    DIKKAT - yuvarlama: iki oran testi SAYIMLAR uzerinde tanimlidir, bu yuzden
    z, geri turetilen tam sayilardan hesaplanir. Kucuk n'de bu, ham oranla
    kucuk bir fark yaratabilir (orn. UAI n=17: ham oran 0.3289, 6/17 = 0.3529;
    z -3,74 yerine -3,59). Sayim tabanli deger daha tutarlidir. Ayrica
    Ultralytics'in raporladigi recall en iyi F1 esiginde hesaplandigi icin tam
    bir k/n oranina karsilik gelmeyebilir; bu yontem bir yaklasimdir.
    Kesin bir manset sayi icin ``goruntu_bootstrap`` tercih edilmelidir.
    """
    kosu = json.loads(Path(kosu_json).read_text(encoding="utf-8"))
    referans = json.loads(Path(referans_json).read_text(encoding="utf-8"))
    if alan not in kosu or alan not in referans:
        raise KeyError(f"'{alan}' iki dosyada da bulunmali (orn. class_recall).")

    kosu_oran = dict(zip(kosu["class_names"], kosu[alan]))
    ref_oran = dict(zip(referans["class_names"], referans[alan]))
    satirlar: list[dict[str, Any]] = []
    for sinif, n in VAL_DIAGNOSTIC_BBOX_N.items():
        if sinif not in kosu_oran or sinif not in ref_oran:
            continue
        basari_k, basari_r = round(kosu_oran[sinif] * n), round(ref_oran[sinif] * n)
        test = iki_oran_testi(basari_k, n, basari_r, n)
        alt, ust = wilson_araligi(basari_k, n)
        satirlar.append({
            "grup": sinif,
            "bbox_n": n,
            "recall": round(kosu_oran[sinif], 4),
            "referans_recall": round(ref_oran[sinif], 4),
            "fark": round(kosu_oran[sinif] - ref_oran[sinif], 4),
            "z": round(test["z"], 2),
            "p": round(test["p"], 5),
            "anlamli_005": test["anlamli_005"],
            "wilson_alt": round(alt, 4),
            "wilson_ust": round(ust, 4),
            "wilson_genislik": round(ust - alt, 4),
        })
    return sorted(satirlar, key=lambda s: -s["bbox_n"])


# --------------------------------------------------------------------------
# C3 protokolu (sartname bolum 8): tabakali, goruntu birimli bootstrap
#
# Sartname acikca soyle diyor:
#   - Yeniden ornekleme birimi GORUNTUDUR (kutu degil). Ayni karedeki kutular
#     iliskilidir; kutu bazli bootstrap guven araligini yapay olarak DARALTIR.
#   - Kaynak grubu bazinda TABAKALI ornekleme yapilir; her grubun payi korunur.
#   - Yuzdelik yontemiyle %95 guven araligi.
#
# Projede yayimlanmis araliklarin bir kismi kutu sayilari uzerinde Wilson ile
# hesaplandi. Wilson, kutulari bagimsiz varsayar; sartnamenin uyardigi
# daralmayi tam olarak uretir. Bu fonksiyon sartnameye uygun alternatifi
# saglar, boylece iki yontem ayni veri uzerinde karsilastirilabilir.
# --------------------------------------------------------------------------


def tabakali_goruntu_bootstrap(
    kayitlar: Sequence[dict[str, Any]],
    alan: str = "sinif",
    grup: str | None = None,
    tekrar: int = 1000,
    seed: int = 42,
    yuzde: float = 95.0,
) -> dict[str, Any]:
    """Kaynak grubuna gore tabakali, goruntu birimli bootstrap guven araligi.

    kayitlar: metrikler.py ciktisindaki ``goruntu_kayitlari`` listesi. Her oge
        ``{"goruntu", "kaynak", "sinif": {ad: [yakalanan, toplam]}, "bant": {...}}``.
    alan: hangi kirilim uzerinde calisilacagi ("sinif" veya "bant").
    grup: tek bir grubun (orn. "insan") recall'u; None ise tum kutular birlikte.

    Tabakalama: her kaynak grubu KENDI icinde, kendi goruntu sayisi kadar
    yeniden orneklenir. Boylece bir turda bir kaynagin payi buyuyup digerinin
    kuculemez; degisenlik yalnizca grup ICI ornekleme belirsizliginden gelir.
    """
    if not kayitlar:
        raise ValueError("en az bir goruntu kaydi gerekir")
    if tekrar < 1:
        raise ValueError("tekrar pozitif olmalidir")

    def ikili(kayit: dict[str, Any]) -> tuple[int, int]:
        kirilim = kayit.get(alan) or {}
        if grup is not None:
            yakalanan, toplam = kirilim.get(grup, [0, 0])
            return int(yakalanan), int(toplam)
        return (sum(int(v[0]) for v in kirilim.values()),
                sum(int(v[1]) for v in kirilim.values()))

    # Kaynak grubuna gore tabakalar
    tabakalar: dict[str, list[tuple[int, int]]] = {}
    for kayit in kayitlar:
        tabakalar.setdefault(str(kayit.get("kaynak", "bilinmiyor")), []).append(ikili(kayit))

    toplam_yakalanan = sum(y for grup_kayit in tabakalar.values() for y, _ in grup_kayit)
    toplam_kutu = sum(t for grup_kayit in tabakalar.values() for _, t in grup_kayit)
    if not toplam_kutu:
        raise ValueError(f"'{grup}' icin hic kutu yok; guven araligi tanimsiz")

    rng = random.Random(seed)
    oranlar: list[float] = []
    for _ in range(tekrar):
        y_top = t_top = 0
        for grup_kayit in tabakalar.values():
            n = len(grup_kayit)
            for _ in range(n):          # her tabaka KENDI boyutunda ornekleniyor
                y, t = grup_kayit[rng.randrange(n)]
                y_top += y
                t_top += t
        if t_top:
            oranlar.append(y_top / t_top)

    oranlar.sort()
    alt_yuzde = (100 - yuzde) / 2
    alt = oranlar[max(0, int(len(oranlar) * alt_yuzde / 100) - 1)]
    ust = oranlar[min(len(oranlar) - 1, int(len(oranlar) * (100 - alt_yuzde) / 100))]
    return {
        "grup": grup or "tumu",
        "alan": alan,
        "oran": toplam_yakalanan / toplam_kutu,
        "alt": alt,
        "ust": ust,
        "genislik": ust - alt,
        "goruntu_sayisi": sum(len(v) for v in tabakalar.values()),
        "kutu_sayisi": toplam_kutu,
        "kaynak_sayisi": len(tabakalar),
        "tekrar": tekrar,
        "yontem": "tabakali_goruntu_bootstrap_yuzdelik",
    }


def yontem_karsilastir(
    kayitlar: Sequence[dict[str, Any]], grup: str, alan: str = "sinif", **kwargs: Any
) -> dict[str, Any]:
    """Ayni veri icin Wilson (kutu birimli) ve tabakali bootstrap araliklarini yan yana verir.

    Sartname, kutu birimli araliklarin yapay olarak DAR oldugunu soyler. Bu
    fonksiyon iddiayi olculebilir kilar: ``genislik_orani`` 1'den buyukse
    bootstrap araligi daha genistir, yani Wilson gercekten dar kalmistir.
    """
    boot = tabakali_goruntu_bootstrap(kayitlar, alan=alan, grup=grup, **kwargs)
    yakalanan = round(boot["oran"] * boot["kutu_sayisi"])
    w_alt, w_ust = wilson_araligi(yakalanan, boot["kutu_sayisi"])
    wilson_genislik = w_ust - w_alt
    return {
        "grup": grup,
        "oran": boot["oran"],
        "kutu_sayisi": boot["kutu_sayisi"],
        "goruntu_sayisi": boot["goruntu_sayisi"],
        "wilson": [w_alt, w_ust],
        "wilson_genislik": wilson_genislik,
        "bootstrap": [boot["alt"], boot["ust"]],
        "bootstrap_genislik": boot["genislik"],
        "genislik_orani": (boot["genislik"] / wilson_genislik) if wilson_genislik else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Guven araligi ve anlamlilik testi")
    parser.add_argument("--kosu", required=True, type=Path,
                        help="reports/kirilim/<run_id>.json veya reports/<x>_sonuc/d1_metrics.json")
    parser.add_argument("--referans", required=True, type=Path, help="saglikli referansin ayni turden dosyasi")
    parser.add_argument(
        "--alan", default="sinif_recall",
        help="kirilim: sinif_recall | boyut_bandi_recall | kaynak_recall | sinif_boyut_recall; "
             "d1_metrics: class_recall | class_precision",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Dosya turunu icerikten anla: d1_metrics.json "class_names" tasir.
    ilk = json.loads(args.kosu.read_text(encoding="utf-8"))
    if "class_names" in ilk:
        alan = args.alan if args.alan.startswith("class_") else "class_recall"
        satirlar = sinif_metrigi_karsilastir(args.kosu, args.referans, alan)
    else:
        satirlar = kirilim_karsilastir(args.kosu, args.referans, args.alan)
    baslik = f"{'grup':24s} {'n':>6s} {'recall':>8s} {'referans':>9s} {'fark':>9s} {'z':>7s} {'%95 GA':>18s}"
    print(baslik)
    print("-" * len(baslik))
    for s in satirlar:
        yildiz = "*" if s["anlamli_005"] else " "
        ga = f"[{s['wilson_alt']:.3f}, {s['wilson_ust']:.3f}]"
        print(
            f"{s['grup']:24s} {s['bbox_n']:6d} {s['recall']:8.4f} {s['referans_recall']:9.4f} "
            f"{s['fark']:+9.4f} {s['z']:+7.2f}{yildiz} {ga:>18s}"
        )
    print("\n* p < 0.05 (iki oran testi, kutu birimli)")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(satirlar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()

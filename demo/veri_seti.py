"""Veri seti ve saglikli referans modelin KUNYESI - hepsi kaynaktan turetilir.

Bu modulde hicbir sayi elle yazilmaz. Kaynaklar:

| Bilgi | Kaynak |
|---|---|
| Split/sinif/kaynak dagilimi | reports/veri_raporu.json |
| Kilitli tani seti | val_diagnostic/manifest.json |
| Sinif basina bbox (tani seti) | teshis/degerlendirme/bootstrap.py |
| Egitim ayarlari | experiments/<kosu>/args.yaml + run_manifest.json |
| Sonuc metrikleri | results.csv + reports/<kosu>/d1_metrics.json |
| Kirilimlar | reports/kirilim/<run_id>.json |

Bilinmeyen bir deger UYDURULMAZ: alan yoksa None doner ve arayuz "kayitta
yok" yazar. Sunumda "bu sayi nereden geliyor?" sorusunun tek cevabi
olmalidir.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

KOK = Path(__file__).resolve().parents[1]
VERI_RAPORU = KOK / "reports/veri_raporu.json"
VAL_MANIFEST = KOK / "val_diagnostic/manifest.json"
RESULTS_CSV = KOK / "results.csv"

# Sinif sozlesmesi DEGISMEZ; config.yaml ile ayni olmali.
SINIF_ADI = {0: "taşıt", 1: "insan", 2: "UAP", 3: "UAI"}
# Kod tarafindaki ASCII adlar (results/kirilim dosyalarinda boyle gecer).
KOD_ADI = {"tasit": "taşıt", "insan": "insan", "UAP": "UAP", "UAI": "UAI"}

BANT_ADI = {
    "cok_kucuk_16_alti": "çok küçük (<16 px)",
    "kucuk_16_32": "küçük (16-32 px)",
    "orta_32_64": "orta (32-64 px)",
    "buyuk_64_ustu": "büyük (>64 px)",
}


def _oku(yol: Path) -> dict:
    if not yol.is_file():
        return {}
    try:
        return json.loads(yol.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def veri_raporu() -> dict:
    return _oku(VERI_RAPORU)


def tani_seti() -> dict:
    """Kilitli degerlendirme setinin kunyesi."""
    return _oku(VAL_MANIFEST)


def split_dagilimi() -> list[dict[str, Any]]:
    rapor = veri_raporu()
    satirlar = []
    for ad, d in (rapor.get("splitler") or {}).items():
        satirlar.append({
            "bölüm": ad,
            "görüntü": d.get("goruntu"),
            "bbox": d.get("bbox"),
            "boş etiket": d.get("bos_etiket"),
            "görüntü başına nesne": (
                round(d["bbox"] / d["goruntu"], 2)
                if d.get("goruntu") else None
            ),
        })
    tani = tani_seti()
    if tani:
        satirlar.append({
            "bölüm": "val_diagnostic (kilitli)",
            "görüntü": tani.get("goruntu_sayisi"),
            "bbox": tani.get("bbox_sayisi"),
            "boş etiket": None,
            "görüntü başına nesne": (
                round(tani["bbox_sayisi"] / tani["goruntu_sayisi"], 2)
                if tani.get("goruntu_sayisi") else None
            ),
        })
    return satirlar


def sinif_dagilimi(kapsam: str = "toplam") -> list[dict[str, Any]]:
    """`kapsam`: 'toplam' | 'train' | 'val' | 'test' | 'tani'."""
    rapor = veri_raporu()
    if kapsam == "tani":
        ham = (tani_seti().get("sinif_bbox") or {})
    elif kapsam == "toplam":
        ham = (rapor.get("toplam") or {}).get("sinif_bbox") or {}
    else:
        ham = ((rapor.get("splitler") or {}).get(kapsam) or {}).get("sinif_bbox") or {}
    toplam = sum(ham.values()) or 1
    return [
        {"sınıf": KOD_ADI.get(k, k), "bbox": v, "pay": round(100 * v / toplam, 2)}
        for k, v in sorted(ham.items(), key=lambda c: -c[1])
    ]


def kaynak_dagilimi(kapsam: str = "toplam") -> list[dict[str, Any]]:
    rapor = veri_raporu()
    if kapsam == "tani":
        ham = tani_seti().get("kaynak_grubu") or {}
        return [
            {"kaynak": k, "görüntü": v.get("goruntu"), "bbox": v.get("bbox")}
            for k, v in sorted(ham.items(), key=lambda c: -(c[1].get("bbox") or 0))
        ]
    ham = rapor.get("kaynak_toplam") or {}
    return [
        {"kaynak": k, "görüntü": v.get("goruntu"), "bbox": v.get("bbox")}
        for k, v in sorted(ham.items(), key=lambda c: -(c[1].get("bbox") or 0))
    ]


def saglik_uyarilari() -> list[dict[str, Any]]:
    """Veri saglik taramasinin sifirdan farkli bulgulari.

    Sifir olanlari gostermek listeyi sisirir; sifirdan farkli olanlar ise
    sunumda "veriyi denetledik mi?" sorusunun somut cevabidir.
    """
    ACIKLAMA = {
        "gecersiz_satir": "Biçimi bozuk etiket satırı",
        "gecersiz_sinif": "Tanımsız sınıf kimliği",
        "eksik_kolon": "Eksik sütunlu etiket satırı",
        "fazla_kolon": "Fazla sütunlu etiket satırı",
        "koordinat_aralik_disi": "Koordinat [0,1] aralığı dışında",
        "kutu_goruntu_disi": "Kutu görüntü sınırının dışına taşıyor",
        "sifir_alan": "Sıfır alanlı kutu",
        "tekrar_kutu": "Aynı görüntüde birebir tekrar eden kutu",
        "cok_kucuk_kutu_sqrt_alan_6_alt": "Çok küçük kutu (√alan < 6 px)",
        "asiri_en_boy": "Aşırı en-boy oranı",
        "etiketsiz_goruntu": "Etiket dosyası olmayan görüntü",
        "goruntusuz_etiket": "Görüntüsü olmayan etiket",
        "okunamayan_goruntu": "Okunamayan görüntü",
    }
    saglik = veri_raporu().get("saglik") or {}
    return [
        {"bulgu": ACIKLAMA.get(k, k), "adet": v, "anahtar": k}
        for k, v in sorted(saglik.items(), key=lambda c: -c[1]) if v
    ]


# --- Saglikli referans modelin kunyesi --------------------------------------

def _defter() -> dict[str, dict[str, str]]:
    with RESULTS_CSV.open(encoding="utf-8") as f:
        return {s["scenario"]: s for s in csv.DictReader(f)}


def kosu_dizini(senaryo: str) -> Path | None:
    satir = _defter().get(senaryo)
    if not satir:
        return None
    dizin = KOK / Path(satir["weights_path"]).parent.parent
    return dizin if dizin.is_dir() else None


def egitim_ayarlari(senaryo: str = "v00_saglikli") -> dict[str, Any]:
    """Kosunun GERCEK egitim ayarlari: args.yaml + manifest + defter.

    Ultralytics'in yazdigi `args.yaml` esas kaynaktir - protokolde ne yazdigi
    degil, egitimin ne ile kostugu onemlidir. `optimizer: auto` ornegi tam
    olarak bu yuzden kritik: protokol lr0 beyan etse bile auto onu yok sayar.
    """
    satir = _defter().get(senaryo, {})
    dizin = kosu_dizini(senaryo)
    args, manifest = {}, {}
    if dizin:
        yol = dizin / "args.yaml"
        if yol.is_file():
            args = yaml.safe_load(yol.read_text(encoding="utf-8")) or {}
        manifest = _oku(dizin / "run_manifest.json")

    def kisa(deger):
        """Mutlak yollari yalnizca dosya adina indirger."""
        if isinstance(deger, str) and ("\\" in deger or "/" in deger):
            return Path(deger).name
        return deger

    return {
        "koşu kimliği": satir.get("run_id"),
        "başlangıç ağırlığı": kisa(args.get("model") or satir.get("model")),
        "veri sürümü": satir.get("data_version") or kisa(args.get("data")),
        "değerlendirme kümesi": satir.get("evaluation_set"),
        "eğitim çözünürlüğü": args.get("imgsz") or satir.get("imgsz_train"),
        "çıkarım çözünürlüğü": satir.get("imgsz_eval"),
        "optimizer": args.get("optimizer"),
        "öğrenme oranı (lr0)": args.get("lr0", satir.get("lr0")),
        "momentum": args.get("momentum"),
        "warmup epoch": args.get("warmup_epochs"),
        "batch": args.get("batch") or satir.get("batch"),
        "planlanan epoch": manifest.get("epochs") or args.get("epochs"),
        "erken durdurma sabrı": args.get("patience"),
        "durduğu epoch": satir.get("epochs"),
        "seed": args.get("seed") or satir.get("seed"),
        "checkpoint": ("last.pt" if str(satir.get("weights_path", "")).endswith(
            "last.pt") else "best.pt"),
        "cos_lr": args.get("cos_lr"),
        "close_mosaic": args.get("close_mosaic"),
        "eğitim süresi (dk)": satir.get("duration_min"),
    }


def optimizer_notu(senaryo: str = "v00_saglikli") -> str | None:
    """`optimizer: auto` iken beyan edilen lr0 BAGLAYICI DEGILDIR.

    Bu, E3'ü sessizce gecersiz kilabilecek bir tuzakti: protokol lr0'i 100
    kat yukselttigini beyan ediyor, Ultralytics auto modda onu yok sayiyor.
    Kunyede gorunmesi gerekir.
    """
    ayar = egitim_ayarlari(senaryo)
    if str(ayar.get("optimizer")).lower() == "auto":
        return (
            "Bu koşu `optimizer: auto` ile eğitildi. Ultralytics bu modda "
            "öğrenme oranını ve momentumu kendisi seçer; beyan edilen `lr0` "
            "**bağlayıcı değildir**. E serisinde öğrenme oranını değiştiren "
            "senaryolar bu yüzden optimizer'ı da açıkça yazar."
        )
    return None


def metrikler(senaryo: str) -> dict[str, Any]:
    """Kosunun olcum dosyasi: sinif bazli AP dahil."""
    from teshis.degerlendirme.raporlar import rapor_klasoru

    try:
        klasor = rapor_klasoru(senaryo)
    except Exception:  # noqa: BLE001 - raporu olmayan kosu akisi durdurmasin
        return {}
    if klasor is None:
        return {}
    return _oku(KOK / "reports" / klasor / "d1_metrics.json")


def sinif_metrikleri(senaryo: str) -> list[dict[str, Any]]:
    """Sinif basina AP50 / AP50-95 / precision / recall + tani setindeki n."""
    from teshis.degerlendirme.bootstrap import VAL_DIAGNOSTIC_BBOX_N

    m = metrikler(senaryo)
    adlar = m.get("class_names") or []
    if not adlar:
        return []
    satirlar = []
    for i, ad in enumerate(adlar):
        def al(alan):
            dizi = m.get(alan) or []
            return round(dizi[i], 4) if i < len(dizi) else None
        n = VAL_DIAGNOSTIC_BBOX_N.get(ad)
        satirlar.append({
            "sınıf": KOD_ADI.get(ad, ad),
            "bbox (tanı seti)": n,
            "AP50": al("class_ap50"),
            "AP50-95": al("class_ap50_95"),
            "precision": al("class_precision"),
            "recall": al("class_recall"),
            "az örnek": "evet" if (n is not None and n < 30) else "hayır",
        })
    return satirlar


def kirilim(senaryo: str) -> dict[str, Any]:
    """Kosunun boyut/kaynak/sinif kirilim olcumu."""
    satir = _defter().get(senaryo, {})
    run_id = satir.get("run_id")
    if not run_id:
        return {}
    return _oku(KOK / f"reports/kirilim/{run_id}.json")


def egitim_egrisi(senaryo: str):
    """Ultralytics'in epoch bazli results.csv'si."""
    import pandas as pd

    dizin = kosu_dizini(senaryo)
    yol = (dizin / "results.csv") if dizin else None
    if not yol or not yol.is_file():
        return None
    df = pd.read_csv(yol)
    df.columns = [c.strip() for c in df.columns]
    return df

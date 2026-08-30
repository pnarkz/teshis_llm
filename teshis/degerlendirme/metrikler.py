"""Sinif, nesne boyutu ve kaynak grubu bazli metrik hesaplari.

Toplam mAP, "model kucuk nesneleri kacirmaya basladi" gibi bir iddiayi
gostermez: kucuk nesneler bbox sayisinin kucuk bir kismini olusturdugu icin
toplam metrikte kaybolurlar. Bu modul, kilitli tanı val setindeki her gercek
kutuyu etkin piksel boyutuna gore bantlara ayirir ve her bant icin ayri
recall hesaplar.

Etkin boyut, goruntu cozunurlugune gore normalize edilir: dataset'te hem
1024x1024 hem 640x512 kareler bulundugu icin ham piksel alani karsilastirilabilir
degildir. teshis/veri/istatistik.py ile ayni formul kullanilir:

    etkin_sqrt_alan = sqrt(w_norm * W * h_norm * H) * (referans / max(W, H))

Bant sinirlari D4 senaryosunun esigiyle (16 px) hizalidir; boylece "egitimden
cikarilan boyut bandi" ile "recall'i olculen bant" birebir ortusur.

Kaynak grubu (aaterm, hituav, termal, sentetik, tf2026) dosya adindan
`teshis/veri/istatistik.py::kaynak_adi` ile turetilir; ayni fonksiyon veri
raporunda da kullanildigi icin gruplama iki yerde ayrisamaz. D5 (kaynak/alan
kaymasi) senaryosunun hipotezi bu kirilimla test edilir.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from teshis.veri.istatistik import kaynak_adi

SINIFLAR = {0: "tasit", 1: "insan", 2: "UAP", 3: "UAI"}
GORUNTU_UZANTILARI = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# (ust_sinir_px, bant_adi); ust sinir haric tutulur. 16, D4 esigidir.
BANTLAR: tuple[tuple[float, str], ...] = (
    (16.0, "cok_kucuk_16_alti"),
    (32.0, "kucuk_16_32"),
    (64.0, "orta_32_64"),
    (float("inf"), "buyuk_64_ustu"),
)


def bant_araliklari() -> dict[str, str]:
    """Bant sinirlarini JSON-guvenli, okunabilir araliklar olarak dondurur.

    BANTLAR'in son siniri float("inf")'tir; bu deger JSON'a ``Infinity`` diye
    yazilir ve RFC 8259'a gore GECERSIZDIR. Gemini API'si boyle bir govdeyi
    400 INVALID_ARGUMENT ile reddeder. Bu fonksiyon sayisal siniri metin
    araligina cevirir; hem gecerli JSON uretir hem de modele daha anlasilir
    gelir ("16-32" gibi).
    """
    araliklar: dict[str, str] = {}
    alt = 0.0
    for ust, ad in BANTLAR:
        araliklar[ad] = f"{alt:g}-{ust:g} px" if ust != float("inf") else f"{alt:g}+ px"
        alt = ust
    return araliklar


def etkin_sqrt_alan(
    w_norm: float, h_norm: float, goruntu_w: int, goruntu_h: int, referans: int = 640
) -> float:
    """YOLO normalize kutu boyutunu, referans uzun kenara olceklenmis piksel boyutuna cevirir."""
    if goruntu_w <= 0 or goruntu_h <= 0:
        return 0.0
    olcek = referans / max(goruntu_w, goruntu_h)
    return (max(w_norm, 0.0) * goruntu_w * max(h_norm, 0.0) * goruntu_h) ** 0.5 * olcek


def boyut_bandi(sqrt_alan_px: float) -> str:
    """Etkin piksel boyutunu bant adina esler."""
    for ust_sinir, ad in BANTLAR:
        if sqrt_alan_px < ust_sinir:
            return ad
    return BANTLAR[-1][1]


def iou(kutu_a: tuple[float, float, float, float], kutu_b: tuple[float, float, float, float]) -> float:
    """Iki xyxy kutusunun kesisim/birlesim oranini dondurur."""
    ax1, ay1, ax2, ay2 = kutu_a
    bx1, by1, bx2, by2 = kutu_b
    kesisim_w = min(ax2, bx2) - max(ax1, bx1)
    kesisim_h = min(ay2, by2) - max(ay1, by1)
    if kesisim_w <= 0 or kesisim_h <= 0:
        return 0.0
    kesisim = kesisim_w * kesisim_h
    alan_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    alan_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    birlesim = alan_a + alan_b - kesisim
    return kesisim / birlesim if birlesim > 0 else 0.0


def yolo_to_xyxy(
    x: float, y: float, w: float, h: float, goruntu_w: int, goruntu_h: int
) -> tuple[float, float, float, float]:
    """Normalize YOLO kutusunu piksel xyxy'ye cevirir."""
    cx, cy = x * goruntu_w, y * goruntu_h
    yari_w, yari_h = w * goruntu_w / 2, h * goruntu_h / 2
    return cx - yari_w, cy - yari_h, cx + yari_w, cy + yari_h


def eslestir(
    gercek: list[dict[str, Any]],
    tahminler: list[dict[str, Any]],
    iou_esigi: float = 0.5,
) -> set[int]:
    """Tahminleri gercek kutulara acgozlu esler; eslesen gercek kutu indislerini dondurur.

    Tahminler guven skoruna gore azalan sirada islenir. Her tahmin, ayni sinifta
    ve IoU esigini gecen, henuz eslesmemis en yuksek IoU'lu gercek kutuyu tutar.
    Bu, Ultralytics'in recall hesabiyla ayni mantiktir; burada amac ayni sayiyi
    yeniden uretmek degil, bant bazinda ayristirabilmektir.
    """
    eslesen: set[int] = set()
    for tahmin in sorted(tahminler, key=lambda t: -t["conf"]):
        en_iyi_indis, en_iyi_iou = -1, iou_esigi
        for indis, hedef in enumerate(gercek):
            if indis in eslesen or hedef["sinif"] != tahmin["sinif"]:
                continue
            deger = iou(hedef["kutu"], tahmin["kutu"])
            if deger >= en_iyi_iou:
                en_iyi_indis, en_iyi_iou = indis, deger
        if en_iyi_indis >= 0:
            eslesen.add(en_iyi_indis)
    return eslesen


def sinif_bagimsiz_eslestir(
    gercek: list[dict[str, Any]],
    tahminler: list[dict[str, Any]],
    iou_esigi: float = 0.5,
) -> list[tuple[int, int]]:
    """Kutulari SINIFTAN BAGIMSIZ eslestirir; (gercek_indis, tahmin_indis) ciftleri dondurur.

    Karisiklik matrisi icin gereklidir: "dogru yerde bulundu ama yanlis sinif
    verildi" durumunu yakalayabilmek icin eslestirme yalnizca konuma (IoU)
    bakmalidir. eslestir() ise ayni sinif sartini arar ve bu durumu goremez.

    D3b tam olarak bu ayrimla ortaya cikti: sinif bazli recall/AP neredeyse
    degismemisken, gercek tasit kutularinin %28'i insan olarak tahmin
    ediliyordu.
    """
    ciftler: list[tuple[int, int]] = []
    kullanilan: set[int] = set()
    for tahmin_indis, _ in sorted(
        enumerate(tahminler), key=lambda oge: -oge[1]["conf"]
    ):
        en_iyi_indis, en_iyi_iou = -1, iou_esigi
        for indis, hedef in enumerate(gercek):
            if indis in kullanilan:
                continue
            deger = iou(hedef["kutu"], tahminler[tahmin_indis]["kutu"])
            if deger >= en_iyi_iou:
                en_iyi_indis, en_iyi_iou = indis, deger
        if en_iyi_indis >= 0:
            kullanilan.add(en_iyi_indis)
            ciftler.append((en_iyi_indis, tahmin_indis))
    return ciftler


def etiketleri_oku(etiket_yolu: Path, goruntu_w: int, goruntu_h: int) -> list[dict[str, Any]]:
    """Bir YOLO etiket dosyasini piksel kutulari + boyut bandi olarak okur."""
    if not etiket_yolu.is_file():
        return []
    kutular: list[dict[str, Any]] = []
    for satir in etiket_yolu.read_text(encoding="utf-8").splitlines():
        alanlar = satir.split()
        if len(alanlar) < 5:
            continue
        try:
            sinif = int(float(alanlar[0]))
            x, y, w, h = (float(deger) for deger in alanlar[1:5])
        except ValueError:
            continue
        boyut = etkin_sqrt_alan(w, h, goruntu_w, goruntu_h)
        kutular.append({
            "sinif": sinif,
            "kutu": yolo_to_xyxy(x, y, w, h, goruntu_w, goruntu_h),
            "sqrt_alan_px": boyut,
            "bant": boyut_bandi(boyut),
        })
    return kutular


def _ozetle(sayaclar: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {
        ad: {
            "gercek_kutu": deger["toplam"],
            "yakalanan": deger["eslesen"],
            "recall": round(deger["eslesen"] / deger["toplam"], 4) if deger["toplam"] else None,
        }
        for ad, deger in sayaclar.items()
    }


def boyut_bazli_recall(
    model_yolu: Path,
    val_root: Path,
    imgsz: int = 768,
    conf: float = 0.25,
    iou_esigi: float = 0.5,
    batch: int = 16,
) -> dict[str, Any]:
    """Tanı val setinde boyut bandi ve sinif bazli recall hesaplar."""
    try:
        from PIL import Image
        from ultralytics import YOLO
    except ImportError as hata:
        raise RuntimeError("Bu komut ultralytics ve Pillow kurulu bir ortamda calisir.") from hata

    images_dir = val_root / "images"
    labels_dir = val_root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"val_diagnostic yapisi bulunamadi: {val_root}")

    goruntuler = sorted(
        yol for yol in images_dir.iterdir() if yol.suffix.lower() in GORUNTU_UZANTILARI
    )
    if not goruntuler:
        raise FileNotFoundError(f"Goruntu bulunamadi: {images_dir}")

    model = YOLO(str(model_yolu))
    bant_sayaclari: dict[str, dict[str, int]] = defaultdict(lambda: {"toplam": 0, "eslesen": 0})
    sinif_sayaclari: dict[str, dict[str, int]] = defaultdict(lambda: {"toplam": 0, "eslesen": 0})
    sinif_bant: dict[str, dict[str, int]] = defaultdict(lambda: {"toplam": 0, "eslesen": 0})
    kaynak_sayaclari: dict[str, dict[str, int]] = defaultdict(lambda: {"toplam": 0, "eslesen": 0})
    kaynak_sinif: dict[str, dict[str, int]] = defaultdict(lambda: {"toplam": 0, "eslesen": 0})
    goruntu_kayitlari: list[dict[str, Any]] = []
    # karisiklik["gercek_sinif"]["tahmin_sinifi" | "bulunamadi"] = adet
    karisiklik: dict[str, Counter[str]] = defaultdict(Counter)
    toplam_tahmin = 0

    for baslangic in range(0, len(goruntuler), batch):
        parca = goruntuler[baslangic : baslangic + batch]
        sonuclar = model.predict(
            [str(yol) for yol in parca], imgsz=imgsz, conf=conf, verbose=False, device=0
        )
        for goruntu_yolu, sonuc in zip(parca, sonuclar):
            # Goruntu birimli kayit: sartnamedeki C3 bootstrap protokolu
            # yeniden ornekleme birimi olarak GORUNTUYU ister. Toplam sayimlar
            # bunu yapmaya yetmez - ayni karedeki kutularin birlikte secilmesi
            # gerekir - bu yuzden kare basina kirilim burada saklanir.
            with Image.open(goruntu_yolu) as goruntu:
                genislik, yukseklik = goruntu.size
            gercek = etiketleri_oku(labels_dir / f"{goruntu_yolu.stem}.txt", genislik, yukseklik)
            tahminler = [
                {
                    "sinif": int(sinif),
                    "conf": float(guven),
                    "kutu": (float(k[0]), float(k[1]), float(k[2]), float(k[3])),
                }
                for k, sinif, guven in zip(
                    sonuc.boxes.xyxy.tolist(), sonuc.boxes.cls.tolist(), sonuc.boxes.conf.tolist()
                )
            ]
            toplam_tahmin += len(tahminler)
            kaynak = kaynak_adi(goruntu_yolu.name)
            eslesen = eslestir(gercek, tahminler, iou_esigi)
            kare_sinif: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            kare_bant: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for indis, hedef in enumerate(gercek):
                sinif_adi = SINIFLAR.get(hedef["sinif"], str(hedef["sinif"]))
                vurus = 1 if indis in eslesen else 0
                kare_sinif[sinif_adi][0] += vurus
                kare_sinif[sinif_adi][1] += 1
                kare_bant[hedef["bant"]][0] += vurus
                kare_bant[hedef["bant"]][1] += 1
                for sayac, anahtar in (
                    (bant_sayaclari, hedef["bant"]),
                    (sinif_sayaclari, sinif_adi),
                    (sinif_bant, f"{sinif_adi}|{hedef['bant']}"),
                    (kaynak_sayaclari, kaynak),
                    (kaynak_sinif, f"{kaynak}|{sinif_adi}"),
                ):
                    sayac[anahtar]["toplam"] += 1
                    sayac[anahtar]["eslesen"] += vurus

            goruntu_kayitlari.append({
                "goruntu": goruntu_yolu.name,
                "kaynak": kaynak,
                "sinif": {ad: list(v) for ad, v in sorted(kare_sinif.items())},
                "bant": {ad: list(v) for ad, v in sorted(kare_bant.items())},
            })

            # Karisiklik matrisi: sinif bagimsiz eslestirme ile "dogru yerde
            # bulundu ama yanlis sinif" durumunu yakala.
            eslesme_haritasi = dict(sinif_bagimsiz_eslestir(gercek, tahminler, iou_esigi))
            for indis, hedef in enumerate(gercek):
                gercek_ad = SINIFLAR.get(hedef["sinif"], str(hedef["sinif"]))
                tahmin_indis = eslesme_haritasi.get(indis)
                if tahmin_indis is None:
                    karisiklik[gercek_ad]["bulunamadi"] += 1
                else:
                    tahmin_sinif = tahminler[tahmin_indis]["sinif"]
                    karisiklik[gercek_ad][SINIFLAR.get(tahmin_sinif, str(tahmin_sinif))] += 1

    bant_sirasi = [ad for _, ad in BANTLAR]
    return {
        "model": str(model_yolu.resolve()),
        "evaluation_set": str(val_root.resolve()),
        "imgsz": imgsz,
        "conf": conf,
        "iou_esigi": iou_esigi,
        "goruntu": len(goruntuler),
        # Goruntu birimli bootstrap icin ham kayitlar (bkz. bootstrap.py
        # tabakali_goruntu_bootstrap). Toplam sayimlardan turetilemez.
        "goruntu_kayitlari": goruntu_kayitlari,
        "toplam_tahmin": toplam_tahmin,
        "bant_tanimi": bant_araliklari(),
        "boyut_bandi_recall": {
            ad: _ozetle(bant_sayaclari)[ad] for ad in bant_sirasi if ad in bant_sayaclari
        },
        "sinif_recall": _ozetle(sinif_sayaclari),
        "sinif_boyut_recall": _ozetle(sinif_bant),
        "kaynak_recall": _ozetle(kaynak_sayaclari),
        "kaynak_sinif_recall": _ozetle(kaynak_sinif),
        "karisiklik_matrisi": {
            gercek: dict(sayimlar) for gercek, sayimlar in sorted(karisiklik.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Boyut bandi ve sinif bazli recall")
    parser.add_argument("--model", required=True)
    parser.add_argument("--val-root", default="val_diagnostic")
    parser.add_argument("--output", default=None, help="JSON cikti yolu")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    sonuc = boyut_bazli_recall(
        Path(args.model), Path(args.val_root), args.imgsz, args.conf, args.iou
    )
    metin = json.dumps(sonuc, ensure_ascii=False, indent=2)
    if args.output:
        cikti = Path(args.output)
        cikti.parent.mkdir(parents=True, exist_ok=True)
        cikti.write_text(metin + "\n", encoding="utf-8")
        print(f"saved={cikti.resolve()}")
    print(metin)


if __name__ == "__main__":
    main()

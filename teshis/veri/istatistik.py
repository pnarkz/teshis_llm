"""YOLO veri seti istatistikleri ve etiket sagligi raporu."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

from PIL import Image

from teshis.config import aktif_ortam, yukle

SPLITLER = ("train", "val", "test")
SINIFLAR = {0: "tasit", 1: "insan", 2: "UAP", 3: "UAI"}
GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RF_EKI = re.compile(r"_(?:jpg|jpeg|png|webp)\.rf\.[0-9a-f]{16,}$", re.IGNORECASE)


def kaynak_adi(dosya_adi: str) -> str:
    """Dosya adindaki veri kaynagi onekini standart bir ada cevirir."""
    kucuk = dosya_adi.lower()
    if kucuk.startswith("aaterm__"):
        return "aaterm"
    if kucuk.startswith("hituav__"):
        return "hituav"
    if kucuk.startswith("termal__") or kucuk.startswith("thermal_"):
        return "termal"
    if kucuk.startswith("synth_"):
        return "sentetik"
    if kucuk.startswith("frame_"):
        return "tf2026"
    if kucuk.startswith("rf"):
        return "tf2026"
    return "bilinmeyen"


def kaynak_govde(dosya_adi: str) -> str:
    """Roboflow hash'ini kaldirarak yaklasik kaynak adini dondurur."""
    return RF_EKI.sub("", Path(dosya_adi).stem)


def _yeni_saglik() -> dict[str, Any]:
    return {
        "gecersiz_satir": 0,
        "gecersiz_sinif": 0,
        "eksik_kolon": 0,
        "fazla_kolon": 0,
        "koordinat_aralik_disi": 0,
        "kutu_goruntu_disi": 0,
        "sifir_alan": 0,
        "tekrar_kutu": 0,
        "cok_kucuk_kutu_sqrt_alan_6_alt": 0,
        "asiri_en_boy": 0,
        "etiketsiz_goruntu": 0,
        "goruntusuz_etiket": 0,
        "okunamayan_goruntu": 0,
    }


def _goruntu_bilgisi(yol: Path) -> tuple[int, int] | None:
    try:
        with Image.open(yol) as goruntu:
            return goruntu.size
    except (OSError, ValueError):
        return None


def _etiketleri_oku(
    yol: Path,
    goruntu_boyutu: tuple[int, int] | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kutular: list[dict[str, Any]] = []
    sorunlar = _yeni_saglik()
    try:
        satirlar = yol.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        sorunlar["gecersiz_satir"] += 1
        return kutular, sorunlar

    for satir in satirlar:
        if not satir.strip():
            continue
        kolonlar = satir.split()
        if len(kolonlar) < 5:
            sorunlar["eksik_kolon"] += 1
            continue
        if len(kolonlar) > 5:
            sorunlar["fazla_kolon"] += 1
        try:
            sinif_id = int(float(kolonlar[0]))
            x, y, w, h = (float(deger) for deger in kolonlar[1:5])
        except ValueError:
            sorunlar["gecersiz_satir"] += 1
            continue

        if sinif_id not in SINIFLAR:
            sorunlar["gecersiz_sinif"] += 1
        if not all(0.0 <= deger <= 1.0 for deger in (x, y, w, h)):
            sorunlar["koordinat_aralik_disi"] += 1
        if x - w / 2 < 0 or x + w / 2 > 1 or y - h / 2 < 0 or y + h / 2 > 1:
            sorunlar["kutu_goruntu_disi"] += 1
        if w <= 0 or h <= 0:
            sorunlar["sifir_alan"] += 1
        if goruntu_boyutu and goruntu_boyutu[0] and goruntu_boyutu[1]:
            genislik, yukseklik = goruntu_boyutu
            olcek = 640 / max(genislik, yukseklik)
            sqrt_alan = (max(w, 0.0) * genislik * max(h, 0.0) * yukseklik) ** 0.5 * olcek
        else:
            sqrt_alan = (max(w, 0.0) * max(h, 0.0)) ** 0.5 * 640
        if sqrt_alan < 6:
            sorunlar["cok_kucuk_kutu_sqrt_alan_6_alt"] += 1
        if h > 0 and (w / h > 10 or w / h < 0.1):
            sorunlar["asiri_en_boy"] += 1
        kutular.append({
            "sinif_id": sinif_id,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })

    gorulen: Counter[tuple[Any, ...]] = Counter(
        (k["sinif_id"], k["x"], k["y"], k["w"], k["h"]) for k in kutular
    )
    sorunlar["tekrar_kutu"] = sum(adet - 1 for adet in gorulen.values() if adet > 1)
    return kutular, sorunlar


def analiz_et(dataset_root: Path) -> dict[str, Any]:
    """Dataset'i tarar ve JSON'a yazilabilir istatistik sozlugu uretir."""
    sinif_sayilari = {isim: 0 for isim in SINIFLAR.values()}
    toplam = {
        "goruntu": 0,
        "etiket": 0,
        "bbox": 0,
        "bos_etiket": 0,
    }
    rapor: dict[str, Any] = {
        "dataset_root": str(dataset_root.resolve()),
        "siniflar": {str(k): v for k, v in SINIFLAR.items()},
        "splitler": {},
        "toplam": toplam,
        "kaynak_toplam": defaultdict(lambda: {"goruntu": 0, "bbox": 0}),
        "saglik": _yeni_saglik(),
    }

    for split in SPLITLER:
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        goruntuler = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in GORUNTU_UZANTILARI}
        etiketler = {p.stem: p for p in labels_dir.glob("*.txt")}
        goruntu_orphan = sorted(set(etiketler) - set(goruntuler))
        etiket_orphan = sorted(set(goruntuler) - set(etiketler))

        split_raporu: dict[str, Any] = {
            "goruntu": len(goruntuler),
            "etiket": len(etiketler),
            "bbox": 0,
            "bos_etiket": 0,
            "cozunurluk": defaultdict(int),
            "sinif_bbox": {isim: 0 for isim in SINIFLAR.values()},
            "kaynak": defaultdict(lambda: {"goruntu": 0, "bbox": 0}),
            "yetim_goruntu_sayisi": len(etiket_orphan),
            "yetim_etiket_sayisi": len(goruntu_orphan),
            "yetim_goruntu_ornekleri": etiket_orphan[:20],
            "yetim_etiket_ornekleri": goruntu_orphan[:20],
        }
        rapor["saglik"]["etiketsiz_goruntu"] += len(etiket_orphan)
        rapor["saglik"]["goruntusuz_etiket"] += len(goruntu_orphan)

        for stem, goruntu_yolu in goruntuler.items():
            bilgi = _goruntu_bilgisi(goruntu_yolu)
            if bilgi is None:
                rapor["saglik"]["okunamayan_goruntu"] += 1
            else:
                split_raporu["cozunurluk"][f"{bilgi[0]}x{bilgi[1]}"] += 1
            etiket_yolu = etiketler.get(stem)
            if etiket_yolu is None:
                continue
            kutular, sorunlar = _etiketleri_oku(etiket_yolu, bilgi)
            for alan, miktar in sorunlar.items():
                rapor["saglik"][alan] += miktar
            toplam["etiket"] += 1
            toplam["goruntu"] += 1
            kaynak = kaynak_adi(goruntu_yolu.name)
            split_raporu["kaynak"][kaynak]["goruntu"] += 1
            rapor["kaynak_toplam"][kaynak]["goruntu"] += 1
            if not kutular:
                split_raporu["bos_etiket"] += 1
                toplam["bos_etiket"] += 1
            split_raporu["bbox"] += len(kutular)
            toplam["bbox"] += len(kutular)
            split_raporu["kaynak"][kaynak]["bbox"] += len(kutular)
            rapor["kaynak_toplam"][kaynak]["bbox"] += len(kutular)
            for kutu in kutular:
                sinif_adi = SINIFLAR.get(kutu["sinif_id"])
                if sinif_adi is not None:
                    split_raporu["sinif_bbox"][sinif_adi] += 1
                    sinif_sayilari[sinif_adi] += 1

        rapor["splitler"][split] = split_raporu

    rapor["toplam"]["sinif_bbox"] = sinif_sayilari
    rapor["kaynak_toplam"] = dict(rapor["kaynak_toplam"])
    for split_raporu in rapor["splitler"].values():
        split_raporu["kaynak"] = dict(split_raporu["kaynak"])
        split_raporu["cozunurluk"] = dict(split_raporu["cozunurluk"])
    return rapor

def raporla(dataset_root: Path, output_dir: Path) -> None:
    """Dataset'i tarar, raporu yazdirir ve JSON dosyasi uretir."""
    beklenen = [dataset_root / tur / split for tur in ("images", "labels") for split in SPLITLER]
    eksik = [str(yol) for yol in beklenen if not yol.is_dir()]
    if eksik:
        raise FileNotFoundError("Eksik dataset klasorleri:\n" + "\n".join(eksik))
    rapor = analiz_et(dataset_root)
    rapor_yolu = output_dir / "veri_raporu.json"
    rapor_yolu.parent.mkdir(parents=True, exist_ok=True)
    rapor_yolu.write_text(json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dataset_root: {dataset_root}")
    print(f"toplam goruntu: {rapor['toplam']['goruntu']}")
    print(f"toplam bbox: {rapor['toplam']['bbox']}")
    print(f"sinif bbox: {rapor['toplam']['sinif_bbox']}")
    print(f"etiketsiz goruntu: {rapor['saglik']['etiketsiz_goruntu']}")
    print(f"goruntusuz etiket: {rapor['saglik']['goruntusuz_etiket']}")
    print(f"rapor: {rapor_yolu.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO veri seti saglik kontrolu")
    parser.add_argument("--config", default="config.yaml", help="YAML config yolu")
    args = parser.parse_args()
    config = yukle(args.config)
    ortam = aktif_ortam(config)
    rapor_dizini = Path(ortam["output_root"]) / config.get("rapor", {}).get("output_dir", "reports")
    raporla(Path(ortam["dataset_root"]), rapor_dizini)


if __name__ == "__main__":
    main()

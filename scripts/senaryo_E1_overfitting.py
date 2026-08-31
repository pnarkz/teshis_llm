"""E1: overfitting icin kucuk bir egitim alt kumesi uretir.

E serisi VERIYI bozmaz; bozulan egitim protokoludur. E1'de iki sey birlikte
degisir ve ikisi de protokolde BEYAN EDILMISTIR
(senaryolar/egitim_protokolu.yaml -> e_serisi.E1):

- augmentasyon tamamen kapatilir (mosaic, fliplr, scale, hsv... hepsi 0),
- egitim yalnizca 1000 kare uzerinde 200 epoch surer.

Etiketler HIC degistirilmez ve goruntu kopyalanmaz; D1/D5/D6b gibi
**manifest-only** calisir - yalnizca bir train goruntu listesi yazilir.

Ornekleme
---------
1000 kare, kaynak grubuna gore TABAKALI secilir. Rastgele secim, kaynak
paylarini kaydirarak E1'e istenmeyen ikinci bir degisken (kaynak dagilimi
kaymasi - ki bu D5'in konusu) katardi. Tabakali secim, alt kumenin kaynak
dagilimini tam kumeninkiyle ayni tutar; boylece tek degisken "az veri +
augmentasyon yok" olarak kalir.

Kullanim:
    python scripts/senaryo_E1_overfitting.py --dataset C:/Users/ASUS/Desktop/HYZ/dataset
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teshis.egitim.protokol import e_senaryo_ayarlari  # noqa: E402
from teshis.veri.istatistik import kaynak_adi  # noqa: E402

UZANTILAR = (".jpg", ".jpeg", ".png", ".bmp")


def find_image(directory: Path, stem: str) -> Path | None:
    for uzanti in UZANTILAR:
        aday = directory / f"{stem}{uzanti}"
        if aday.is_file():
            return aday
    return None


def tabakali_sec(
    kareler: list[tuple[Path, Path]], hedef: int, seed: int
) -> list[tuple[Path, Path]]:
    """Kaynak grubu paylarini koruyarak `hedef` kadar kare secer.

    En buyuk artik (largest remainder) yontemi kullanilir: her gruba once
    payinin tam kismi verilir, kalan kontenjan en buyuk ondalik artiga sahip
    gruplara dagitilir. Boylece toplam tam olarak `hedef` olur ve kucuk
    gruplar tamamen silinmez.
    """
    gruplar: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for etiket, goruntu in kareler:
        gruplar[kaynak_adi(goruntu.name)].append((etiket, goruntu))

    toplam = len(kareler)
    paylar = {ad: len(v) * hedef / toplam for ad, v in gruplar.items()}
    kota = {ad: int(p) for ad, p in paylar.items()}
    kalan = hedef - sum(kota.values())
    for ad, _ in sorted(paylar.items(), key=lambda kv: -(kv[1] - int(kv[1]))):
        if kalan <= 0:
            break
        if kota[ad] < len(gruplar[ad]):
            kota[ad] += 1
            kalan -= 1

    rng = random.Random(seed)
    secilen: list[tuple[Path, Path]] = []
    for ad in sorted(gruplar):
        havuz = sorted(gruplar[ad], key=lambda p: p[1].name)
        rng.shuffle(havuz)
        secilen.extend(havuz[: min(kota[ad], len(havuz))])
    secilen.sort(key=lambda p: p[1].name)
    return secilen


def _siniflar(etiket_yolu: Path) -> Counter:
    sayac: Counter = Counter()
    if not etiket_yolu.is_file():
        return sayac
    for satir in etiket_yolu.read_text(encoding="utf-8").splitlines():
        parca = satir.split()
        if parca:
            sayac[int(parca[0])] += 1
    return sayac


def build_dataset(dataset_root: Path, output_root: Path, seed: int) -> Path:
    ayar = e_senaryo_ayarlari("E1")["kosu_ayarlari"]
    hedef = int(ayar["kare_sayisi"])

    labels_dir = dataset_root / "labels/train"
    images_dir = dataset_root / "images/train"
    if not labels_dir.is_dir() or not images_dir.is_dir():
        raise FileNotFoundError(f"Kaynak dataset eksik: {dataset_root}")

    kareler = []
    for etiket in sorted(labels_dir.glob("*.txt")):
        goruntu = find_image(images_dir, etiket.stem)
        if goruntu:
            kareler.append((etiket, goruntu))
    if len(kareler) < hedef:
        raise ValueError(f"{hedef} kare istendi ama yalnizca {len(kareler)} var")

    secilen = tabakali_sec(kareler, hedef, seed)

    tam_dagilim = Counter(kaynak_adi(g.name) for _, g in kareler)
    alt_dagilim = Counter(kaynak_adi(g.name) for _, g in secilen)
    bbox = Counter()
    for etiket, _ in secilen:
        bbox.update(_siniflar(etiket))

    output = output_root / "v10_e1_overfitting_alt_kume"
    output.mkdir(parents=True, exist_ok=True)

    liste = output / "train_images.txt"
    liste.write_text(
        "\n".join(str(g.resolve()).replace("\\", "/") for _, g in secilen) + "\n",
        encoding="utf-8",
    )

    kok = str(dataset_root.resolve()).replace("\\", "/")
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "# E1: overfitting. Etiketler DEGISTIRILMEZ, goruntu kopyalanmaz.\n"
        "# Yalnizca train goruntu listesi 1000 kareye indirilir; augmentasyon\n"
        "# kapatmasi egitim protokolunde beyan edilir (e_serisi.E1).\n"
        f"path: {kok}\n"
        f"train: {str(liste.resolve()).replace(chr(92), '/')}\n"
        f"val: {kok}/images/val\n"
        f"test: {kok}/images/test\n"
        "nc: 4\n"
        "names: [tasit, insan, UAP, UAI]\n",
        encoding="utf-8",
    )

    siniflar = ["tasit", "insan", "UAP", "UAI"]
    manifest = {
        "format": "dataset_manifest_v1",
        "version": output.name,
        "scenario": "E1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_root.resolve()),
        "source_dataset_unchanged": True,
        "copy_mode": "manifest_only",
        "bozulma": None,
        "not": (
            "E1'de veri BOZULMAZ; yalnizca kucultulur. Asil sapma egitim "
            "protokolundedir (augmentasyon kapali, 200 epoch) ve "
            "senaryolar/egitim_protokolu.yaml -> e_serisi.E1 altinda beyan edilir."
        ),
        "ornekleme": {
            "yontem": "kaynak grubuna gore tabakali (en buyuk artik)",
            "seed": seed,
            "gerekce": (
                "Rastgele secim kaynak paylarini kaydirir ve E1'e ikinci bir "
                "degisken (kaynak dagilimi kaymasi, D5'in konusu) katardi."
            ),
        },
        "counts": {
            "tam_kume_kare": len(kareler),
            "alt_kume_kare": len(secilen),
            "alt_kume_bbox": sum(bbox.values()),
            "sinif_bbox": {siniflar[k]: v for k, v in sorted(bbox.items())},
        },
        "kaynak_dagilimi": {
            ad: {
                "tam_kume": tam_dagilim[ad],
                "alt_kume": alt_dagilim.get(ad, 0),
                "tam_pay": round(tam_dagilim[ad] / len(kareler), 4),
                "alt_pay": round(alt_dagilim.get(ad, 0) / len(secilen), 4),
            }
            for ad in sorted(tam_dagilim)
        },
        "files": {
            "data_yaml": str(data_yaml.resolve()),
            "train_images_list": str(liste.resolve()),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"{len(secilen)} kare secildi ({sum(bbox.values())} bbox)")
    print(f"{'kaynak':<12} {'tam pay':>8} {'alt pay':>8} {'sapma':>8}")
    for ad, d in manifest["kaynak_dagilimi"].items():
        print(f"{ad:<12} {d['tam_pay']:>8.4f} {d['alt_pay']:>8.4f} "
              f"{d['alt_pay'] - d['tam_pay']:>+8.4f}")
    print(f"\ndata.yaml: {data_yaml}")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-root", default=ROOT / "veri_surumleri", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.dataset, args.output_root, args.seed)


if __name__ == "__main__":
    main()

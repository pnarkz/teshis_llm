"""D5: kaynak/alan kaymasi veri surumunu uretir.

Egitim setini yalnizca izin verilen kaynak gruplarindan gelen karelerle
sinirlar. Etiketler hic degistirilmez, goruntu kopyalanmaz: D1 gibi
**manifest-only** calisir, yani yalnizca bir train goruntu listesi yazilir.
Bu, bozulmanin "veri secimi" oldugunu ve etiket icerigine dokunulmadigini
yapisal olarak garanti eder.

Kaynak grubu dosya adindan `teshis/veri/istatistik.py::kaynak_adi` ile
turetilir; ayni fonksiyon hem veri raporunda hem de
`teshis/degerlendirme/metrikler.py`'deki kaynak bazli recall hesabinda
kullanilir. Boylece "egitimden cikarilan kaynak" ile "performansi olculen
kaynak" ayni tanima dayanir.

Hipotez: izin verilmeyen kaynaklardan gelen kareler uzerinde recall belirgin
sekilde duser, izin verilen kaynakta ise korunur. Bu ayrisma, toplam mAP'te
kaybolabilir; kaynak bazli kirilim ile olculur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teshis.veri.istatistik import kaynak_adi  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VARSAYILAN_CONFIG = ROOT / "senaryolar/veri/d5_kaynak_alani_kaymasi.yaml"


def find_image(directory: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def build_dataset(
    source: Path, output: Path, izinli: list[str], scenario: str, version: str
) -> Path:
    """Train listesini yalnizca izinli kaynaklarla sinirlar (etiket degismez)."""
    if not izinli:
        raise ValueError("En az bir izinli kaynak belirtilmelidir")

    train_images = source / "images/train"
    train_labels = source / "labels/train"
    if not train_images.is_dir() or not train_labels.is_dir():
        raise FileNotFoundError(f"Kaynak train bolmesi bulunamadi: {source}")

    izinli_kume = set(izinli)
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    tutulan: list[Path] = []
    tum_kaynaklar: Counter[str] = Counter()
    tutulan_kaynaklar: Counter[str] = Counter()
    tutulan_bbox = cikarilan_bbox = 0
    eksik_goruntu = 0

    for label_path in sorted(train_labels.glob("*.txt")):
        image_path = find_image(train_images, label_path.stem)
        if image_path is None:
            eksik_goruntu += 1
            continue
        source_hash.update(label_path.read_bytes())
        kaynak = kaynak_adi(image_path.name)
        tum_kaynaklar[kaynak] += 1
        bbox = sum(
            1 for satir in label_path.read_text(encoding="utf-8").splitlines()
            if len(satir.split()) >= 5
        )
        if kaynak in izinli_kume:
            tutulan.append(image_path)
            tutulan_kaynaklar[kaynak] += 1
            tutulan_bbox += bbox
        else:
            cikarilan_bbox += bbox

    if not tutulan:
        raise ValueError(
            f"Izinli kaynaklar {izinli} icin hic kare bulunamadi. "
            f"Mevcut kaynaklar: {sorted(tum_kaynaklar)}"
        )

    train_list = output / "train_images.txt"
    train_list.write_text(
        "\n".join(str(p.resolve()) for p in tutulan) + "\n", encoding="utf-8"
    )
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "# D5: yalnizca train goruntu listesi daraltilir; etiketler degismez,\n"
        "# goruntu kopyalanmaz. val/test kaynak dataset'in operasyonel bolmeleridir.\n"
        f"path: {source.resolve().as_posix()}\n"
        f"train: {train_list.resolve().as_posix()}\n"
        f"val: {(source / 'images/val').resolve().as_posix()}\n"
        f"test: {(source / 'images/test').resolve().as_posix()}\n"
        "nc: 4\n"
        "names: [tasit, insan, UAP, UAI]\n",
        encoding="utf-8",
    )
    manifest = {
        "format": "dataset_manifest_v1",
        "scenario": scenario,
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source.resolve()),
        "source_dataset_unchanged": True,
        "copy_mode": "manifest_only",
        "labels_modified": False,
        "parameters": {
            "izinli_kaynaklar": sorted(izinli_kume),
            "kaynak_fonksiyonu": "teshis.veri.istatistik.kaynak_adi",
        },
        "counts": {
            "train_frames_before": sum(tum_kaynaklar.values()),
            "train_frames_after": len(tutulan),
            "train_frames_removed": sum(tum_kaynaklar.values()) - len(tutulan),
            "kept_bbox": tutulan_bbox,
            "removed_bbox": cikarilan_bbox,
            "labels_without_images": eksik_goruntu,
            "frames_by_source_before": dict(tum_kaynaklar),
            "frames_by_source_after": dict(tutulan_kaynaklar),
        },
        "val_test_modified": False,
        "source_train_labels_sha256": source_hash.hexdigest(),
        "files": {
            "data_yaml": str(data_yaml.resolve()),
            "train_images_list": str(train_list.resolve()),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="D5 kaynak/alan kaymasi veri surumu")
    parser.add_argument("--dataset", default="C:/Users/ASUS/Desktop/HYZ/dataset", type=Path)
    parser.add_argument("--config", default=VARSAYILAN_CONFIG, type=Path)
    parser.add_argument(
        "--output-dataset", default="veri_surumleri/v07_d5_kaynak_alani_kaymasi", type=Path
    )
    parser.add_argument("--scenario", default="D5")
    parser.add_argument("--version", default="v07_d5_kaynak_alani_kaymasi")
    parser.add_argument(
        "--izinli", nargs="+", default=None, help="Verilmezse config'ten okunur"
    )
    args = parser.parse_args()

    izinli = args.izinli
    if izinli is None:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        izinli = list(config["parametreler"]["izinli_kaynaklar"])
    build_dataset(
        args.dataset.resolve(), args.output_dataset.resolve(), izinli, args.scenario, args.version
    )


if __name__ == "__main__":
    main()

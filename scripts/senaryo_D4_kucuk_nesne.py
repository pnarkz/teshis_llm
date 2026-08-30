"""D4: kucuk nesne sinyal kaybi veri surumunu uretir.

Egitim etiketlerinden, etkin piksel boyutu esigin altinda kalan kutulari
siler. Goruntuler ve diger kutular dokunulmaz; yani model kucuk nesneleri
"arka plan" olarak ogrenir.

Onemli tasarim karari: bozulma esigi ile olcum bandi AYNI formulu kullanir.
Etkin boyut hesabi teshis/degerlendirme/metrikler.py'den import edilir; orada
`BANTLAR` ilk siniri da bu esikle hizalidir. Boylece "egitimden cikarilan
boyut bandi" ile "recall'i olculen bant" birebir ortusur ve D4'un hipotezi
("yalnizca kucuk nesne recall duser") dogrudan test edilebilir olur.

Esik degeri senaryolar/veri/d4_kucuk_nesne_sinyal_kaybi.yaml'dan okunur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teshis.degerlendirme.metrikler import boyut_bandi, etkin_sqrt_alan  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
CLASS_NAMES = {0: "tasit", 1: "insan", 2: "UAP", 3: "UAI"}
VARSAYILAN_CONFIG = ROOT / "senaryolar/veri/d4_kucuk_nesne_sinyal_kaybi.yaml"


def find_image(directory: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except (OSError, NotImplementedError):
        shutil.copy2(source, target)


def build_dataset(source: Path, output: Path, esik_px: float, scenario: str, version: str) -> Path:
    """Etkin boyutu esigin altinda kalan train kutularini siler."""
    if esik_px <= 0:
        raise ValueError("esik_px pozitif olmalidir")

    source_labels = sorted((source / "labels/train").glob("*.txt"))
    if not source_labels:
        raise FileNotFoundError(f"Train etiketi bulunamadi: {source / 'labels/train'}")
    (output / "images/train").mkdir(parents=True, exist_ok=True)
    (output / "labels/train").mkdir(parents=True, exist_ok=True)

    source_hash = hashlib.sha256()
    image_count = missing_images = 0
    kept_rows = removed_rows = invalid_rows = 0
    removed_by_class: Counter[str] = Counter()
    kept_by_band: Counter[str] = Counter()
    removed_by_band: Counter[str] = Counter()

    for label_path in source_labels:
        image_path = find_image(source / "images/train", label_path.stem)
        if image_path is None:
            missing_images += 1
            continue
        link_or_copy(image_path, output / "images/train" / image_path.name)
        image_count += 1
        source_hash.update(label_path.read_bytes())

        with Image.open(image_path) as goruntu:
            genislik, yukseklik = goruntu.size

        tutulan: list[str] = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            alanlar = raw.split()
            if len(alanlar) != 5:
                if raw.strip():
                    invalid_rows += 1
                    tutulan.append(raw)  # bozuk satirlari oldugu gibi koru
                continue
            try:
                sinif = int(float(alanlar[0]))
                _, _, w, h = (float(x) for x in alanlar[1:5])
            except ValueError:
                invalid_rows += 1
                tutulan.append(raw)
                continue
            boyut = etkin_sqrt_alan(w, h, genislik, yukseklik)
            bant = boyut_bandi(boyut)
            if boyut < esik_px:
                removed_rows += 1
                removed_by_class[CLASS_NAMES.get(sinif, str(sinif))] += 1
                removed_by_band[bant] += 1
            else:
                tutulan.append(raw)
                kept_rows += 1
                kept_by_band[bant] += 1

        (output / "labels/train" / label_path.name).write_text(
            "\n".join(tutulan) + ("\n" if tutulan else ""), encoding="utf-8"
        )

    data_yaml = output / "data.yaml"
    # val/test kaynak dataset'in operasyonel bolmeleridir; kilitli tanı seti DEGIL.
    data_yaml.write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
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
        "parameters": {
            "etkin_sqrt_alan_esigi_px": esik_px,
            "olcum_formulu": "teshis.degerlendirme.metrikler.etkin_sqrt_alan",
            "referans_uzun_kenar": 640,
        },
        "counts": {
            "train_images": image_count,
            "missing_images": missing_images,
            "kept_bbox_rows": kept_rows,
            "removed_bbox_rows": removed_rows,
            "invalid_rows_preserved": invalid_rows,
            "removed_by_class": dict(removed_by_class),
            "removed_by_band": dict(removed_by_band),
            "kept_by_band": dict(kept_by_band),
        },
        "val_test_modified": False,
        "source_train_labels_sha256": source_hash.hexdigest(),
        "files": {"data_yaml": str(data_yaml.resolve()), "output_dataset": str(output.resolve())},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="D4 kucuk nesne sinyal kaybi veri surumu")
    parser.add_argument("--dataset", default="C:/Users/ASUS/Desktop/HYZ/dataset", type=Path)
    parser.add_argument("--config", default=VARSAYILAN_CONFIG, type=Path)
    parser.add_argument(
        "--output-dataset", default="veri_surumleri/v06_d4_kucuk_nesne_sinyal_kaybi", type=Path
    )
    parser.add_argument("--scenario", default="D4")
    parser.add_argument("--version", default="v06_d4_kucuk_nesne_sinyal_kaybi")
    parser.add_argument("--esik-px", type=float, default=None, help="Verilmezse config'ten okunur")
    args = parser.parse_args()

    esik = args.esik_px
    if esik is None:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        esik = float(config["parametreler"]["etkin_sqrt_alan_esigi_px"])
    build_dataset(
        args.dataset.resolve(), args.output_dataset.resolve(), esik, args.scenario, args.version
    )


if __name__ == "__main__":
    main()

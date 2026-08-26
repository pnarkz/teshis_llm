"""Sinif karisikligi (class ID hatasi) senaryo ailesini yerel GPU'da uretir ve egitir.

Iki senaryoyu ayni kodla uretir; fark yalnizca karistirilan sinif ciftidir:

- **D3** (varsayilan): UAP <-> UAI. Gercekci ama tanı setinde yalnizca 15 ve 17
  bbox ile olculebiliyor, bu yuzden istatistiksel gucu dusuk.
- **D3b**: tasit <-> insan. Ayni bilimsel soru (sinif ID etiket hatasinin
  etkisi), ama tanı setinde 1.264 + 2.718 = 3.982 bbox ile olculuyor.

Ikisi birlikte, projenin belirsizlik raporlama ilkesinin vitrinidir: ayni
bozulma, nadir siniflarda genis guven araligi, bol siniflarda dar aralik
uretir. Varsayilanlar D3'un orijinal parametreleridir; boylece D3 bu
degisiklikten sonra da ayni sekilde yeniden uretilebilir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teshis.egitim.protokol import egitim_kwargs  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
CLASS_NAMES = {0: "tasit", 1: "insan", 2: "UAP", 3: "UAI"}


def find_image(directory: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def link_or_copy(source: Path, target: Path) -> None:
    """Goruntuyu hardlink ile baglar; ayni birimde degilse kopyalar.

    17.515 goruntunun tam kopyasi hem yavas hem gereksiz yer kaplar; icerik
    zaten salt okunur kaynaktan geliyor ve degistirilmiyor.
    """
    if target.exists():
        return
    try:
        os.link(source, target)
    except (OSError, NotImplementedError):
        shutil.copy2(source, target)


def build_dataset(
    source: Path,
    output: Path,
    swap_ratio: float,
    seed: int,
    class_pair: tuple[int, int] = (2, 3),
    scenario: str = "D3",
    version: str = "v04_d3_uap_uai_sinif_karisikligi",
) -> Path:
    if not 0 <= swap_ratio <= 1:
        raise ValueError("swap_ratio 0 ile 1 arasinda olmalidir")
    first, second = class_pair
    if first == second:
        raise ValueError("class_pair iki farkli sinif icermelidir")
    hedef = {str(first), str(second)}
    takas = {str(first): str(second), str(second): str(first)}
    ad_ilk, ad_ikinci = CLASS_NAMES.get(first, str(first)), CLASS_NAMES.get(second, str(second))

    rng = random.Random(seed)
    source_labels = sorted((source / "labels/train").glob("*.txt"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "images/train").mkdir(parents=True, exist_ok=True)
    (output / "labels/train").mkdir(parents=True, exist_ok=True)
    rows = []
    source_hash = hashlib.sha256()
    image_count = missing_images = 0

    for label_path in source_labels:
        image_path = find_image(source / "images/train", label_path.stem)
        if image_path is None:
            missing_images += 1
            continue
        link_or_copy(image_path, output / "images/train" / image_path.name)
        image_count += 1
        source_hash.update(label_path.read_bytes())
        for line_index, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            fields = raw.split()
            if len(fields) == 5 and fields[0] in hedef:
                rows.append((label_path.name, line_index, raw))

    swap_count = round(len(rows) * swap_ratio)
    selected = set(rng.sample(range(len(rows)), swap_count))
    row_ids = {key: index for index, key in enumerate(rows)}
    changed_rows = 0
    changed_by_class = {f"{ad_ilk}_to_{ad_ikinci}": 0, f"{ad_ikinci}_to_{ad_ilk}": 0}

    for label_path in source_labels:
        image_path = find_image(source / "images/train", label_path.stem)
        if image_path is None:
            continue
        output_label = output / "labels/train" / label_path.name
        output_label.parent.mkdir(parents=True, exist_ok=True)
        changed = []
        for line_index, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            fields = raw.split()
            if len(fields) == 5 and fields[0] in hedef:
                row_index = row_ids[(label_path.name, line_index, raw)]
                if row_index in selected:
                    old_id = fields[0]
                    fields[0] = takas[old_id]
                    changed_rows += 1
                    eski_ad = CLASS_NAMES.get(int(old_id), old_id)
                    yeni_ad = CLASS_NAMES.get(int(fields[0]), fields[0])
                    changed_by_class[f"{eski_ad}_to_{yeni_ad}"] += 1
                    raw = " ".join(fields)
            changed.append(raw)
        output_label.write_text("\n".join(changed) + ("\n" if changed else ""), encoding="utf-8")

    data_yaml = output / "data.yaml"
    # val, kaynak dataset'in operasyonel val bolmesidir; val_diagnostic DEGILDIR.
    # val_diagnostic egitim val'i olarak kullanilirsa best.pt, sonradan
    # senaryolari karsilastirmak icin kullandigimiz setin uzerinde secilmis olur
    # (checkpoint secimi = degerlendirme seti). Bu, o senaryoya digerlerine gore
    # iyimser bir yanlilik kazandirir. Bkz. tests/test_veri_surumu_val.py.
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
        "seed": seed,
        "parameters": {
            "swap_ratio": swap_ratio,
            "class_ids": [first, second],
            "class_names": [ad_ilk, ad_ikinci],
        },
        "counts": {
            "train_images": image_count,
            "missing_images": missing_images,
            "target_rows_before": len(rows),
            "changed_rows": changed_rows,
            "changed_by_class": changed_by_class,
        },
        "val_diagnostic_modified": False,
        "source_train_labels_sha256": source_hash.hexdigest(),
        "files": {"data_yaml": str(data_yaml.resolve()), "output_dataset": str(output.resolve())},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinif karisikligi senaryosu (D3 / D3b)")
    parser.add_argument("--dataset", default="C:/Users/ASUS/Desktop/HYZ/dataset", type=Path)
    parser.add_argument("--model", default="main_model.pt", type=Path)
    parser.add_argument("--output-dataset", default="veri_surumleri/v04_d3_uap_uai_sinif_karisikligi", type=Path)
    parser.add_argument("--output-root", default="experiments", type=Path)
    parser.add_argument("--scenario", default="D3")
    parser.add_argument("--version", default="v04_d3_uap_uai_sinif_karisikligi")
    parser.add_argument("--run-name", default=None, help="Varsayilan: run_<senaryo>_<seed>_local")
    parser.add_argument(
        "--class-pair", type=int, nargs=2, default=(2, 3), metavar=("A", "B"),
        help="Karistirilacak sinif ID cifti. D3: 2 3 (UAP/UAI), D3b: 0 1 (tasit/insan)",
    )
    parser.add_argument("--swap-ratio", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sadece-veri", action="store_true", help="Yalnizca veri surumunu uret, egitme")
    args = parser.parse_args()

    data_yaml = build_dataset(
        args.dataset.resolve(), args.output_dataset.resolve(), args.swap_ratio, args.seed,
        tuple(args.class_pair), args.scenario, args.version,
    )
    if args.sadece_veri:
        print(f"data_yaml={data_yaml}")
        return

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA bulunamadi; bu senaryo yerel GPU ile kosulmalidir.")
    run_name = args.run_name or f"run_{args.scenario}_{args.seed}_local"
    model = YOLO(str(args.model.resolve()))
    model.train(
        data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, epochs=args.epochs,
        device=0, workers=0, seed=args.seed,
        project=str(args.output_root.resolve()), name=run_name,
        exist_ok=True, plots=False, val=True,
        **egitim_kwargs(),
    )
    print(f"best_model={args.output_root.resolve() / run_name / 'weights/best.pt'}")


if __name__ == "__main__":
    main()

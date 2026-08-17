"""Mevcut best modelden kontrollu 768px fine-tune kosusu."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def nadir_kare_mi(label_path: Path) -> bool:
    """UAP veya UAI iceren kareleri belirler."""
    try:
        satirlar = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return False
    return any(
        satir.strip() and satir.split()[0] in {"2", "3"}
        for satir in satirlar
    )


def egitim_listesi_uret(dataset_root: Path, output_root: Path, repeat: int, seed: int) -> Path:
    """Oversampling listesini olusturur; orijinal dataset'e yazmaz."""
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    normal: list[str] = []
    nadir: list[str] = []

    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = next(
            (image_dir / f"{label_path.stem}{ext}" for ext in IMAGE_EXTENSIONS
             if (image_dir / f"{label_path.stem}{ext}").exists()),
            None,
        )
        if image_path is None:
            continue
        (nadir if nadir_kare_mi(label_path) else normal).append(str(image_path.resolve()))

    satirlar = normal + nadir * repeat
    random.Random(seed).shuffle(satirlar)
    output_root.mkdir(parents=True, exist_ok=True)
    # DDP kosu klasorunu yeniden olusturabildigi icin listeyi disarida tut.
    liste = output_root.parent / "train_oversampled.txt"
    liste.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print(f"normal={len(normal)} nadir={len(nadir)} repeat={repeat} toplam={len(satirlar)}")
    return liste


def data_yaml_uret(dataset_root: Path, train_list: Path, output_root: Path, val_root: Path | None) -> Path:
    """Ultralytics icin egitim YAML'i yazar."""
    val_images = (val_root / "images").resolve() if val_root else (dataset_root / "images" / "val").resolve()
    yaml = (
        f"path: {dataset_root.resolve().as_posix()}\n"
        f"train: {train_list.resolve().as_posix()}\n"
        f"val: {val_images.as_posix()}\n"
        "nc: 4\n"
        "names: [tasit, insan, UAP, UAI]\n"
    )
    # DDP alt surecleri kosu klasorunu yeniden kurabildigi icin YAML'i onun disinda tut.
    data_path = output_root.parent / "data_v3.yaml"
    data_path.write_text(yaml, encoding="utf-8")
    return data_path


def _egit(
    dataset_root: Path,
    model_path: Path,
    output_root: Path,
    val_root: Path | None = None,
    repeat: int = 5,
    epochs: int = 40,
    batch: int = 12,
    imgsz: int = 768,
    seed: int = 42,
) -> None:
    """Egitimi notebook veya komut satirindan baslatir."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as hata:
        raise RuntimeError("Bu kosu Kaggle GPU ortaminda ultralytics ile calistirilmalidir.") from hata

    train_list = egitim_listesi_uret(dataset_root, output_root, repeat, seed)
    data_path = data_yaml_uret(dataset_root, train_list, output_root, val_root)
    if not data_path.is_file():
        raise FileNotFoundError(f"Egitim YAML'i yazilamadi: {data_path}")
    print(f"data_yaml={data_path} mevcut={data_path.is_file()}")
    device = [0, 1] if torch.cuda.device_count() > 1 else 0
    print(f"dataset={dataset_root}")
    print(f"model={model_path}")
    print(f"device={device} imgsz={imgsz} batch={batch}")
    model = YOLO(str(model_path.resolve()))
    model.train(
        data=str(data_path),
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        device=device,
        workers=4,
        seed=seed,
        deterministic=True,
        cos_lr=True,
        patience=20,
        project=str(output_root.parent),
        name=output_root.name,
        exist_ok=True,
        plots=True,
        val=True,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.15,
        degrees=5.0,
        translate=0.08,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.7,
        close_mosaic=15,
        mixup=0.0,
        copy_paste=0.0,
    )


def _kaggle_bul(aranan: str, dataset_mi: bool) -> Path:
    """Kaggle input altinda dataset veya model dosyasini bulur."""
    input_root = Path("/kaggle/input")
    if dataset_mi:
        adaylar = [
            yol.parent.parent
            for yol in input_root.rglob("images/train")
            if (yol.parent.parent / "labels" / "train").is_dir()
        ]
    else:
        adaylar = list(input_root.rglob(aranan))
    if not adaylar:
        raise FileNotFoundError(f"Kaggle input icinde bulunamadi: {aranan}")
    return sorted(adaylar, key=str)[0]


def main() -> None:
    # Notebook kernelinde sys.argv, kernelin kendi parametrelerini tasir.
    if "ipykernel" in sys.modules and Path("/kaggle/input").is_dir():
        dataset = _kaggle_bul("dataset", dataset_mi=True)
        try:
            model = _kaggle_bul("final_best.pt", dataset_mi=False)
        except FileNotFoundError:
            model = _kaggle_bul("best.pt", dataset_mi=False)
        _egit(dataset, model, Path("/kaggle/working/v3_768_finetune"))
        return

    parser = argparse.ArgumentParser(description="YOLO11m kontrollu iyilestirme egitimi")
    parser.add_argument("--dataset", required=True, help="dataset kok yolu")
    parser.add_argument("--model", required=True, help="Baslangic best.pt yolu")
    parser.add_argument("--output", default="runs/v3_768_finetune")
    parser.add_argument("--val-root", default=None, help="Opsiyonel val_diagnostic yolu")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    output_root = Path(args.output).resolve()
    _egit(
        dataset_root,
        Path(args.model).resolve(),
        output_root,
        Path(args.val_root).resolve() if args.val_root else None,
        args.repeat,
        args.epochs,
        args.batch,
        args.imgsz,
        args.seed,
    )


if __name__ == "__main__":
    main()

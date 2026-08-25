"""Start a controlled Ultralytics training run from a versioned YAML."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .kayit import write_run_manifest
from .protokol import egitim_kwargs


def run_training(
    model_path: Path,
    data_path: Path,
    output_root: Path,
    scenario: str,
    epochs: int,
    batch: int,
    imgsz: int,
    seed: int,
    device: int | str,
    workers: int,
) -> Path:
    """Train without touching source data or the selected source model."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Bu komut ultralytics ve torch kurulu yerel/GPU ortaminda calisir.") from error

    model_path = model_path.resolve()
    data_path = data_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model bulunamadi: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Data YAML bulunamadi: {data_path}")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU istendi fakat CUDA kullanilamiyor. --device cpu ile bilincli baslatin.")

    run_name = f"run_{datetime.now():%Y%m%d_%H%M%S}_{scenario}_{seed}"
    run_dir = output_root.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(
        run_dir,
        {
            "scenario": scenario,
            "model": str(model_path),
            "data": str(data_path),
            "device": str(device),
            "seed": seed,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "test_evaluated_during_training": False,
        },
    )

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        device=device,
        workers=workers,
        seed=seed,
        project=str(output_root.resolve()),
        name=run_name,
        exist_ok=True,
        # D1 tanı kosusunda labels.jpg uretimi Windows'ta takilabildigi icin kapali.
        plots=False,
        val=True,
        pretrained=str(model_path),
        **egitim_kwargs(),
    )
    print(json.dumps({"run_dir": str(run_dir), "weights": str(run_dir / "weights")}, indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Versioned YOLO training run")
    parser.add_argument("--model", default="main_model.pt")
    parser.add_argument("--data", default="veri_surumleri/v01_d1_sinif_yetersizligi/data.yaml")
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--scenario", default="D1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0", help="GPU id, or cpu")
    parser.add_argument("--workers", type=int, default=0, help="Windows yerel kosuda 0 daha kararlidir")
    args = parser.parse_args()
    device: int | str = "cpu" if args.device.lower() == "cpu" else int(args.device)
    run_training(
        Path(args.model), Path(args.data), Path(args.output_root), args.scenario,
        args.epochs, args.batch, args.imgsz, args.seed, device, args.workers,
    )


if __name__ == "__main__":
    main()

"""Komut girisi: kaggle_D2a_lokalizasyon_gurultusu.

Uygulama: teshis/veri/senaryo_d2a_lokalizasyon_gurultusu.py
Dosya haritasi: docs/KOD_HARITASI.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teshis.egitim.protokol import egitim_kwargs  # noqa: E402

IMGSZ = 768
EPOCHS = 30
BATCH = 12


from teshis.veri.senaryo_d2a_lokalizasyon_gurultusu import (
    SEED,
    find_image,
    link_or_copy,
    noisy_label,
    build_d2a,
)


def find_input_dataset() -> Path:
    root = Path("/kaggle/input")
    candidates = []
    for image_dir in root.rglob("images/train"):
        dataset = image_dir.parent.parent
        if (dataset / "labels/train").is_dir():
            candidates.append(dataset)
    if not candidates:
        raise FileNotFoundError("/kaggle/input altinda images/train ve labels/train bulunamadi")
    return sorted(candidates, key=str)[0]


def find_input_model() -> Path:
    root = Path("/kaggle/input")
    preferred = list(root.rglob("main_model.pt"))
    if preferred:
        return sorted(preferred, key=str)[0]
    fallback = list(root.rglob("best.pt"))
    if fallback:
        return sorted(fallback, key=str)[0]
    raise FileNotFoundError("/kaggle/input altinda main_model.pt bulunamadi")


def main() -> None:
    from ultralytics import YOLO
    import torch

    source = find_input_dataset()
    model_path = find_input_model()
    work = Path("/kaggle/working")
    dataset = work / "v02_d2a_lokalizasyon_gurultusu"
    output = work / "experiments"
    dataset.mkdir(parents=True, exist_ok=True)
    manifest = build_d2a(source, dataset)
    print(json.dumps(manifest, indent=2))
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.train(
        data=str(dataset / "data.yaml"),
        imgsz=IMGSZ,
        batch=BATCH,
        epochs=EPOCHS,
        device=device,
        workers=2,
        seed=SEED,
        project=str(output),
        name="run_D2a_42",
        exist_ok=True,
        plots=True,
        val=True,
        **egitim_kwargs(),
    )
    print(f"best_model={output / 'run_D2a_42/weights/best.pt'}")


if __name__ == "__main__":
    main()

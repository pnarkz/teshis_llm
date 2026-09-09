"""Komut girisi: kaggle_D2b_eksik_etiket.

Uygulama: teshis/veri/senaryo_d2b_eksik_etiket.py
Dosya haritasi: docs/KOD_HARITASI.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kaggle_D2a_lokalizasyon_gurultusu import find_input_dataset, find_input_model
from teshis.egitim.protokol import egitim_kwargs  # noqa: E402


IMGSZ = 768
EPOCHS = 30
BATCH = 12


from teshis.veri.senaryo_d2b_eksik_etiket import SEED, build_d2b


def main() -> None:
    import torch
    from ultralytics import YOLO

    source = find_input_dataset()
    model_path = find_input_model()
    output_root = Path("/kaggle/working")
    dataset = output_root / "v03_d2b_eksik_etiket"
    experiment_root = output_root / "experiments"
    dataset.mkdir(parents=True, exist_ok=True)
    manifest = build_d2b(source, dataset)
    print(json.dumps(manifest, indent=2))
    model = YOLO(str(model_path))
    model.train(
        data=str(dataset / "data.yaml"),
        imgsz=IMGSZ,
        batch=BATCH,
        epochs=EPOCHS,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        seed=SEED,
        project=str(experiment_root),
        name="run_D2b_42",
        exist_ok=True,
        plots=True,
        val=True,
        **egitim_kwargs(),
    )
    print(f"best_model={experiment_root / 'run_D2b_42/weights/best.pt'}")


if __name__ == "__main__":
    main()

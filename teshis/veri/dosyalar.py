"""Veri surumleri icin goruntu bulma ve kaynak etiket ozeti."""
from __future__ import annotations
import hashlib
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def find_image(image_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None



def train_labels_sha256(dataset_root: Path) -> tuple[str, int, int]:
    """Train etiketlerinin birlesik sha256'sini, dosya ve bbox sayisini dondurur.

    Diger senaryo ureticileri (scripts/senaryo_*.py, scripts/kaggle_*.py) ayni alani
    manifest'e yaziyor. v00 icin de kaydedilince, tum bozuk surumlerin ayni
    kaynaktan turedigi hash karsilastirmasiyla ispatlanabilir hale gelir.
    """
    digest = hashlib.sha256()
    dosya = bbox = 0
    for label_path in sorted((dataset_root / "labels" / "train").iterdir()):
        if label_path.suffix != ".txt":
            continue
        veri = label_path.read_bytes()
        digest.update(veri)
        dosya += 1
        bbox += sum(1 for satir in veri.decode("utf-8").splitlines() if len(satir.split()) >= 5)
    return digest.hexdigest(), dosya, bbox


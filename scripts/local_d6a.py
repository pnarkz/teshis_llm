"""D6a: split sizintisi degerlendirme kumesini uretir.

Senaryo, veri setinin **degerlendirme tarafindaki** bir kusuru gosterir:
train karelerinin bir kismi val icine de sizmissa, val skoru yapay olarak
yuksek cikar ve gelistirme boyunca yaniltir.

Onemli tasarim notu: bu senaryo YENIDEN EGITIM GEREKTIRMEZ. Sizinti modeli
degil olcumu bozar; bu yuzden zaten egitilmis saglikli referans (v00) modeli,
iki farkli degerlendirme kumesinde olculur:

  1. Kilitli temiz tanı seti (val_diagnostic)      -> gercek performans
  2. Sizintili kume (tanı seti + train kareleri)   -> yapay yuksek performans

Aradaki fark, "val'iniz size yalan soyluyor" iddiasinin dogrudan olcusudur.

Kural 3 (test seti final asamaya kadar kullanilamaz) ihlal edilmez: temiz
karsilastirma tabani olarak kilitli tanı seti kullanilir, test setine hic
dokunulmaz. Senaryo config'indeki "test-val farki" ifadesi ayni olguyu
anlatir; burada temiz taban olarak tanı seti alinmistir.
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

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teshis.veri.istatistik import kaynak_adi  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VARSAYILAN_CONFIG = ROOT / "senaryolar/veri/d6a_split_sizintisi.yaml"


def find_image(directory: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except (OSError, NotImplementedError):
        shutil.copy2(source, target)


def build_dataset(
    source: Path,
    tani_seti: Path,
    output: Path,
    sizdirma_orani: float,
    seed: int,
    scenario: str = "D6a",
    version: str = "v08_d6a_split_sizintisi",
) -> Path:
    """Tanı setine train karelerini ekleyerek sizintili degerlendirme kumesi uretir."""
    if not 0 < sizdirma_orani < 1:
        raise ValueError("sizdirma_orani 0 ile 1 arasinda olmalidir")
    tani_images = tani_seti / "images"
    tani_labels = tani_seti / "labels"
    if not tani_images.is_dir() or not tani_labels.is_dir():
        raise FileNotFoundError(f"Tanı seti bulunamadi: {tani_seti}")

    temiz_goruntuler = sorted(
        p for p in tani_images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not temiz_goruntuler:
        raise FileNotFoundError(f"Tanı setinde goruntu yok: {tani_images}")

    # Sizdirilacak kare sayisi: son kumenin `sizdirma_orani` kadari train'den gelsin.
    # n_temiz / (n_temiz + n_sizan) = 1 - oran  =>  n_sizan = n_temiz * oran / (1 - oran)
    n_temiz = len(temiz_goruntuler)
    n_sizan = round(n_temiz * sizdirma_orani / (1 - sizdirma_orani))

    train_labels = sorted((source / "labels/train").glob("*.txt"))
    adaylar = [p for p in train_labels if find_image(source / "images/train", p.stem)]
    if len(adaylar) < n_sizan:
        raise ValueError(f"Train'de yeterli kare yok: {len(adaylar)} < {n_sizan}")
    secilen = random.Random(seed).sample(adaylar, n_sizan)

    out_images = output / "images"
    out_labels = output / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # 1) Temiz tanı seti karelerini bagla.
    for goruntu in temiz_goruntuler:
        link_or_copy(goruntu, out_images / goruntu.name)
        etiket = tani_labels / f"{goruntu.stem}.txt"
        if etiket.is_file():
            link_or_copy(etiket, out_labels / etiket.name)

    # 2) Train karelerini SIZDIR (ayni dosyalar, degistirilmeden).
    sizan_hash = hashlib.sha256()
    sizan_kaynak: dict[str, int] = {}
    for etiket in secilen:
        goruntu = find_image(source / "images/train", etiket.stem)
        link_or_copy(goruntu, out_images / goruntu.name)
        link_or_copy(etiket, out_labels / etiket.name)
        sizan_hash.update(etiket.read_bytes())
        ad = kaynak_adi(goruntu.name)
        sizan_kaynak[ad] = sizan_kaynak.get(ad, 0) + 1

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "# D6a: SIZINTILI degerlendirme kumesi. Train kareleri val icine eklenmistir;\n"
        "# bu kume yalnizca sizintinin etkisini olcmek icindir, model secimi icin\n"
        "# KULLANILMAMALIDIR.\n"
        f"path: {output.resolve().as_posix()}\n"
        "train: images\n"
        "val: images\n"
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
        "clean_reference_set": str(tani_seti.resolve()),
        "source_dataset_unchanged": True,
        "labels_modified": False,
        "requires_training": False,
        "seed": seed,
        "parameters": {"kaynak_sizdirma_orani": sizdirma_orani},
        "counts": {
            "temiz_kare": n_temiz,
            "sizdirilan_train_karesi": n_sizan,
            "toplam_kare": n_temiz + n_sizan,
            "gerceklesen_sizinti_orani": round(n_sizan / (n_temiz + n_sizan), 4),
            "sizan_kaynak_dagilimi": sizan_kaynak,
        },
        "test_split_used": False,
        "sizan_train_etiketleri_sha256": sizan_hash.hexdigest(),
        "files": {"data_yaml": str(data_yaml.resolve())},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="D6a sizintili degerlendirme kumesi")
    parser.add_argument("--dataset", default="C:/Users/ASUS/Desktop/HYZ/dataset", type=Path)
    parser.add_argument("--tani-seti", default=ROOT / "val_diagnostic", type=Path)
    parser.add_argument("--config", default=VARSAYILAN_CONFIG, type=Path)
    parser.add_argument("--output", default=ROOT / "veri_surumleri/v08_d6a_split_sizintisi", type=Path)
    parser.add_argument("--oran", type=float, default=None, help="Verilmezse config'ten okunur")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    oran = args.oran if args.oran is not None else float(config["parametreler"]["kaynak_sizdirma_orani"])
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    build_dataset(args.dataset.resolve(), args.tani_seti.resolve(), args.output.resolve(), oran, seed)


if __name__ == "__main__":
    main()

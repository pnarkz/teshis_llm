"""D6b: tekrar agirligi veri surumunu uretir.

Bir kare kumesi egitim listesinde defalarca tekrarlanir; boylece o sahneler
egitimde asiri temsil edilir. Gercek hayatta bu, ayni sahnenin cok kez
yakalanmasi veya augment kopyalarinin veri setine birden fazla girmesiyle
olusur.

D1 ve D5 gibi **manifest-only** calisir: etiketler hic degistirilmez, goruntu
kopyalanmaz, yalnizca train goruntu listesi yeniden yazilir. Bozulmanin
"veri agirliklandirmasi" oldugu boylece yapisal olarak garanti edilir.

Hipotez: asiri temsil edilen sahneler modeli kendine ceker; bu sahnelerde
bulunmayan nadir siniflarin performansi goreli olarak bozulur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from teshis.veri.istatistik import kaynak_adi  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
CLASS_NAMES = {0: "tasit", 1: "insan", 2: "UAP", 3: "UAI"}
VARSAYILAN_CONFIG = ROOT / "senaryolar/veri/d6b_tekrar_agirligi.yaml"


def find_image(directory: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _siniflar(etiket_yolu: Path) -> Counter:
    sayim: Counter = Counter()
    for satir in etiket_yolu.read_text(encoding="utf-8").splitlines():
        alanlar = satir.split()
        if len(alanlar) >= 5:
            try:
                sayim[CLASS_NAMES.get(int(float(alanlar[0])), alanlar[0])] += 1
            except ValueError:
                continue
    return sayim


def build_dataset(
    source: Path,
    output: Path,
    hedef_tekrar: int,
    secim_orani: float,
    seed: int,
    scenario: str = "D6b",
    version: str = "v09_d6b_tekrar_agirligi",
) -> Path:
    """Secilen karelerin egitim listesinde `hedef_tekrar` kez gecmesini saglar."""
    if hedef_tekrar < 2:
        raise ValueError("hedef_tekrar en az 2 olmalidir")
    if not 0 < secim_orani < 1:
        raise ValueError("secim_orani 0 ile 1 arasinda olmalidir")

    train_images = source / "images/train"
    train_labels = source / "labels/train"
    etiketler = sorted(train_labels.glob("*.txt"))
    kareler = [(e, find_image(train_images, e.stem)) for e in etiketler]
    kareler = [(e, g) for e, g in kareler if g is not None]
    if not kareler:
        raise FileNotFoundError(f"Train karesi bulunamadi: {train_images}")

    rng = random.Random(seed)
    n_secilen = max(1, round(len(kareler) * secim_orani))
    secilen = set(rng.sample(range(len(kareler)), n_secilen))

    satirlar: list[str] = []
    kaynak_hash = hashlib.sha256()
    tekrarlanan_sinif: Counter = Counter()
    normal_sinif: Counter = Counter()
    tekrarlanan_kaynak: Counter = Counter()

    for indis, (etiket, goruntu) in enumerate(kareler):
        kaynak_hash.update(etiket.read_bytes())
        yol = str(goruntu.resolve())
        siniflar = _siniflar(etiket)
        if indis in secilen:
            satirlar.extend([yol] * hedef_tekrar)
            for ad, adet in siniflar.items():
                tekrarlanan_sinif[ad] += adet * hedef_tekrar
            tekrarlanan_kaynak[kaynak_adi(goruntu.name)] += 1
        else:
            satirlar.append(yol)
            for ad, adet in siniflar.items():
                normal_sinif[ad] += adet

    rng.shuffle(satirlar)
    output.mkdir(parents=True, exist_ok=True)
    liste = output / "train_images.txt"
    liste.write_text("\n".join(satirlar) + "\n", encoding="utf-8")

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "# D6b: yalnizca train goruntu listesi yeniden agirliklandirilir.\n"
        "# Etiketler degistirilmez, goruntu kopyalanmaz.\n"
        f"path: {source.resolve().as_posix()}\n"
        f"train: {liste.resolve().as_posix()}\n"
        f"val: {(source / 'images/val').resolve().as_posix()}\n"
        f"test: {(source / 'images/test').resolve().as_posix()}\n"
        "nc: 4\n"
        "names: [tasit, insan, UAP, UAI]\n",
        encoding="utf-8",
    )

    toplam_bbox = sum(tekrarlanan_sinif.values()) + sum(normal_sinif.values())
    manifest = {
        "format": "dataset_manifest_v1",
        "scenario": scenario,
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source.resolve()),
        "source_dataset_unchanged": True,
        "copy_mode": "manifest_only",
        "labels_modified": False,
        "seed": seed,
        "parameters": {"hedef_tekrar": hedef_tekrar, "secim_orani": secim_orani},
        "counts": {
            "train_frames_before": len(kareler),
            "tekrarlanan_kare": n_secilen,
            "train_satiri_after": len(satirlar),
            "tekrar_katsayisi": round(len(satirlar) / len(kareler), 3),
            "tekrarlanan_karelerin_egitim_payi": round(
                n_secilen * hedef_tekrar / len(satirlar), 4
            ),
            "bbox_payi_tekrarlanan": {
                ad: round(tekrarlanan_sinif[ad] / toplam_bbox, 4)
                for ad in sorted(set(tekrarlanan_sinif) | set(normal_sinif))
            },
            "tekrarlanan_kaynak_dagilimi": dict(tekrarlanan_kaynak),
        },
        "val_test_modified": False,
        "source_train_labels_sha256": kaynak_hash.hexdigest(),
        "files": {"data_yaml": str(data_yaml.resolve()), "train_images_list": str(liste.resolve())},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="D6b tekrar agirligi veri surumu")
    parser.add_argument("--dataset", default="C:/Users/ASUS/Desktop/HYZ/dataset", type=Path)
    parser.add_argument("--config", default=VARSAYILAN_CONFIG, type=Path)
    parser.add_argument("--output", default=ROOT / "veri_surumleri/v09_d6b_tekrar_agirligi", type=Path)
    parser.add_argument("--tekrar", type=int, default=None, help="Verilmezse config'ten okunur")
    parser.add_argument(
        "--secim-orani", type=float, default=0.01,
        help="Tekrarlanacak kare orani (varsayilan %1).",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tekrar = args.tekrar if args.tekrar is not None else int(config["parametreler"]["hedef_tekrar"])
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    build_dataset(args.dataset.resolve(), args.output.resolve(), tekrar, args.secim_orani, seed)


if __name__ == "__main__":
    main()

"""Senaryo tasimasinda eski girisler ve veri davranisi korunur."""
import importlib
import json
import random
from pathlib import Path

import pytest


@pytest.mark.parametrize("giris,modul,fonksiyon", [
    ("teshis.veri.surum_uret", "teshis.veri.referans_v00_saglikli", "build_v00"),
    ("teshis.veri.surum_uret", "teshis.veri.senaryo_d1_sinif_yetersizligi", "build_d1"),
    ("teshis.veri.bozulmalar", "teshis.veri.senaryo_d1_sinif_yetersizligi", "d1_remove_class_frames"),
    ("scripts.kaggle_D2a_lokalizasyon_gurultusu", "teshis.veri.senaryo_d2a_lokalizasyon_gurultusu", "build_d2a"),
    ("scripts.kaggle_D2b_eksik_etiket", "teshis.veri.senaryo_d2b_eksik_etiket", "build_d2b"),
    ("scripts.senaryo_D2b_eksik_etiket", "teshis.veri.senaryo_d2b_eksik_etiket", "build_dataset"),
    ("scripts.senaryo_D3_D3b_sinif_karisikligi", "teshis.veri.senaryo_d3_d3b_sinif_karisikligi", "build_dataset"),
    ("scripts.senaryo_D4_kucuk_nesne", "teshis.veri.senaryo_d4_kucuk_nesne_sinyal_kaybi", "build_dataset"),
    ("scripts.senaryo_D5_kaynak_kaymasi", "teshis.veri.senaryo_d5_kaynak_alani_kaymasi", "build_dataset"),
    ("scripts.senaryo_D6a_split_sizintisi", "teshis.veri.senaryo_d6a_split_sizintisi", "build_dataset"),
    ("scripts.senaryo_D6b_tekrar_agirligi", "teshis.veri.senaryo_d6b_tekrar_agirligi", "build_dataset"),
    ("scripts.senaryo_E1_overfitting", "teshis.veri.senaryo_e1_overfitting", "build_dataset"),
    ("scripts.senaryo_E4_cozunurluk_uyumsuzlugu", "teshis.degerlendirme.senaryo_e4_cozunurluk_uyumsuzlugu", "tarama_raporu"),
])
def test_eski_giris_ayni_uygulamayi_cagirir(giris, modul, fonksiyon):
    assert getattr(importlib.import_module(giris), fonksiyon) is getattr(importlib.import_module(modul), fonksiyon)


def test_d2a_ayni_seed_ayni_gurultuyu_uretir():
    from teshis.veri.senaryo_d2a_lokalizasyon_gurultusu import noisy_label
    etiket = "1 0.5 0.5 0.2 0.2"
    a = noisy_label(etiket, random.Random(42))
    b = noisy_label(etiket, random.Random(42))
    assert a == b
    assert a[0] != etiket
    assert a[0].split()[0] == "1"


@pytest.mark.parametrize("oran,kalan", [(0.0, 2), (1.0, 0)])
def test_d2b_yalnizca_hedef_etiketleri_degistirir(tmp_path, oran, kalan):
    from teshis.veri.senaryo_d2b_eksik_etiket import build_dataset
    src, out = tmp_path / "src", tmp_path / "out"
    for alt in ("images/train", "labels/train", "images/val", "images/test"):
        (src / alt).mkdir(parents=True)
    (src / "images/train/a.jpg").write_bytes(b"image-placeholder")
    etiket = "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n"
    (src / "labels/train/a.txt").write_text(etiket, encoding="utf-8")
    build_dataset(src, out, oran, 42)
    assert len((out / "labels/train/a.txt").read_text().splitlines()) == kalan
    assert (src / "labels/train/a.txt").read_text() == etiket
    m = json.loads((out / "manifest.json").read_text())
    assert m["counts"]["removed_bbox_rows"] == 2 - kalan
    assert not m["val_test_modified"]


def test_e4_tasindiktan_sonra_proje_kokunu_bulur():
    from teshis.degerlendirme.senaryo_e4_cozunurluk_uyumsuzlugu import ROOT, rapor_yolu
    assert ROOT == Path(__file__).resolve().parents[1]
    assert (rapor_yolu(512) / "d1_metrics.json").is_file()

"""D serisi egitim protokolunun tek kaynaktan geldigini dogrulayan sozlesme testleri.

Bu testler, D1 kosusunun (lr0=0.0005, warmup_epochs=2) D2a/D2b/D3'ten
(lr0=0.001, warmup_epochs=3) sessizce farkli hiperparametrelerle
egitilmesine yol acan hatanin bir daha sessizce geri gelmemesini saglar:
her egitim scripti artik senaryolar/egitim_protokolu.yaml'i tek kaynak
olarak kullanmak ZORUNDADIR.
"""

from pathlib import Path

import pytest

from teshis.egitim.protokol import PROTOKOL_YOLU, egitim_kwargs

ROOT = Path(__file__).resolve().parents[1]

PROTOKOLU_KULLANMASI_GEREKEN_DOSYALAR = [
    ROOT / "teshis/egitim/kos.py",
    ROOT / "scripts/local_d2b.py",
    ROOT / "scripts/local_d3.py",
    ROOT / "scripts/kaggle_d2a.py",
    ROOT / "scripts/kaggle_d2b.py",
]


def test_protokol_dosyasi_mevcut():
    assert PROTOKOL_YOLU.is_file()


def test_egitim_kwargs_beklenen_alanlari_icerir():
    kwargs = egitim_kwargs()
    for alan in ("lr0", "lrf", "warmup_epochs", "cos_lr", "patience", "deterministic"):
        assert alan in kwargs


def test_egitim_kwargs_veri_bozma_disi_alan_icermez():
    """Bu fonksiyon yalnizca optimizasyon/augmentasyon protokolunu dondurmelidir;
    epochs/batch/device/imgsz gibi kosuya ozgu degerler burada olmamalidir."""
    kwargs = egitim_kwargs()
    for alan in ("epochs", "batch", "device", "imgsz", "seed", "data", "model"):
        assert alan not in kwargs


@pytest.mark.parametrize("dosya", PROTOKOLU_KULLANMASI_GEREKEN_DOSYALAR, ids=lambda yol: yol.name)
def test_script_egitim_kwargs_kullaniyor(dosya: Path):
    kaynak = dosya.read_text(encoding="utf-8")
    assert "egitim_kwargs(" in kaynak, f"{dosya.name} artik protokol.egitim_kwargs() kullanmali"


@pytest.mark.parametrize("dosya", PROTOKOLU_KULLANMASI_GEREKEN_DOSYALAR, ids=lambda yol: yol.name)
def test_script_lr0_hardcode_etmiyor(dosya: Path):
    kaynak = dosya.read_text(encoding="utf-8")
    assert "lr0=0.0005" not in kaynak, f"{dosya.name} eski D1 lr0 degerini hala hardcode ediyor"
    assert "lr0=0.001," not in kaynak.replace(" ", ""), f"{dosya.name} lr0'i hardcode ediyor, egitim_kwargs() kullanmali"

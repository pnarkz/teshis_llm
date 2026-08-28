"""kos.py::devam_et — yarida kalan egitimi surdurme davranisi.

Bu testler gercek bir olcumden dogdu: v00n kosusu epoch 24'te kesildi,
--devam ile surduruldu ve egitim BOZULDU (cls_loss 0.7477 -> 3.1210,
mAP50 0.8136 -> 0.2849, kismi toparlanmaya ragmen eski seviyeye donmedi).
Ozellik silinmedi ama artik bilincli onay istiyor; bu testler o korumanin
kaldirilmadigini garanti eder.
"""

from pathlib import Path

import pytest

from teshis.egitim import kos


def test_onaysiz_devam_reddedilir(tmp_path):
    """Onay verilmeden devam edilememeli; sessizce bozuk kosu uretilmemeli."""
    run_dir = tmp_path / "run_x"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights/last.pt").write_bytes(b"sahte")
    with pytest.raises(RuntimeError, match="egitimi bozabilir"):
        kos.devam_et(run_dir)


def test_checkpoint_yoksa_acik_hata(tmp_path):
    run_dir = tmp_path / "run_y"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Devam edilecek checkpoint yok"):
        kos.devam_et(run_dir, onayla=True)


def test_uyari_metni_olculen_degerleri_iceriyor():
    """Docstring, sorunu somut sayilarla belgelemelidir.

    Ileride biri bu ozelligi 'sanki calisiyor' diye kullanmasin diye
    olculen kanit kodun icinde durur.
    """
    metin = kos.devam_et.__doc__
    for parca in ("0.7477", "3.1210", "0.8136", "0.2849"):
        assert parca in metin, f"olculen deger docstring'de yok: {parca}"


def test_devam_onayla_bayragi_tanimli():
    """CLI'da onay bayragi bulunmali; aksi halde ozellik hic kullanilamaz."""
    kaynak = Path(kos.__file__).read_text(encoding="utf-8")
    assert "--devam-onayla" in kaynak

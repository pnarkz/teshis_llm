"""teshis/veri/istatistik.py::analiz_et icin sentetik mini-dataset testi.

Gercek dataset'e (C:/Users/ASUS/Desktop/HYZ/dataset) dokunmadan, gecici bir
klasorde YOLO klasor yapisini taklit ederek saglik kontrollerinin dogru
sayildigini test eder.
"""

from pathlib import Path

from PIL import Image

from teshis.veri.istatistik import analiz_et


def _yolo_dataset_kur(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Gecerli bir tasit (sinif 0) etiketi ve goruntusu.
    Image.new("RGB", (100, 100)).save(root / "images/train/aaterm__001.jpg")
    (root / "labels/train/aaterm__001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # Gecersiz sinif id (99) iceren ikinci bir etiket.
    Image.new("RGB", (100, 100)).save(root / "images/train/hituav__002.jpg")
    (root / "labels/train/hituav__002.txt").write_text("99 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    # Yetim etiket: karsilik gelen goruntusu yok.
    (root / "labels/train/hituav__orphan.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    # Yetim goruntu: karsilik gelen etiketi yok.
    Image.new("RGB", (100, 100)).save(root / "images/train/hituav__orphan_img.jpg")

    # val ve test bos kalabilir (klasorler zaten olusturuldu).


def test_analiz_et_temel_sayim(tmp_path: Path):
    _yolo_dataset_kur(tmp_path)
    rapor = analiz_et(tmp_path)

    assert rapor["toplam"]["goruntu"] == 2  # sadece etiketli+goruntulu ciftler sayilir
    assert rapor["toplam"]["bbox"] == 2
    assert rapor["toplam"]["sinif_bbox"]["tasit"] == 1
    assert rapor["saglik"]["gecersiz_sinif"] == 1
    assert rapor["splitler"]["train"]["yetim_goruntu_sayisi"] == 1  # etiketi olup goruntusu olmayan
    assert rapor["splitler"]["train"]["yetim_etiket_sayisi"] == 1  # goruntusu olup etiketi olmayan


def test_analiz_et_kaynak_gruplamasi(tmp_path: Path):
    _yolo_dataset_kur(tmp_path)
    rapor = analiz_et(tmp_path)
    kaynaklar = rapor["kaynak_toplam"]
    assert kaynaklar["aaterm"]["goruntu"] == 1
    assert kaynaklar["hituav"]["goruntu"] == 1


def test_analiz_et_cok_kucuk_kutu_sayilir(tmp_path: Path):
    for split in ("train", "val", "test"):
        (tmp_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 640)).save(tmp_path / "images/train/synth_001.jpg")
    (tmp_path / "labels/train/synth_001.txt").write_text("0 0.5 0.5 0.005 0.005\n", encoding="utf-8")

    rapor = analiz_et(tmp_path)
    assert rapor["saglik"]["cok_kucuk_kutu_sqrt_alan_6_alt"] == 1

"""teshis/degerlendirme/metrikler.py saf mantik fonksiyonlari icin testler.

Model/GPU gerektiren boyut_bazli_recall burada test edilmez; IoU, boyut
normalizasyonu, bant sinirlari ve eslestirme mantigi izole edilerek test edilir.
"""

import pytest

from teshis.degerlendirme.metrikler import (
    BANTLAR,
    boyut_bandi,
    eslestir,
    etkin_sqrt_alan,
    iou,
    yolo_to_xyxy,
)


def test_iou_tam_ortusme():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_ayrik_kutular():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_kismi_ortusme():
    # 10x10 ve 10x10, 5px kayik -> kesisim 5*10=50, birlesim 100+100-50=150
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_sifir_alanli_kutu_patlamaz():
    assert iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


def test_etkin_alan_cozunurlugu_normalize_eder():
    """Ayni fiziksel oranli nesne, farkli cozunurlukte ayni etkin boyutu vermelidir."""
    # 1024x1024 goruntude 0.05 normalize kenar -> 51.2 px, 640'a olceklenir -> 32
    buyuk = etkin_sqrt_alan(0.05, 0.05, 1024, 1024)
    # 640x640 goruntude ayni oran -> 32 px, olcek 1.0 -> 32
    kucuk = etkin_sqrt_alan(0.05, 0.05, 640, 640)
    assert buyuk == pytest.approx(kucuk)
    assert buyuk == pytest.approx(32.0)


def test_etkin_alan_gecersiz_boyutta_sifir():
    assert etkin_sqrt_alan(0.1, 0.1, 0, 0) == 0.0


def test_boyut_bandi_sinirlari():
    assert boyut_bandi(15.9) == "cok_kucuk_16_alti"
    assert boyut_bandi(16.0) == "kucuk_16_32"   # ust sinir haric
    assert boyut_bandi(31.9) == "kucuk_16_32"
    assert boyut_bandi(32.0) == "orta_32_64"
    assert boyut_bandi(64.0) == "buyuk_64_ustu"
    assert boyut_bandi(5000.0) == "buyuk_64_ustu"


def test_bant_esigi_d4_senaryosuyla_hizali():
    """Ilk bant siniri, D4'un etkin_sqrt_alan_esigi_px parametresiyle ayni olmalidir."""
    import yaml
    from pathlib import Path

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "senaryolar/veri/d4_kucuk_nesne_sinyal_kaybi.yaml")
        .read_text(encoding="utf-8")
    )
    assert BANTLAR[0][0] == float(config["parametreler"]["etkin_sqrt_alan_esigi_px"])


def test_yolo_to_xyxy():
    # merkez (0.5, 0.5), boyut 0.5x0.5, 100x100 goruntu -> (25,25,75,75)
    assert yolo_to_xyxy(0.5, 0.5, 0.5, 0.5, 100, 100) == pytest.approx((25.0, 25.0, 75.0, 75.0))


def _kutu(sinif, xyxy, conf=0.9):
    return {"sinif": sinif, "kutu": xyxy, "conf": conf}


def test_eslestirme_dogru_sinifta_eslesir():
    gercek = [{"sinif": 1, "kutu": (0, 0, 10, 10)}]
    tahmin = [_kutu(1, (0, 0, 10, 10))]
    assert eslestir(gercek, tahmin) == {0}


def test_eslestirme_yanlis_sinifi_eslestirmez():
    gercek = [{"sinif": 1, "kutu": (0, 0, 10, 10)}]
    tahmin = [_kutu(2, (0, 0, 10, 10))]
    assert eslestir(gercek, tahmin) == set()


def test_eslestirme_dusuk_iou_reddeder():
    gercek = [{"sinif": 0, "kutu": (0, 0, 10, 10)}]
    tahmin = [_kutu(0, (9, 9, 19, 19))]  # cok kucuk ortusme
    assert eslestir(gercek, tahmin, iou_esigi=0.5) == set()


def test_eslestirme_bir_gercek_kutuyu_iki_kez_saymaz():
    """Iki tahmin ayni gercek kutuyu gosteriyorsa yalnizca biri eslesir (digeri FP'dir)."""
    gercek = [{"sinif": 0, "kutu": (0, 0, 10, 10)}]
    tahmin = [_kutu(0, (0, 0, 10, 10), conf=0.9), _kutu(0, (0, 0, 10, 10), conf=0.8)]
    assert eslestir(gercek, tahmin) == {0}


def test_eslestirme_yuksek_guveni_once_isler():
    """Yuksek guvenli tahmin, ortak gercek kutuyu once almalidir."""
    gercek = [
        {"sinif": 0, "kutu": (0, 0, 10, 10)},
        {"sinif": 0, "kutu": (100, 100, 110, 110)},
    ]
    tahmin = [_kutu(0, (0, 0, 10, 10), conf=0.95), _kutu(0, (100, 100, 110, 110), conf=0.60)]
    assert eslestir(gercek, tahmin) == {0, 1}


def test_eslestirme_bos_girdilerde_patlamaz():
    assert eslestir([], []) == set()
    assert eslestir([], [_kutu(0, (0, 0, 5, 5))]) == set()
    assert eslestir([{"sinif": 0, "kutu": (0, 0, 5, 5)}], []) == set()

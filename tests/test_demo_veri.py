"""Demo veri katmaninin sozlesme testleri.

Sayfa render testleri tests/test_demo_konsol.py icine tasindi (konsol alti
bolumlu yeni yapiya gecti). Burada kalan iki test veri katmanina aittir ve
sayfa yapisindan bagimsizdir.
"""

from pathlib import Path

import pytest

pytest.importorskip("pandas")

APP = Path(__file__).resolve().parents[1] / "demo" / "app.py"


def test_her_kosunun_kaniti_konvansiyonla_bulunur():
    """Demo, rapor klasorunu senaryo adindan turetmeli; elle harita tutmamali.

    Onceki surumde sabit kodlu bir sozluk vardi ve her yeni senaryoda geride
    kaliyordu: D6a, D6b, v00n ve D1n eklendiginde demo onlari sessizce
    "kanit yok" gosteriyordu.
    """
    import sys

    sys.path.insert(0, str(APP.parent))
    from data_loader import evidence_for, images_for, load_results

    eksik = []
    for _, row in load_results().iterrows():
        senaryo = row["scenario"]
        if senaryo == "Baseline":
            continue  # baseline metrikleri model_secimi altindan gelir
        if not evidence_for(senaryo) or not images_for(senaryo):
            eksik.append(senaryo)
    assert not eksik, f"Bu senaryolarin kaniti demo'da bulunamiyor: {eksik}"


def test_demo_sabit_rapor_haritasi_tutmuyor():
    """Sabit kodlu yol sozlukleri geri gelmemeli."""
    kaynak = (APP.parent / "data_loader.py").read_text(encoding="utf-8")
    for yasak in ("EVIDENCE_JSON", "EVIDENCE_FOLDERS"):
        assert yasak not in kaynak, f"{yasak} sabit haritasi geri gelmis"


def test_senaryo_bilgisi_elle_tutulmuyor():
    """app.py'deki 24 girdilik scenario_info sozlugu geri gelmemeli.

    Senaryo ozeti artik kaynak dosyalardan turetilir
    (teshis/degerlendirme/senaryo_ozeti.py).
    """
    kaynak = APP.read_text(encoding="utf-8")
    assert "scenario_info" not in kaynak


def test_galeri_anahtarlari_senaryo_adlariyla_eslesir():
    """Galeriler klasor adiyla degil SENARYO adiyla anahtarlanmali.

    GERCEK HATA: klasor adindaki alt cizgi sokulmeden senaryo adi sayiliyordu;
    "C2 seed13" senaryosunun galerisi `C2_seed13` anahtariyla duruyordu ve
    defterdeki adla eslesmiyordu. Demonun galeri secicisinde klasor adlari
    goruntuleniyordu.
    """
    import sys

    sys.path.insert(0, str(APP.parent))
    from data_loader import error_galleries, load_results

    adlar = set(load_results()["scenario"])
    eslesmeyen = [k for k in error_galleries() if k not in adlar]
    assert not eslesmeyen, f"Senaryo adiyla eslesmeyen galeri anahtarlari: {eslesmeyen}"


def test_her_kosunun_hata_galerisi_var():
    """Kanit sozlesmesi hata ornekleri istiyor; demo da onlari gosterebilmeli."""
    import sys

    sys.path.insert(0, str(APP.parent))
    from data_loader import error_galleries, load_results

    galeriler = set(error_galleries())
    eksik = [s for s in load_results()["scenario"] if s not in galeriler and s != "Baseline"]
    assert not eksik, f"Hata galerisi olmayan kosular: {eksik}"

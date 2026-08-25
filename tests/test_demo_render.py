"""Demo konsolunun her sayfa/senaryo kombinasyonunda istisnasiz render edildigini dogrular.

Streamlit'in kendi AppTest cercevesini kullanir: uygulama headless calistirilir
ve widget degerleri programatik olarak ayarlanir. Boylece bir senaryo icin
eksik rapor dosyasi, yeniden adlandirilmis sutun veya bozuk bir veri yolu
sessizce sunum aninda degil, testte ortaya cikar.

Bu testler uygulamayi bastan calistirdigi icin diger testlerden yavastir
(~2 sn/kombinasyon). Yalnizca demo bagimliliklari kuruluysa calisir:
    python -m pip install -r requirements-demo.txt
"""

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1", reason="demo bagimliliklari kurulu degil (requirements-demo.txt)"
)
AppTest = streamlit_testing.AppTest

APP = Path(__file__).resolve().parents[1] / "demo" / "app.py"
PAGES = ["Genel Bakis", "Senaryo Incele", "Hata Galerisi", "Proje ve Senaryolar", "LLM Ajan"]
SCENARIOS = ["Baseline", "D1", "D2a", "D2b", "D2b final_best", "D3"]


def _hatalar(app) -> list[str]:
    return [str(getattr(item, "message", item)) for item in app.exception]


def _calistir(page: str):
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception, f"ilk yukleme basarisiz: {_hatalar(app)}"
    app.radio[0].set_value(page).run()
    return app


def test_uygulama_hatasiz_aciliyor():
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception, _hatalar(app)


@pytest.mark.parametrize("page", PAGES)
def test_her_sayfa_render_ediliyor(page: str):
    app = _calistir(page)
    assert not app.exception, f"{page}: {_hatalar(app)}"
    assert app.markdown, f"{page}: hic icerik uretilmedi"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_her_senaryo_render_ediliyor(scenario: str):
    app = _calistir("Senaryo Incele")
    assert not app.exception, _hatalar(app)
    app.selectbox[0].set_value(scenario).run()
    assert not app.exception, f"{scenario}: {_hatalar(app)}"
    assert app.markdown, f"{scenario}: hic icerik uretilmedi"


@pytest.mark.parametrize("order", ["Skor", "Yanlis negatif", "Yanlis pozitif", "Dusuk IoU"])
def test_hata_galerisi_siralamalari(order: str):
    app = _calistir("Hata Galerisi")
    assert not app.exception, _hatalar(app)
    if len(app.radio) < 2:
        pytest.skip("hata galerisi bulunamadi (reports/*_hata_galerisi yok)")
    app.radio[1].set_value(order).run()
    assert not app.exception, f"siralama={order}: {_hatalar(app)}"

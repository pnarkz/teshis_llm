"""Konsolun her bolumu ve her senaryosu istisnasiz render ediliyor mu?

Sunum sirasinda bir bolumun cokmesi en kotu senaryodur. Bu testler
Streamlit'in kendi AppTest cercevesiyle uygulamayi headless calistirir ve
her kombinasyonu dener.

Yavastir (~2 sn/kombinasyon) ve yalnizca demo bagimliliklari kuruluysa
calisir: python -m pip install -r requirements-demo.txt
"""

import csv
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1", reason="demo bagimliliklari kurulu degil"
)
AppTest = streamlit_testing.AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "demo" / "app.py"
BOLUMLER = [
    "Genel Bakış", "Deney Tasarımı ve Sınırlar", "Senaryolar",
    "Karşılaştırma ve Gürültü", "Hata Analizi", "Ajan",
]


def _senaryolar() -> list[str]:
    """Demonun acilir listesinde GERCEKTEN gorunen kosular.

    results.csv'yi okumak yetmiyordu: demo veri katmani ayrica bir "Baseline"
    satiri ekliyor (fine-tune edilmemis model). Listeyi demonun kendi
    kaynagindan almak, acilir listede secilebilen her secenegin test
    edilmesini garanti eder.
    """
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    from data_loader import load_results

    return load_results()["scenario"].tolist()


def _sorunlar(app) -> list[str]:
    """Hem yakalanmamis istisnalar hem app.py'nin gosterdigi bolum hatasi."""
    hatalar = [str(getattr(x, "message", x)) for x in app.exception]
    hatalar += [
        str(e.value) for e in app.error
        if "yüklenemedi" in str(e.value)
    ]
    return hatalar


def _bolum(ad: str):
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    assert not _sorunlar(app), f"ilk yukleme: {_sorunlar(app)}"
    app.radio[0].set_value(ad).run()
    return app


def test_uygulama_aciliyor():
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    assert not _sorunlar(app), _sorunlar(app)


@pytest.mark.parametrize("bolum", BOLUMLER)
def test_her_bolum_render_ediliyor(bolum: str):
    app = _bolum(bolum)
    assert not _sorunlar(app), f"{bolum}: {_sorunlar(app)}"
    assert app.markdown, f"{bolum}: hic icerik uretilmedi"


@pytest.mark.parametrize("senaryo", _senaryolar())
def test_her_senaryo_render_ediliyor(senaryo: str):
    """24 kosunun tamami; biri bile cokerse sunumda o kosu acilamaz."""
    app = _bolum("Senaryolar")
    app.selectbox[0].set_value(senaryo).run()
    assert not _sorunlar(app), f"{senaryo}: {_sorunlar(app)}"


def test_ajan_bolumu_kayitli_modda_api_gerektirmiyor(monkeypatch):
    """Kayitli mod anahtarsiz calismali; sunumun guvenli yolu budur."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = _bolum("Ajan")
    assert not _sorunlar(app), _sorunlar(app)


def test_bolumler_ayri_dosyalarda():
    """Tek dev app.py geri gelmemeli; bolum bazli duzenleme kolay kalmali."""
    assert (ROOT / "demo/bolumler").is_dir()
    moduller = sorted(p.stem for p in (ROOT / "demo/bolumler").glob("*.py")
                      if p.stem != "__init__")
    assert len(moduller) >= 6, moduller
    assert len((ROOT / "demo/app.py").read_text(encoding="utf-8").splitlines()) < 120, (
        "app.py yeniden buyumus; icerik bolum modullerinde durmali"
    )


def test_karsilastirma_tablolari_turetiliyor():
    """Karsilastirma sayfasindaki tablolar elle yazilmis sayilar tasimamali.

    Elle yazilan tablolar bu projede tekrar tekrar bayatladi. Erken durdurma
    ve checkpoint cifti tablolari artik defterden ve kosu dizinlerinden
    turetilir; yeni bir kontrol kosusu veya last_pt satiri eklendiginde
    kendiliginden buyurler.
    """
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    from bolumler.karsilastirma import _checkpoint_ciftleri, _erken_durdurma
    from data_loader import load_results

    sonuclar = load_results()

    erken = _erken_durdurma(sonuclar)
    assert len(erken) >= 3, "erken durdurma tablosu bos veya eksik"
    assert set(erken["seed"]) >= {7, 42}, erken["seed"].tolist()

    cift = _checkpoint_ciftleri(sonuclar)
    assert len(cift) >= 6, "checkpoint cifti tablosu eksik"
    adlar = set(cift["kosu"])
    assert "v00_saglikli best.pt" in adlar and "v00_saglikli last.pt" in adlar, (
        "Saglikli referansin checkpoint cifti tabloda yok; last.pt dususunun "
        "tabani gorunmezse her dusus bozulma sanilir"
    )


def test_checkpoint_notundaki_sayilar_tablodan_geliyor():
    """Tablonun altindaki yorum, tablonun kendi degerleriyle uyusmali."""
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    from bolumler.karsilastirma import _checkpoint_ciftleri
    from data_loader import load_results

    df = _checkpoint_ciftleri(load_results()).set_index("kosu")
    kaynak = (ROOT / "demo/bolumler/karsilastirma.py").read_text(encoding="utf-8")
    for kosu in ("v00_saglikli last.pt", "D5 last.pt", "E1 last.pt"):
        deger = df.loc[kosu, "Δ v00"]
        assert f"{deger:.4f}" in kaynak, (
            f"{kosu} icin yorumdaki sayi ({deger:.4f}) tabloyla uyusmuyor"
        )

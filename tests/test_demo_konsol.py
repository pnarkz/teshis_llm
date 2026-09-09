"""Konsolun her bolumu ve her senaryosu istisnasiz render ediliyor mu?

Sunum sirasinda bir bolumun cokmesi en kotu senaryodur. Bu testler
Streamlit'in kendi AppTest cercevesiyle uygulamayi headless calistirir ve
her kombinasyonu dener.

Yavastir (~2 sn/kombinasyon) ve yalnizca demo bagimliliklari kuruluysa
calisir: python -m pip install -r requirements-demo.txt
"""

import csv
import re
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1", reason="demo bagimliliklari kurulu degil"
)
AppTest = streamlit_testing.AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "demo" / "app.py"
def _bolum_adlari() -> list[str]:
    """Bolum listesi app.py'nin KENDI sozlugunden gelir.

    Liste burada elle yazilirken bir bolum eklendiginde test onu hic
    denemiyordu; adi degistiginde de sessizce eski adi ariyordu. Kaynak
    tektir: demo/app.py icindeki BOLUMLER.
    """
    import re

    kaynak = (ROOT / "demo/app.py").read_text(encoding="utf-8")
    govde = kaynak[kaynak.index("BOLUMLER = {"):kaynak.index("}", kaynak.index("BOLUMLER = {"))]
    return re.findall(r'"([^"]+)":', govde)


BOLUMLER = _bolum_adlari()


def _bolum_adi(parca: str) -> str:
    """Bolum adini PARCASINDAN bulur; ad degisince test kirilmasin."""
    return next(b for b in BOLUMLER if parca.lower() in b.lower())


def _senaryolar() -> list[str]:
    """Senaryo katalogundaki ARASTIRMA senaryolari.

    Onceden bu liste `results.csv`'den geliyordu ve 26 kosuyu kapsiyordu -
    ama senaryo ile kosu ayni sey degil. Katalog yalnizca arastirma sorusu
    tasiyan senaryolari icerir; referanslar, kontroller ve seed/checkpoint/
    cozunurluk varyantlari kosu defterine aittir.
    """
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    import katalog

    return [s["kod"] for s in katalog.senaryolar()]


def _kosular() -> list[str]:
    """Defterdeki her kosu - kosu defteri ve karsilastirma sayfasi icin."""
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    from teshis.degerlendirme.karsilastirilabilirlik import kimlik
    from data_loader import load_results

    return [str(a) for a in load_results()["scenario"] if kimlik(str(a))]


def test_baseline_senaryo_kartlarina_girmiyor():
    """Fine-tune edilmemis model bir DENEY degildir.

    Defterde satiri olmadigi icin olcegi, referansi ve gurultu esigi yoktur;
    senaryo listesinde gorunse "karsilastirilamaz" diye durur ve deney gibi
    okunurdu. Yeri "Veri ve Saglikli Model" sayfasidir.
    """
    assert "Baseline" not in _senaryolar()
    assert "Baseline" not in _kosular()
    kaynak = (ROOT / "demo/bolumler/veri_ve_model.py").read_text(encoding="utf-8")
    assert "Baseline" in kaynak, (
        "Baseline hicbir yerde gosterilmiyor; fine-tune etkisi kayboldu"
    )


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
    """Katalogdaki her senaryo; biri bile cokerse sunumda o senaryo acilamaz.

    Secim acilir liste degil DUGME: kart HTML'i tiklanabilir degildir ve
    gorsel olarak kart, davranis olarak dugme olan bir yapi kullaniciyi
    yaniltirdi.

    ONEMLI: secili senaryonun dugmesi DEVRE DISI olur. Ona tiklamak hicbir
    sey yapmaz, yani test o senaryo icin bos gecerdi. Bu durumda dugmeye
    tiklamak yerine BASKA bir senaryoya gecip geri donulur - boylece her
    senaryonun ayrinti ekrani gercekten render edilir.
    """
    app = _bolum(_bolum_adi("Senaryo"))
    dugme = next((d for d in app.button if d.key == f"sec_{senaryo}"), None)
    assert dugme is not None, f"{senaryo}: secim dugmesi yok"

    if getattr(dugme, "disabled", False):
        # Zaten secili: once baskasina gec, sonra bu senaryoya geri don.
        baska = next(d for d in app.button
                     if d.key != f"sec_{senaryo}" and not getattr(d, "disabled", False))
        baska.click().run()
        assert not _sorunlar(app), f"{senaryo} (ara adim): {_sorunlar(app)}"
        dugme = next(d for d in app.button if d.key == f"sec_{senaryo}")

    dugme.click().run()
    assert not _sorunlar(app), f"{senaryo}: {_sorunlar(app)}"
    # Ayrinti ekrani gercekten acildi mi?
    metin = " ".join(str(m.value) for m in app.markdown)
    assert senaryo in metin, f"{senaryo}: ayrinti ekrani acilmadi"


@pytest.mark.parametrize("kosu", _kosular())
def test_her_kosu_karsilastirmada_render_ediliyor(kosu: str):
    """Karsilastirma sayfasi defterdeki her bozulma kosusunu acabilmeli.

    Secim IKI ASAMALI: once senaryo, sonra o senaryonun ana kosusu veya
    varyanti. Tek listede 20 kosu gostermek D4 ile "D4 last_pt"yi ayni
    seviyedeymis gibi yan yana koyuyordu.
    """
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    import katalog
    from teshis.degerlendirme.karsilastirilabilirlik import bozulmasiz_mi

    if bozulmasiz_mi(kosu):
        pytest.skip("bozulmasiz kosu senaryo seciciye girmez")

    senaryo = next(
        (x for x in katalog.senaryolar()
         if kosu == x["ana_kosu"] or kosu in x["varyantlar"]), None)
    assert senaryo is not None, f"{kosu} hicbir senaryoya bagli degil"

    app = _bolum(_bolum_adi("Karşılaştırma"))
    sen_secici = next(x for x in app.selectbox
                      if "senaryo" in (x.label or "").lower())
    sen_secici.set_value(senaryo).run()
    assert not _sorunlar(app), f"{senaryo['kod']}: {_sorunlar(app)}"

    kosu_secici = next(x for x in app.selectbox if "Koşu" in (x.label or ""))
    kosu_secici.set_value(kosu).run()
    assert not _sorunlar(app), f"{kosu}: {_sorunlar(app)}"


def test_ajan_bolumu_kayitli_modda_api_gerektirmiyor(monkeypatch):
    """Kayitli mod anahtarsiz calismali; sunumun guvenli yolu budur."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = _bolum(_bolum_adi("Ajan"))
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
    adlar = set(cift["koşu"])
    assert "v00_saglikli best.pt" in adlar and "v00_saglikli last.pt" in adlar, (
        "Saglikli referansin checkpoint cifti tabloda yok; last.pt dususunun "
        "tabani gorunmezse her dusus bozulma sanilir"
    )


def test_checkpoint_notunda_elle_yazilmis_sayi_kalmadi():
    """Tablonun altindaki yorum, sayilari TABLODAN almalidir.

    Onceki surumde bu kutuda dort tane elle yazilmis fark vardi ve tablo
    kendi referansina gecince onunla celiskiye dustu (D4 icin -0.0329
    yaziyordu, dogrusu -0.0105 idi). Simdi kutu turetiliyor; test bunun
    geriye donmedigini korur.
    """
    kaynak = (ROOT / "demo/bolumler/karsilastirma.py").read_text(encoding="utf-8")
    kutu = kaynak[kaynak.index("checkpoint düşüşünün"):]
    kutu = kutu[: kutu.index("    )")]
    elle = re.findall(r"(?<![:.\d])[-−]?0\.\d{3,}", kutu)
    assert not elle, f"Kutuda elle yazilmis sayi kalmis: {elle}"


def test_checkpoint_dususu_kendi_referansina_gore():
    """Her checkpoint satiri KENDI ailesinin referansiyla karsilastirilmali.

    Onceden last.pt satirlari best.pt referansiyla tartiliyordu; bu yuzden
    saglikli referansin kendi last.pt'si bile "bozulmus" gorunuyordu.
    """
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    from bolumler.karsilastirma import _checkpoint_ciftleri
    from data_loader import load_results

    df = _checkpoint_ciftleri(load_results()).set_index("koşu")
    assert df.loc["v00_saglikli last.pt", "referansı"] == "v00_saglikli last_pt"
    assert df.loc["v00_saglikli best.pt", "referansı"] == "v00_saglikli"
    # Referansin kendisi her iki ailede de sifir fark gostermeli.
    assert abs(df.loc["v00_saglikli last.pt", "Δ kendi referansına"]) < 1e-9
    assert abs(df.loc["v00_saglikli best.pt", "Δ kendi referansına"]) < 1e-9

"""Gorsel dil: renk sozlesmesi ve grafik fabrikalari.

Renk bu konsolda dekoratif degil SEMANTIKTIR: ayni renk her sayfada ayni
seyi soyler. Sozlesme iki yerde yasarsa (bir sayfada referans gri, digerinde
mavi) okuyucu yaniltilir - bu projede tekrarlayan hata oruntusu tam olarak
budur.

Testler grafik uretiminin cokmedigini ve sozlesmenin korundugunu dogrular;
gorsel estetigi test edilmez.
"""

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

import grafik  # noqa: E402
import stil  # noqa: E402

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def test_palet_gecerli_renkler():
    for ad in ("ZEMIN", "YUZEY", "YUZEY_2", "CIZGI", "METIN", "METIN_SOLUK",
               "ADAY", "IKINCIL", "REFERANS", "GUCLU", "UYARI", "KRITIK", "NOTR"):
        assert HEX.match(getattr(stil, ad)), ad


def test_streamlit_temasi_stil_paletiyle_ayni():
    """config.toml ile stil.py ayrisirsa Streamlit'in kendi bilesenleri
    (secici, sekme, tablo) sayfadan kopuk gorunur."""
    metin = (ROOT / ".streamlit/config.toml").read_text(encoding="utf-8")
    assert f'backgroundColor = "{stil.ZEMIN}"' in metin
    assert f'secondaryBackgroundColor = "{stil.YUZEY}"' in metin
    assert f'textColor = "{stil.METIN}"' in metin
    assert f'primaryColor = "{stil.ADAY}"' in metin
    assert 'base = "dark"' in metin


def test_her_kanit_seviyesinin_adi_ve_rengi_var():
    from teshis.degerlendirme.senaryo_ozeti import kanit_gucu

    import csv

    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        seviyeler = {kanit_gucu(r["scenario"])["seviye"]
                     for r in csv.DictReader(f)}
    for seviye in seviyeler:
        assert seviye in stil.SEVIYE, f"'{seviye}' icin ad/renk tanimi yok"
        assert stil.seviye_adi(seviye)
        assert stil.seviye_aciklamasi(seviye)
        assert HEX.match(stil.seviye_rengi(seviye))


def test_derecelendirilen_seviyeler_sozlukte_var():
    for s in stil.DERECELENDIRILEN:
        assert s in stil.SEVIYE


def test_rozet_html_uretiyor():
    html = stil.rozet("deneme", "guclu")
    assert stil.GUCLU in html and "deneme" in html


def test_guven_rozeti_yuzde_gostermiyor():
    """Guven ajanin OZ-BILDIRIMIDIR; kalibre edilmis bir olasilik degil."""
    html = stil.guven_rozeti("yüksek")
    assert "%" not in html
    assert "öz-bildirim" in html


# --- Grafik fabrikalari -----------------------------------------------------

def test_etki_haritasi_ciziliyor():
    veri = pd.DataFrame([
        {"senaryo": "D4", "metrik": "mAP50", "band oranı": 3.7},
        {"senaryo": "D1", "metrik": "mAP50", "band oranı": 0.8},
    ])
    grafik.etki_haritasi(veri, "metrik", "senaryo", "band oranı").to_dict()


def test_etki_haritasi_renk_kirpiyor_ama_deger_korunuyor():
    """Kirpma yalnizca RENGI etkiler; gercek oran tooltip'te tam degeriyle durur."""
    veri = pd.DataFrame([
        {"senaryo": "D5", "metrik": "mAP50", "band oranı": 97.4},
    ])
    sozluk = grafik.etki_haritasi(veri, "metrik", "senaryo", "band oranı").to_dict()
    olcek = sozluk["encoding"]["color"]["scale"]["domain"]
    assert olcek[1] == grafik.HARITA_RENK_TAVANI
    # Ham deger veride bozulmadan kalir
    assert veri["band oranı"].iloc[0] == 97.4
    ipuclari = [t["field"] for t in sozluk["encoding"]["tooltip"]]
    assert "band oranı" in ipuclari and "renk" not in ipuclari


def test_gurultu_bandi_hem_renk_hem_sekil_kullaniyor():
    """Erisilebilirlik: ayrim yalnizca renge birakilmaz."""
    veri = pd.DataFrame([
        {"senaryo": "D4", "fark": -0.03, "band_alt": -0.006,
         "band_ust": 0.006, "asiyor": "evet"},
        {"senaryo": "D1", "fark": 0.002, "band_alt": -0.006,
         "band_ust": 0.006, "asiyor": "hayir"},
    ])
    sozluk = grafik.gurultu_bandi_grafigi(veri).to_dict()
    nokta = sozluk["layer"][-1]["encoding"]
    assert "color" in nokta and "shape" in nokta


def test_gruplu_barda_referans_serisi_notr_renkte():
    """Referans rengi BUTUN sayfalarda ayni kalmali."""
    veri = pd.DataFrame([
        {"metrik": "mAP50", "seri": "v00_saglikli (referans)", "değer": 0.92},
        {"metrik": "mAP50", "seri": "D4", "değer": 0.89},
    ])
    sozluk = grafik.gruplu_bar(veri, "metrik", "değer", "seri").to_dict()
    aralik = sozluk["encoding"]["color"]["scale"]["range"]
    assert aralik[0] == stil.REFERANS
    assert aralik[1] == stil.ADAY


def test_yatay_bar_ve_cizgi_ciziliyor():
    veri = pd.DataFrame([{"sınıf": "insan", "bbox": 2718},
                         {"sınıf": "taşıt", "bbox": 1264}])
    grafik.yatay_bar(veri, "sınıf", "bbox").to_dict()
    egri = pd.DataFrame([{"epoch": 1, "kayıp": 1.2, "seri": "train"},
                         {"epoch": 2, "kayıp": 0.9, "seri": "train"}])
    grafik.cizgi(egri, "epoch", "kayıp", seri="seri").to_dict()


def test_fark_bari_esigi_asani_vurguluyor():
    veri = pd.DataFrame([
        {"metrik": "mAP50", "mutlak fark": -0.03, "gürültü eşiği": 0.006},
    ])
    sozluk = grafik.fark_bar(veri, "metrik", "mutlak fark",
                             esik_alani="gürültü eşiği").to_dict()
    assert "condition" in sozluk["encoding"]["color"]


def test_grafiklerde_hover_her_zaman_var():
    """Sunumda kesin deger sorusu gelirse tabloya gitmek gerekmemeli."""
    veri = pd.DataFrame([{"a": "x", "b": 1.0, "c": "seri"}])
    for cagri in (
        lambda: grafik.yatay_bar(veri, "a", "b"),
        lambda: grafik.gruplu_bar(veri, "a", "b", "c"),
        lambda: grafik.isi_haritasi(veri, "a", "c", "b"),
    ):
        sozluk = cagri().to_dict()
        assert sozluk["encoding"].get("tooltip"), sozluk["encoding"].keys()


def test_bolumler_kendi_grafik_temasini_yazmiyor():
    """Grafik temasi TEK yerde: grafik.py.

    Bolumler dogrudan altair'e configure/mark cagrisi yaparsa tema ayrisir.
    """
    for yol in sorted((ROOT / "demo/bolumler").glob("*.py")):
        kaynak = yol.read_text(encoding="utf-8")
        assert "import altair" not in kaynak, yol.name
        assert ".configure(" not in kaynak, yol.name

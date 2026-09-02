"""Alt grup gurultu bandi: ajanin yanlis pozitifinin kok nedenini kapatir.

GERCEK HATA: ajan, hicbir bozulma icermeyen bir kontrol kosusunda "kaynak_d
grubunda belirgin recall kaybi" teshisi koydu ve guveni "yuksek" idi.
Rakamlar dogruydu (-0.1321) ve iki ayri istatistik testi de anlamli
buluyordu; ama o grubun saglikli kosular arasindaki yayilimi zaten 0.1321'di.

Ajan artik farki tek basina degil, o grubun bandiyla birlikte goruyor.
"""

import pytest

from teshis.ajan import araclar
from teshis.degerlendirme import gurultu


@pytest.fixture(scope="module")
def band():
    return gurultu.alt_grup_bandi()


def test_band_en_az_iki_saglikli_kosudan_hesaplanir(band):
    """Tek gozlemden band uydurmak, kapatmaya calistigi hatanin aynisi olurdu."""
    for alan, gruplar in band.items():
        for grup, d in gruplar.items():
            assert d["n_kosu"] >= 2, f"{alan}/{grup}: band {d['n_kosu']} kosudan"


def test_buyuk_grup_kucuk_band_kucuk_grup_buyuk_band(band):
    """En kalabalik kaynak en dar banda sahip olmali (kaba tutarlilik)."""
    kaynak = band["kaynak_recall"]
    if len(kaynak) < 2:
        pytest.skip("yeterli kaynak grubu yok")
    en_kalabalik = max(kaynak, key=lambda g: kaynak[g]["bbox_n"] or 0)
    assert kaynak[en_kalabalik]["band"] < 0.05, (
        f"{en_kalabalik} en kalabalik grup ama bandi {kaynak[en_kalabalik]['band']}"
    )


def test_band_yalnizca_orneklem_buyuklugune_bagli_degil(band):
    """`termal` 858 bbox'a ragmen genis bandli; bu bir kucuk-n sorunu degildir.

    Bu test bir OLGUYU kayda geciriyor: band, grup buyuklugunden bagimsiz
    olarak genis olabilir, dolayisiyla "bu grup buyuk, farki guvenilirdir"
    cikarimi yanlistir.
    """
    kaynak = band["kaynak_recall"]
    genis = [g for g, d in kaynak.items() if d["band"] > 0.05 and (d["bbox_n"] or 0) > 500]
    assert genis, (
        "Buyuk ama oynak bir kaynak grubu bulunamadi; bu testin dayandigi olgu "
        "degismis olabilir, docs/BULGULAR.md 'Ilk Yanlis Pozitif' gozden gecirilmeli"
    )


def test_gurultu_icindeki_fark_boyle_etiketlenir():
    """Bandin altinda kalan fark, acikca 'gurultu icinde' denmeli."""
    d = gurultu.fark_degerlendir("kaynak_recall", "hituav", 0.005)
    assert d["band_orani"] < 1
    assert "GURULTU ICINDE" in d["yorum"]


def test_az_gozlemde_uyari_verilir():
    """Band birkac kosudan geliyorsa, bandi asmak tek basina yeterli sayilmamali."""
    d = gurultu.fark_degerlendir("kaynak_recall", "tf2026", -0.20)
    if d["band_kosu_sayisi"] >= 5:
        pytest.skip("artik yeterli kontrol kosusu var")
    assert "ANCAK" in d["yorum"] or "belirgin" in d["yorum"]


def test_cok_buyuk_oranda_az_gozlem_uyarisi_kalkar():
    """Bandin 5 katini asan fark, band tahmininin kararsizligiyla aciklanamaz."""
    d = gurultu.fark_degerlendir("kaynak_recall", "hituav", -0.50)
    assert d["band_orani"] > 5
    assert "ANCAK" not in d["yorum"]


def test_kontrol_kosusu_kendi_bandini_tanimlamaz():
    """Degerlendirilen saglikli kosu banda dahil olursa olcut anlamini yitirir.

    Dahil olsaydi farki her zaman bandin tam sinirinda (oran 1.00) gorunurdu.
    """
    tam = gurultu.alt_grup_bandi()["kaynak_recall"]["tf2026"]
    haric = gurultu.alt_grup_bandi("c2_seed13_20260902")["kaynak_recall"]["tf2026"]
    assert haric["n_kosu"] == tam["n_kosu"] - 1
    assert haric["band"] != tam["band"]


def test_ajan_araclari_band_alanlarini_donduruyor():
    """Uc kirilim alani da bandla birlikte gelmeli; ham fark tek basina yaniltir."""
    for arac in (araclar.kaynak_bazli_recall_getir, araclar.boyut_bazli_recall_getir):
        cikti = arac("kosu_08")
        gruplar = cikti.get("kaynaklar") or cikti.get("bantlar")
        assert gruplar, f"{arac.__name__}: grup bulunamadi"
        for ad, d in gruplar.items():
            assert "gurultu_bandi" in d, f"{arac.__name__}/{ad}: band yok"
            assert "band_orani" in d
            assert "band_yorumu" in d


def test_gercek_bozulma_ile_gurultu_ayirt_ediliyor():
    """D4'un gercek etkisi ile kontrol kosusunun gurultusu farkli etiketlenmeli.

    Ajan uc ayri kosuda kaynak_d'yi suclamisti; band orani onu dogru gruba
    yonlendirmeli.
    """
    d4 = araclar.kaynak_bazli_recall_getir("kosu_08")["kaynaklar"]
    assert d4["kaynak_a"]["band_orani"] > 5, "D4'un gercek etkisi one cikmiyor"
    assert d4["kaynak_d"]["band_orani"] < 1, (
        "kaynak_d D4'te hala one cikiyor; ajanin sistematik yanliligi kapanmamis"
    )

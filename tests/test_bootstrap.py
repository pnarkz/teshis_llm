"""teshis/degerlendirme/bootstrap.py — guven araligi ve anlamlilik testleri.

Bu modul bir tekrarlanabilirlik bosluğunu kapatti: README'deki z degerleri ve
guven araliklari tek seferlik scriptlerde hesaplanmisti, repoda karsiligi
yoktu. Buradaki testler hem matematigi hem de README sayilarinin yeniden
uretilebildigini dogrular.
"""

import json
import math
from pathlib import Path

import pytest

from teshis.degerlendirme import bootstrap as bs

ROOT = Path(__file__).resolve().parents[1]


# --- Wilson araligi --------------------------------------------------------

def test_wilson_bilinen_deger():
    """Wilson araligi bilinen bir referans degerle uyusmali (11/17)."""
    alt, ust = bs.wilson_araligi(11, 17)
    assert alt == pytest.approx(0.413, abs=0.002)
    assert ust == pytest.approx(0.827, abs=0.002)


def test_wilson_uc_degerlerde_sifir_genislik_vermez():
    """Wald yaklasimi 0/n ve n/n'de sifir genislik verir; Wilson vermemeli."""
    for k, n in ((0, 15), (15, 15)):
        alt, ust = bs.wilson_araligi(k, n)
        assert ust - alt > 0.1, f"{k}/{n} araligi cok dar: [{alt}, {ust}]"


def test_wilson_arali_disina_tasmaz():
    for k, n in ((0, 5), (5, 5), (1, 3), (17, 17)):
        alt, ust = bs.wilson_araligi(k, n)
        assert 0.0 <= alt <= ust <= 1.0


def test_wilson_buyuk_n_daha_dar():
    """Ornek buyudukce aralik daralmali; projenin nadir sinif uyarisinin temeli."""
    dar = bs.wilson_araligi(1359, 2718)   # ~0.5, n=2718
    genis = bs.wilson_araligi(8, 17)      # ~0.5, n=17
    assert (dar[1] - dar[0]) < (genis[1] - genis[0]) / 5


def test_wilson_gecersiz_girdi():
    with pytest.raises(ValueError):
        bs.wilson_araligi(5, 0)
    with pytest.raises(ValueError):
        bs.wilson_araligi(6, 5)


# --- Iki oran testi --------------------------------------------------------

def test_iki_oran_ayni_oranlarda_z_sifir():
    sonuc = bs.iki_oran_testi(50, 100, 50, 100)
    assert sonuc["z"] == pytest.approx(0.0)
    assert not sonuc["anlamli_005"]


def test_iki_oran_buyuk_fark_anlamli():
    sonuc = bs.iki_oran_testi(90, 100, 50, 100)
    assert sonuc["z"] > 1.96
    assert sonuc["anlamli_005"]
    assert sonuc["fark"] == pytest.approx(0.4)


def test_iki_oran_p_degeri_makul():
    """z=1.96 civarinda p ~ 0.05 olmali."""
    sonuc = bs.iki_oran_testi(62, 100, 46, 100)
    assert 0.01 < sonuc["p"] < 0.10


def test_iki_oran_yon_isareti():
    assert bs.iki_oran_testi(30, 100, 60, 100)["z"] < 0
    assert bs.iki_oran_testi(60, 100, 30, 100)["z"] > 0


# --- Goruntu birimli bootstrap ---------------------------------------------

def test_bootstrap_nokta_tahmini_dogru():
    kayitlar = [(3, 5), (4, 5), (5, 5)]      # 12/15
    sonuc = bs.goruntu_bootstrap(kayitlar, tekrar=500)
    assert sonuc["oran"] == pytest.approx(12 / 15)
    assert sonuc["kutu_sayisi"] == 15
    assert sonuc["goruntu_sayisi"] == 3


def test_bootstrap_araligi_noktayi_icerir():
    kayitlar = [(2, 4)] * 30
    sonuc = bs.goruntu_bootstrap(kayitlar, tekrar=1000)
    assert sonuc["alt"] <= sonuc["oran"] <= sonuc["ust"]


def test_bootstrap_deterministik():
    kayitlar = [(1, 3), (2, 4), (3, 3), (0, 2)]
    a = bs.goruntu_bootstrap(kayitlar, tekrar=300, seed=7)
    b = bs.goruntu_bootstrap(kayitlar, tekrar=300, seed=7)
    assert a == b


def test_bootstrap_goruntu_birimli_kutu_biriminden_genis():
    """Ayni goruntudeki kutular korele; goruntu birimli aralik daha genis olmali.

    Asiri korele bir kurgu: her goruntu ya tamamen dogru ya tamamen yanlis.
    Kutu birimli bir hesap bunu bagimsiz sayar ve araligi oldugundan dar
    gosterir; bootstrap bu tuzagi yakalamalidir.
    """
    kayitlar = [(10, 10)] * 10 + [(0, 10)] * 10      # 100/200, tam korele
    sonuc = bs.goruntu_bootstrap(kayitlar, tekrar=2000)
    wilson_alt, wilson_ust = bs.wilson_araligi(100, 200)
    assert sonuc["genislik"] > (wilson_ust - wilson_alt) * 2


def test_bootstrap_bos_girdi_reddedilir():
    with pytest.raises(ValueError):
        bs.goruntu_bootstrap([])
    with pytest.raises(ValueError):
        bs.goruntu_bootstrap([(1, 2)], tekrar=0)


# --- README sayilarinin yeniden uretilebilirligi ----------------------------

D6B = ROOT / "reports/senaryo_D6b_last_pt/d1_metrics.json"
V00 = ROOT / "reports/referans_v00/d1_metrics.json"


@pytest.mark.skipif(not (D6B.is_file() and V00.is_file()), reason="D6b/v00 metrikleri yok")
def test_readme_d6b_sayilari_yeniden_uretilir():
    """README'nin D6b tablosundaki farklar ve z degerleri koddan cikmali."""
    satirlar = {s["grup"]: s for s in bs.sinif_metrigi_karsilastir(D6B, V00, "class_recall")}
    beklenen = {          # README "D6b Sonucu" tablosu
        "tasit": (+0.0356, +2.52),
        "insan": (+0.0224, +1.91),
        "UAP": (0.0000, 0.00),
        "UAI": (-0.6123, -3.59),
    }
    for sinif, (fark, z) in beklenen.items():
        assert satirlar[sinif]["fark"] == pytest.approx(fark, abs=0.0002), sinif
        assert satirlar[sinif]["z"] == pytest.approx(z, abs=0.02), sinif


@pytest.mark.skipif(not (D6B.is_file() and V00.is_file()), reason="metrikler yok")
def test_bbox_sayilari_kilitli_setle_tutarli():
    """Modulun kullandigi bbox sayilari, projenin sabitleriyle ayni olmali."""
    from teshis.ajan import araclar

    assert bs.VAL_DIAGNOSTIC_BBOX_N == araclar.VAL_DIAGNOSTIC_BBOX_N


# --------------------------------------------------------------------------
# C3 protokolu: tabakali, goruntu birimli bootstrap (sartname bolum 8)
# --------------------------------------------------------------------------


def _kayitlar(desen: list[tuple[str, int, int]]) -> list[dict]:
    """(kaynak, yakalanan, toplam) uclulerinden goruntu kaydi listesi uretir."""
    return [
        {"goruntu": f"{i}.jpg", "kaynak": kaynak,
         "sinif": {"insan": [y, t]}, "bant": {"orta": [y, t]}}
        for i, (kaynak, y, t) in enumerate(desen)
    ]


def test_tabakali_bootstrap_nokta_tahmini_dogru():
    kayitlar = _kayitlar([("a", 3, 4), ("a", 1, 2), ("b", 2, 2)])
    sonuc = bs.tabakali_goruntu_bootstrap(kayitlar, grup="insan", tekrar=200)
    assert sonuc["oran"] == pytest.approx(6 / 8)
    assert sonuc["kutu_sayisi"] == 8
    assert sonuc["goruntu_sayisi"] == 3
    assert sonuc["kaynak_sayisi"] == 2
    assert sonuc["alt"] <= sonuc["oran"] <= sonuc["ust"]


def test_tabakali_bootstrap_her_kaynagin_payini_korur():
    """Tabakalama, bir kaynagin payinin turler arasi dalgalanmasini engeller.

    Kaynak b tek goruntudur ve recall'u 0'dir. Tabakali ornekleme her turda
    tam olarak bir b goruntusu secer, yani b'nin payi sabittir. Tabakasiz
    ornekleme b'yi hic secmeyebilir veya birkac kez secebilirdi; bu, araliga
    gercekte var olmayan bir degiskenlik katardi.
    """
    kayitlar = _kayitlar([("a", 1, 1)] * 20 + [("b", 0, 1)])
    sonuc = bs.tabakali_goruntu_bootstrap(kayitlar, grup="insan", tekrar=500)
    # a'nin tamami 1/1, b'nin tamami 0/1 -> her turda tam olarak 20/21 cikar
    assert sonuc["alt"] == pytest.approx(20 / 21)
    assert sonuc["ust"] == pytest.approx(20 / 21)
    assert sonuc["genislik"] == pytest.approx(0.0)


def test_kumelenmis_kutularda_bootstrap_wilsondan_genis():
    """Sartnamenin asil iddiasi: kutu birimli aralik YAPAY OLARAK dardir.

    Kutular kareler icinde kumelenmisse (bir kare ya tamamen bulunur ya
    tamamen kacar) gercek belirsizlik kare sayisina baglidir, kutu sayisina
    degil. Wilson kutulari bagimsiz saydigi icin araligi daraltir.

    Burada 20 kare var; her karede 10 kutu ve kare ya hep vurus ya hep kacis.
    Wilson 200 bagimsiz kutu gorur, bootstrap 20 kare gorur.
    """
    desen = [("a", 10, 10)] * 10 + [("a", 0, 10)] * 10
    karsilastirma = bs.yontem_karsilastir(
        _kayitlar(desen), grup="insan", tekrar=2000
    )
    assert karsilastirma["oran"] == pytest.approx(0.5)
    assert karsilastirma["genislik_orani"] > 2.0, (
        "kumelenmis kutularda bootstrap araligi Wilson'dan belirgin genis olmali; "
        f"olculen oran {karsilastirma['genislik_orani']:.2f}"
    )


def test_kume_yoksa_iki_yontem_yakinsar():
    """Her karede tek kutu varsa korelasyon yoktur; iki yontem benzer olmali."""
    desen = [("a", i % 2, 1) for i in range(200)]
    karsilastirma = bs.yontem_karsilastir(
        _kayitlar(desen), grup="insan", tekrar=2000
    )
    assert 0.7 < karsilastirma["genislik_orani"] < 1.4, karsilastirma["genislik_orani"]


def test_ayni_seed_ayni_sonuc():
    kayitlar = _kayitlar([("a", 3, 5), ("b", 2, 4), ("b", 1, 3)])
    ilk = bs.tabakali_goruntu_bootstrap(kayitlar, grup="insan", tekrar=300, seed=7)
    ikinci = bs.tabakali_goruntu_bootstrap(kayitlar, grup="insan", tekrar=300, seed=7)
    assert ilk == ikinci


def test_kutusu_olmayan_grup_hata_verir():
    """Sessizce 0 dondurmek yerine hata: tanimsiz bir aralik rapora girmemeli."""
    with pytest.raises(ValueError, match="kutu yok"):
        bs.tabakali_goruntu_bootstrap(
            _kayitlar([("a", 1, 2)]), grup="UAP", tekrar=50
        )


def test_grup_verilmezse_tum_kutular_birlestirilir():
    kayitlar = [
        {"goruntu": "1.jpg", "kaynak": "a",
         "sinif": {"insan": [1, 2], "tasit": [3, 3]}, "bant": {}},
    ]
    sonuc = bs.tabakali_goruntu_bootstrap(kayitlar, tekrar=100)
    assert sonuc["kutu_sayisi"] == 5
    assert sonuc["oran"] == pytest.approx(4 / 5)

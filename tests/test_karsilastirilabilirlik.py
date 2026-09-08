"""Karsilastirma kurali: bir fark ancak AYNI olcekte anlamlidir.

GERCEK HATA (bu modul bu yuzden var)
------------------------------------
Demo butun kosulari `v00_saglikli` ile karsilastiriyordu. Sonuc:

    v00_saglikli last_pt  ->  "guclu" (mAP50, precision, recall esigi asiyor)
    v00n                  ->  "guclu" (dort metrikte de asiyor)

Ikisi de HICBIR bozulma icermez. Birincinin tek farki checkpoint secimi,
ikincinin tek farki baslangic modeli. Ayni filtre ajan araclarinda zaten
vardi; demo katmaninda yoktu.

Buradaki testler tek bir kurali korur: **bozulmasiz bir kosu asla bozulma
kaniti olarak derecelendirilmez.** Bu, projenin butun metodolojisinin
dayandigi cumledir; kirilirsa sunumdaki her iddia dayanaksiz kalir.
"""

import csv

import pytest

from teshis.degerlendirme import karsilastirilabilirlik as ka
from teshis.degerlendirme.senaryo_ozeti import kanit_gucu, ne_gozlendi

DERECELENDIRME = ("guclu", "zayif", "gurultu icinde")


@pytest.fixture(scope="module")
def senaryolar():
    with ka.RESULTS_CSV.open(encoding="utf-8") as f:
        return [r["scenario"] for r in csv.DictReader(f)]


# --- Asil kural -------------------------------------------------------------

def test_bozulmasiz_kosu_asla_bozulma_kaniti_sayilmaz(senaryolar):
    for s in senaryolar:
        if not ka.bozulmasiz_mi(s):
            continue
        seviye = kanit_gucu(s)["seviye"]
        assert seviye not in DERECELENDIRME, (
            f"{s} hicbir bozulma icermiyor ama '{seviye}' olarak "
            "derecelendirildi - projenin olcmeye calistigi hata tam olarak bu"
        )


def test_v00_last_pt_regresyonu():
    """Somut vaka: eskiden 'guclu', simdi kendi olceginin referansi."""
    assert kanit_gucu("v00_saglikli last_pt")["seviye"] == "referans"
    assert ne_gozlendi("v00_saglikli last_pt")["referans_senaryo"] == (
        "v00_saglikli last_pt"
    )


def test_yolo26n_regresyonu():
    """v00n saglikli bir yolo26n referansiydi, 'guclu bozulma' gorunuyordu."""
    assert kanit_gucu("v00n")["seviye"] == "referans"
    # D1n ise gercek bir bozulmadir ve referansi v00n olmalidir - v00 degil.
    assert ne_gozlendi("D1n")["referans_senaryo"] == "v00n"


# --- Kimlik alanlari --------------------------------------------------------

def test_referans_dort_kimlik_alaninda_da_ayni(senaryolar):
    for s in senaryolar:
        ref = ka.referans_senaryo(s)
        if ref is None:
            continue
        assert ka.kimlik(s) == ka.kimlik(ref), (
            f"{s} -> {ref}: kimlik alanlari ayni degil, fark bozulmanin "
            "degil o alanin etkisini tasir"
        )


def test_esik_yalnizca_kendi_olceginden_gelir(senaryolar):
    """Baska bir olcegin esigi ODUNC ALINAMAZ."""
    for s in senaryolar:
        kontroller = ka.kontrol_kosulari(s)
        for c in kontroller:
            assert ka.kimlik(c) == ka.kimlik(s), f"{s} esigini {c}'den aliyor"
        if not kontroller:
            assert all(v is None for v in ka.esikler(s).values()), (
                f"{s}: kontrol kosusu yok ama esik hesaplanmis"
            )


def test_kosu_kendi_esigini_tanimlamaz(senaryolar):
    """Kontrol kosusu kendi bandina dahil edilirse olcut anlamini yitirir."""
    for s in senaryolar:
        assert s not in ka.kontrol_kosulari(s), s


# --- Derecelendirilmeyen durumlar ------------------------------------------

def test_referanssiz_kosu_derecelendirilmez():
    """D2b final_best'in olceginde saglikli kosu yok."""
    guc = kanit_gucu("D2b final_best")
    assert guc["seviye"] == "karsilastirilamaz"
    # Aciklama, eksik olcumu ADIYLA soylemeli - "karsilastirilamaz" demek
    # tek basina bir sonraki adimi gostermez.
    assert "final_best.pt" in guc["aciklama"]


def test_eslenik_olcum_ayri_isaretlenir():
    """Ayni agirliklar, farkli cikarim ayari: egitim gurultusu devrede degil."""
    for s in ("E4 imgsz512", "D6a"):
        gozlem = ne_gozlendi(s)
        assert gozlem["karsilastirma_turu"] == "eslenik", s
        assert gozlem["referans_senaryo"] == "v00_saglikli", s
        assert kanit_gucu(s)["seviye"] == "eslenik olcum", s


def test_last_pt_ailesinde_esik_yok():
    """Saglikli last.pt kontrolu henuz olculmedi; bu gizlenmemeli."""
    for s in ("D4 last_pt", "D5 last_pt", "D6b last_pt", "E1 last_pt"):
        assert kanit_gucu(s)["seviye"] == "esik yok", s
        assert ne_gozlendi(s)["referans_senaryo"] == "v00_saglikli last_pt", s


def test_gercek_bozulmalar_hala_derecelendiriliyor():
    """Kural bir kalkan degil; asil bulgular ayakta kalmali."""
    for s in ("D2a", "D3", "D4", "D5"):
        assert kanit_gucu(s)["seviye"] in DERECELENDIRME, s
    assert kanit_gucu("D6b")["seviye"] == "gurultu icinde"


# --- Kural demonun her widget'inda gecerli olmali --------------------------

def test_etki_haritasinda_bozulmasiz_kosu_yok():
    """Genel Bakis'taki isi haritasi da ayni kurala tabidir.

    GERCEK HATA: kural once yalnizca "kanit gucu" etiketine uygulandi. Etki
    haritasi ayri bir yol izliyordu ve `v00_saglikli` ile uc kontrol kosusu
    haritada renkli hucreler olarak duruyordu - yani saglikli kosular
    "etkisi olan" senaryolar gibi gorunuyordu. Kural bir yerde uygulanip
    digerinde unutulursa isi bitmis olmuyor.
    """
    import sys

    sys.path.insert(0, str(ka.KOK / "demo"))
    from bolumler.genel_bakis import _etki_verisi
    from data_loader import load_results

    haritadakiler = set(_etki_verisi(load_results())["senaryo"])
    sizan = {s for s in haritadakiler if ka.bozulmasiz_mi(s)}
    assert not sizan, f"Bozulmasiz kosular etki haritasinda: {sorted(sizan)}"

    derecelendirilen = {
        s for s in haritadakiler if kanit_gucu(s)["seviye"] in DERECELENDIRME
    }
    assert haritadakiler == derecelendirilen, (
        "Etki haritasi derecelendirilemeyen kosulari da gosteriyor: "
        f"{sorted(haritadakiler - derecelendirilen)}"
    )


def test_gurultu_bandi_grafiginde_bozulmasiz_kosu_yok():
    import sys

    sys.path.insert(0, str(ka.KOK / "demo"))
    from bolumler.karsilastirma import _band_verisi
    from data_loader import load_results

    for metrik in ("mAP50", "precision", "recall"):
        veri = _band_verisi(load_results(), metrik)
        sizan = {s for s in veri["senaryo"] if ka.bozulmasiz_mi(s)}
        assert not sizan, f"{metrik}: bozulmasiz kosular grafikte: {sorted(sizan)}"

"""Senaryo ile kosu ayrimi: konsolun bilgi mimarisi.

GERCEK SORUN (bu modul bu yuzden var)
--------------------------------------
Senaryolar sayfasi 26 kosuyu esit agirlikta kart olarak gosteriyordu:

    D4          <- arastirilan hipotez
    D4 last_pt  <- ayni hipotezin baska checkpoint kaydi
    v00_saglikli<- karsilastirma tabani
    C2 seed21   <- dogal oynakligi olcen kontrol
    E4 imgsz512 <- cozunurluk varyanti

Bunlar ayni seviyede kavramlar degil. Hepsini esit karta donusturmek, veri
modelindeki her satiri arayuze siziyordu ve kullanici ilk ekranda D4 ile
`C2 seed21`'i ayni onemde goruyordu.

Buradaki testler ayrimin korunmasini saglar.
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

import katalog  # noqa: E402


@pytest.fixture(scope="module")
def senaryolar() -> list[dict]:
    return katalog.senaryolar()


# --- Ayrim ------------------------------------------------------------------

def test_katalog_yalnizca_arastirma_senaryolari(senaryolar):
    """Referans, kontrol ve varyantlar katalogda YER ALMAZ."""
    from teshis.degerlendirme.karsilastirilabilirlik import bozulmasiz_mi

    kodlar = {s["kod"] for s in senaryolar}
    for kod in kodlar:
        assert not bozulmasiz_mi(kod), f"{kod} bozulmasiz bir kosu"
    # Varyant adlari (bosluk iceren) katalogda olmamali.
    for s in senaryolar:
        assert " " not in s["kod"], s["kod"]


def test_katalog_katalog_yamldan_turetiliyor(senaryolar):
    """Liste koda gomulu OLMAMALI; senaryo defteri katalog.yaml'dir."""
    ham = yaml.safe_load(
        (ROOT / "senaryolar/katalog.yaml").read_text(encoding="utf-8")
    )
    beklenen = [s["kod"] for s in ham["senaryolar"]]
    assert [s["kod"] for s in senaryolar] == beklenen


def test_her_senaryonun_gorunen_adi_ve_ailesi_var(senaryolar):
    """Kartlarda kod degil INSAN DILINDEKI ad baskin olmali."""
    for s in senaryolar:
        assert s["ad"] and s["ad"] != s["kod"], s["kod"]
        assert "_" not in s["ad"], f"{s['kod']}: ad hala kod gibi ({s['ad']})"
        assert s["aile"] in katalog.AILE_ADI, s["kod"]


def test_negatif_bulgu_katalogdan_dusurulmuyor(senaryolar):
    """E3 iraksadi ve defterde satiri yok - ama bu bir NEGATIF BULGUDUR.

    Katalogdan cikarilirsa hipotezin test edilemedigi bilgisi kaybolur.
    """
    kodlar = {s["kod"] for s in senaryolar}
    assert "E3" in kodlar
    e3 = katalog.senaryo("E3")
    assert e3["olculdu"] is False
    assert e3["ana_kosu"] is None
    assert katalog.olculemeyen() and katalog.olculemeyen()[0]["kod"] == "E3"


def test_varyantlar_ana_kosudan_ayri_tutuluyor():
    """D4'un last_pt kaydi ayri bir senaryo degil, ayni hipotezin varyanti."""
    d4 = katalog.senaryo("D4")
    assert d4["ana_kosu"] == "D4"
    assert "D4 last_pt" in d4["varyantlar"]
    assert katalog.kosu_defteri_disinda_mi("D4")
    assert not katalog.kosu_defteri_disinda_mi("D4 last_pt")


def test_e3b_ana_kosusu_bir_seed_kosusu():
    """E3b iki seed ile kosuldu; ana kosu biri, digeri varyanttir."""
    e3b = katalog.senaryo("E3b")
    assert e3b["ana_kosu"] and e3b["ana_kosu"].startswith("E3b")
    assert e3b["varyantlar"]


# --- Ana etki gurultuye gore tartilmis ---------------------------------------

def test_ana_etki_gurultuye_gore_secilir(senaryolar):
    """GERCEK HATA: ham fark buyuklugune gore secilince her senaryoda
    `tf2026 recall` kazaniyordu - o grup 106 bbox tasiyor ve dogal yayilimi
    butun gruplarin en genisi (band 0.1321). Yani "en buyuk fark" cogu zaman
    en gurultulu grubun gurultusuydu."""
    for s in senaryolar:
        e = s["ana_etki"]
        if not e or e.get("eslenik"):
            continue
        assert e["oran"] > 1, (
            f"{s['kod']}: ana etki gurultunun icinde ({e['alan']} "
            f"oran={e['oran']:.2f})"
        )


def test_d4un_ana_etkisi_kucuk_nesne_bandinda():
    """D4'un asil bulgusu toplam metriklerde DEGIL, <16 px bandinda."""
    e = katalog.senaryo("D4")["ana_etki"]
    assert e and e["kirilim"] is True
    assert "16_alti" in e["alan"]
    assert e["fark"] < -0.4


def test_gurultu_icinde_kalan_senaryoda_ana_etki_yok():
    """D6b gurultu tabani olculunce bulgu olmaktan cikti."""
    assert not katalog.senaryo("D6b")["ana_etki"]


def test_eslenik_olcumde_esik_uygulanmaz():
    """E4 ve D6a'da ayni agirliklar kullanilir; egitim gurultusu devrede degil."""
    for kod in ("E4", "D6a"):
        e = katalog.senaryo(kod)["ana_etki"]
        assert e and e.get("eslenik") is True, kod


def test_az_gozlemli_band_ana_etki_olamaz():
    """n<5 kontrolle band kararsizdir; kural gurultu.py'de TEK yerde durur."""
    from teshis.degerlendirme.gurultu import fark_degerlendir

    d = fark_degerlendir("kaynak_recall", "tf2026", 0.05)
    assert "az_gozlem" in d, (
        "az_gozlem yapisal alan olarak disari verilmeli; yorum metninde "
        "arama yapmak 'ayni kural iki yerde' demektir"
    )


# --- Ajan kayitlari ---------------------------------------------------------

def test_kontrol_tekrarlari_demoya_bagli():
    """GERCEK BOSLUK: kosu_12 ve kosu_13 ana denemeden SONRA eklendi.

    Ajan cevaplari uretildi ve diske yazildi ama ayri dosyalarda durdugu
    icin demo "kayitli cevap yok" diyordu. Kayitlar kayip degildi, BAGLI
    degildi.
    """
    from data_loader import ajan_kaydi, ajan_kaydi_var_mi

    kayit = ajan_kaydi()
    tekrarlar = kayit.get("tekrarlar") or {}
    if not tekrarlar:
        pytest.skip("kontrol tekrari kaydi yok")
    for kosu_id in ("kosu_12", "kosu_13"):
        assert ajan_kaydi_var_mi(kosu_id) == "tekrar", kosu_id
        assert tekrarlar[kosu_id][0]["cevap"].get("diagnosis")
        assert tekrarlar[kosu_id][0]["arac_cagrilari"], (
            f"{kosu_id}: arac cagri kaydi bos"
        )


def test_tekrarlar_ana_denemenin_puanina_karistirilmiyor():
    """Iki ayri deney; birlestirilirse "ajan 13 kosuda %X" gibi yanlis bir
    sayi uretilirdi."""
    from data_loader import ajan_kaydi

    kayit = ajan_kaydi()
    puanlanan = set(kayit["puanlar"])
    assert "kosu_12" not in puanlanan and "kosu_13" not in puanlanan
    assert len(puanlanan) == len(kayit["cevaplar"])


def test_kayit_durumu_uc_degerden_biri():
    from data_loader import ajan_kaydi_var_mi, ajan_kosu_haritasi

    for kosu_id in ajan_kosu_haritasi():
        assert ajan_kaydi_var_mi(kosu_id) in ("ana", "tekrar", "yok")


# --- Arayuz metni -----------------------------------------------------------

def test_mutlak_fark_basligi_kalmadi():
    """GERCEK HATA: sutun adi "mutlak fark"ti ama degerler negatifti.

    Mutlak fark negatif olamaz; baslik bilimsel olarak yanlisti.
    """
    for yol in sorted((ROOT / "demo").rglob("*.py")):
        if "__pycache__" in yol.parts:
            continue
        metin = yol.read_text(encoding="utf-8")
        for satir in metin.splitlines():
            # Yorumlar kapsam disi: hatanin NEDEN yanlis oldugunu anlatan
            # aciklama satirlari ifadeyi zorunlu olarak iceriyor.
            if satir.strip().startswith("#"):
                continue
            if '"mutlak fark"' in satir:
                pytest.fail(f"{yol.name}: {satir.strip()}")

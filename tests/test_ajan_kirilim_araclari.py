"""Ajanin kirilim araclari (boyut / kaynak / sinif karisikligi) icin testler.

Bu araclar gercek bir bosluktan dogdu: ajan yalnizca toplam mAP/precision/
recall ve sinif AP50 goruyordu, oysa D3b, D4 ve D5 tam olarak o veride
gorunmuyor. Ajan bu senaryolari yapisal olarak teshis edemezdi.

En kritik sozlesme anonimlik: bu araclar senaryo adi, veri surumu, run_id
veya gercek kaynak adi (aaterm, hituav ...) sizdirmamalidir.
"""

import json
from pathlib import Path

import pytest

from teshis.ajan import ajan, araclar, semalar

ROOT = Path(__file__).resolve().parents[1]
KIRILIM_VAR = araclar.KIRILIM_DIR.is_dir() and any(araclar.KIRILIM_DIR.glob("*.json"))
pytestmark = pytest.mark.skipif(not KIRILIM_VAR, reason="kirilim analizleri uretilmemis")

YENI_ARACLAR = (
    "boyut_bazli_recall_getir",
    "kaynak_bazli_recall_getir",
    "sinif_karisikligini_getir",
)

# Ajana asla sizmamasi gereken terimler.
YASAKLI = [
    "d1", "d2a", "d2b", "d3", "d3b", "d4", "d5", "v00", "v01", "v02", "v03",
    "v04", "v05", "v06", "v07", "saglikli", "manifest", "run_", "best.pt",
    "aaterm", "hituav", "termal", "sentetik", "tf2026", "kucuk_nesne",
    "eksik_etiket", "sinif_yetersizligi", "kaynak_alani",
]


def _senaryo_kosulari() -> list[str]:
    return [k for k in araclar.kosu_listesini_getir() if k != "kosu_01"]


@pytest.mark.parametrize("arac", YENI_ARACLAR)
def test_arac_bildirimi_ve_uygulamasi_eslesir(arac):
    adlar = {b["name"] for b in semalar.ARAC_BILDIRIMLERI}
    assert arac in adlar, f"{arac} semalar.ARAC_BILDIRIMLERI'de yok"
    assert arac in ajan.ARAC_UYGULAMALARI, f"{arac} ajan.ARAC_UYGULAMALARI'da yok"


def test_bildirim_ve_uygulama_kumeleri_birebir():
    assert {b["name"] for b in semalar.ARAC_BILDIRIMLERI} == set(ajan.ARAC_UYGULAMALARI)


@pytest.mark.parametrize("arac", YENI_ARACLAR)
def test_her_senaryo_kosusunda_calisir(arac):
    fonksiyon = ajan.ARAC_UYGULAMALARI[arac]
    for kosu_id in _senaryo_kosulari():
        sonuc = fonksiyon(kosu_id)
        assert isinstance(sonuc, dict) and sonuc, f"{arac}({kosu_id}) bos dondu"


@pytest.mark.parametrize("arac", YENI_ARACLAR)
def test_anonimlik_sizintisi_yok(arac):
    """Arac ciktilarinda senaryo/kaynak/dosya adi gecmemelidir."""
    fonksiyon = ajan.ARAC_UYGULAMALARI[arac]
    for kosu_id in _senaryo_kosulari():
        blob = json.dumps(fonksiyon(kosu_id), ensure_ascii=False).lower()
        sizan = [terim for terim in YASAKLI if terim in blob]
        assert not sizan, f"{arac}({kosu_id}) sizdiriyor: {sizan}"


def test_kaynak_adlari_anonimlestirilir():
    """Gercek kaynak adlari yerine kaynak_a, kaynak_b ... kullanilmali."""
    sonuc = araclar.kaynak_bazli_recall_getir(_senaryo_kosulari()[0])
    for ad in sonuc["kaynaklar"]:
        assert ad.startswith("kaynak_"), f"anonimlestirilmemis kaynak adi: {ad}"


def test_kaynak_takma_adlari_kosular_arasinda_tutarli():
    """Ayni takma ad, tum kosularda ayni gercek kaynagi gostermeli.

    Aksi halde ajan iki kosuyu karsilastirdiginda farkli kaynaklari
    kiyaslamis olur.
    """
    kosular = _senaryo_kosulari()
    referans = {
        ad: deger["bbox_n"]
        for ad, deger in araclar.kaynak_bazli_recall_getir(kosular[0])["kaynaklar"].items()
    }
    for kosu_id in kosular[1:]:
        simdiki = {
            ad: deger["bbox_n"]
            for ad, deger in araclar.kaynak_bazli_recall_getir(kosu_id)["kaynaklar"].items()
        }
        assert simdiki == referans, f"{kosu_id} kaynak takma adlari kaymis"


def test_boyut_bandi_referansla_birlikte_doner():
    sonuc = araclar.boyut_bazli_recall_getir(_senaryo_kosulari()[0])
    assert sonuc["bantlar"], "bant bulunamadi"
    for bant, deger in sonuc["bantlar"].items():
        assert {"bbox_n", "recall", "referans_recall", "fark"} <= set(deger)


def test_karisiklik_matrisi_referansla_birlikte_doner():
    sonuc = araclar.sinif_karisikligini_getir(_senaryo_kosulari()[0])
    assert sonuc, "karisiklik matrisi bos"
    for sinif, deger in sonuc.items():
        assert "tahminler" in deger and "referans_tahminler" in deger
        assert deger["toplam_gercek_kutu"] == sum(deger["tahminler"].values())


@pytest.mark.parametrize("arac", YENI_ARACLAR)
def test_gecersiz_kosu_ajani_dusurmez(arac):
    """Arac hatasi istisna firlatmak yerine ajan katmaninda hata sozlugune donusmeli."""
    sonuc = ajan._arac_cagrisini_calistir(arac, {"kosu_id": "kosu_99"})
    assert "hata" in sonuc


@pytest.mark.parametrize("arac", YENI_ARACLAR)
def test_baseline_kosusu_acik_hata_verir(arac):
    """kosu_01 icin kirilim yok; sessizce bos donmek yerine acik hata vermeli."""
    sonuc = ajan._arac_cagrisini_calistir(arac, {"kosu_id": "kosu_01"})
    assert "hata" in sonuc

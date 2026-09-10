"""Gorsel kanit secimi gercekten senaryoya bagli mi?

Kapatilan hata
--------------
Hata galerileri kareleri senaryodan BAGIMSIZ bir zorluk skoruyla
(FN + FP + 1-IoU) siralar. Konsol da kareyi o siralamadan aliyordu; sonuc
olculdu: **26 galerinin 24'unde en ustteki kare ayni**
(`hituav__1_130_30_0_03841.jpg`). Yani "gorsel kanit" bolumu D3b ile D4'u
birbirinden ayirt etmiyordu.

Buradaki testlerin her biri o davranisi yeniden uretir: eski secim geri
konursa (yani secim yine mutlak skora dayanirsa veya imza filtresi
uygulanmazsa) ilgili test coker.

Kilitli tani seti (`val_diagnostic/`) Git disidir; etiket okunamayan
makinede imzaya bagli testler atlanir - ama imzanin KENDISI senaryo
YAML'lerinden okundugu icin o testler her yerde calisir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "demo"))

pytest.importorskip("yaml", reason="PyYAML kurulu degil")
pytest.importorskip("pandas", reason="demo bagimliliklari kurulu degil")

import kanit_secimi  # noqa: E402


def _tani_seti_var() -> bool:
    import gorseller

    return gorseller.durum()["kaynak"] == "tam"


def _galeri_var(kosu: str) -> bool:
    from data_loader import error_galleries

    return kosu in error_galleries()


# --- Imza senaryo YAML'inden turetiliyor mu --------------------------------

def test_sinif_takasi_senaryolari_iki_sinifi_birden_ister():
    """D3b'nin imzasi tasit VE insan; tek sinif yeterli sayilirsa kanit bos."""
    im = kanit_secimi.imza("D3b")
    assert im, "D3b imzasiz kaldi - senaryo YAML'i okunamiyor"
    assert sorted(im["siniflar"]) == [0, 1]
    assert im["hepsi_gerekli"] is True


def test_d3_imzasi_nadir_siniflari_gosterir():
    im = kanit_secimi.imza("D3")
    assert sorted(im["siniflar"]) == [2, 3]
    assert im["hepsi_gerekli"] is True


def test_sinif_yetersizligi_tek_sinif_ister():
    """D1 sinif seyreltir, takas etmez: tek sinifin varligi yeter."""
    im = kanit_secimi.imza("D1")
    assert im["siniflar"] == [1]
    assert im["hepsi_gerekli"] is False


def test_boyut_ve_kaynak_imzalari_okunuyor():
    assert kanit_secimi.imza("D4")["boyut_esigi_px"] == 16
    assert kanit_secimi.imza("D5")["izinli_kaynaklar"] == ["aaterm"]


def test_imzasiz_senaryolar_bos_doner():
    """Bu bozulmalar sinifa/banda/kaynaga bagli degil; uydurma filtre olmamali."""
    for kod in ("D2a", "D2b", "D6a", "D6b", "E1", "E2"):
        assert kanit_secimi.imza(kod) == {}, kod


def test_yazim_hatasi_duzeltilse_de_imza_okunur():
    """D3b'nin YAML anahtari `sinif_ciftii` (yazim hatali); ikisi de okunur."""
    assert "sinif_cifti" in kanit_secimi.SINIF_ANAHTARLARI
    assert "sinif_ciftii" in kanit_secimi.SINIF_ANAHTARLARI


# --- Yon ayrimi -------------------------------------------------------------

def test_yon_ayrimi_iyilesmeyi_bozulma_saymaz():
    """Kosunun referanstan DAHA AZ hata yaptigi kare bozulma kaniti degildir."""
    assert kanit_secimi.yon(2.0) == "bozulma"
    assert kanit_secimi.yon(-2.0) == "iyileşme"
    assert kanit_secimi.yon(0.0) == "fark yok"


def test_delta_referansa_gore_hesaplanir():
    kayit = {"false_negatives": 5, "false_positives": 2, "mean_iou": 0.60}
    referans = {"false_negatives": 3, "false_positives": 2, "mean_iou": 0.90}
    assert kanit_secimi._delta(kayit, referans) == pytest.approx(2.3)


# --- Secim gercekten senaryoya bagli mi ------------------------------------

@pytest.mark.skipif(not _galeri_var("D3b"), reason="hata galerisi yok")
@pytest.mark.skipif(not _tani_seti_var(), reason="kilitli tani seti bu makinede yok")
def test_d3b_kanitinda_her_iki_karistirilan_sinif_bulunur():
    """Kapatilan hata: D3b'nin kaniti yalnizca insan iceren bir kareydi."""
    import gorseller

    secim = kanit_secimi.adaylar("D3b")
    assert secim["olcut"] == "imza", secim["not"]
    assert secim["kayitlar"], "imzaya uyan kare bulunamadi"

    kat = {k["dosya"]: k for k in gorseller.katalog()}
    for kayit in secim["kayitlar"]:
        siniflar = {b["sinif"] for b in kat[kayit["source"]]["kutular"]}
        assert {0, 1} <= siniflar, (
            f"{kayit['source']} karesinde taşıt ve insan birlikte yok; "
            "bu kare sınıf karışıklığını gösteremez"
        )


@pytest.mark.skipif(not _galeri_var("D4"), reason="hata galerisi yok")
@pytest.mark.skipif(not _tani_seti_var(), reason="kilitli tani seti bu makinede yok")
def test_d4_kanitinda_esik_altinda_nesne_bulunur():
    ozetler = kanit_secimi._kare_ozetleri()
    secim = kanit_secimi.adaylar("D4")
    assert secim["olcut"] == "imza", secim["not"]
    for kayit in secim["kayitlar"]:
        assert ozetler[kayit["source"]]["en_kucuk_px"] < 16


@pytest.mark.skipif(not _galeri_var("D5"), reason="hata galerisi yok")
@pytest.mark.skipif(not _tani_seti_var(), reason="kilitli tani seti bu makinede yok")
def test_d5_kaniti_egitimde_tutulan_kaynaktan_gelmez():
    ozetler = kanit_secimi._kare_ozetleri()
    secim = kanit_secimi.adaylar("D5")
    for kayit in secim["kayitlar"]:
        assert ozetler[kayit["source"]]["kaynak"] != "aaterm"


@pytest.mark.skipif(not _galeri_var("D3"), reason="hata galerisi yok")
@pytest.mark.skipif(not _tani_seti_var(), reason="kilitli tani seti bu makinede yok")
def test_imza_saglanamayinca_sessizce_dusulmez():
    """D3'un imzasi (UAP/UAI) galeride yok; bu SOYLENMELI, gizlenmemeli.

    Uymayan bir kareyi kanit diye sunmak, projenin metodolojisiyle
    celisirdi. Genel siralamaya dusuldugunde `olcut` bunu bildirir ve `not`
    nedenini yazar.
    """
    secim = kanit_secimi.adaylar("D3")
    assert secim["olcut"] == "genel"
    assert secim["imza"], "D3 imzasiz gorunuyor"
    assert "değildir" in secim["not"] or "uygulanamadı" in secim["not"]


@pytest.mark.skipif(not _tani_seti_var(), reason="kilitli tani seti bu makinede yok")
def test_senaryolar_ayni_kareyi_gostermiyor():
    """Kapatilan hatanin ta kendisi: 26 galerinin 24'u ayni kareyi veriyordu."""
    from data_loader import error_galleries

    ilk_kareler = []
    for kosu in error_galleries():
        secim = kanit_secimi.adaylar(kosu)
        if secim["kayitlar"]:
            ilk_kareler.append(secim["kayitlar"][0]["source"])

    assert len(ilk_kareler) >= 10, "yeterli galeri yok, test anlamsiz"
    en_sik = max(set(ilk_kareler), key=ilk_kareler.count)
    pay = ilk_kareler.count(en_sik) / len(ilk_kareler)
    assert pay < 0.5, (
        f"koşuların %{pay * 100:.0f}'i aynı kareyi ({en_sik}) gösteriyor; "
        "seçim senaryodan bağımsız kalmış"
    )


@pytest.mark.skipif(not _galeri_var("v00_saglikli"), reason="hata galerisi yok")
def test_referans_kendisiyle_karsilastirilmaz():
    """Referansin kendi galerisinde her delta sifirdir; bu 'bozulma yok' degildir."""
    secim = kanit_secimi.adaylar("v00_saglikli")
    assert secim["kendi_referansi"] is True

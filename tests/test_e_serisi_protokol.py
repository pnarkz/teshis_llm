"""E serisi protokol sapmalarinin beyan edilmis ve denetlenebilir olmasini dogrular.

D serisi ile E serisi ters yonde calisir:

- D serisi VERIYI bozar, egitim protokolunu sabit tutar. Karsilastirma
  gecerlidir cunku tek degisken veri surumudur.
- E serisi veriyi temiz tutar (v00_saglikli), EGITIM PROTOKOLUNU bozar.

Bu yuzden E kosulari `sabit` bloktan kasitli olarak sapar. Tehlike, bu
sapmalarin CLI bayraklari veya script icine gomulu sayilar halinde dagilmasi
ve hangi kosunun protokolden nerede ayrildiginin izlenemez hale gelmesidir.
Asagidaki testler sapmalarin tek bir yerde - senaryolar/egitim_protokolu.yaml
icinde - beyan edilmis kalmasini zorunlu kilar.
"""

import pytest
import yaml

from teshis.egitim import protokol

E_KODLARI = ["E1", "E2", "E3", "E4"]


@pytest.fixture(scope="module")
def ham():
    return yaml.safe_load(protokol.PROTOKOL_YOLU.read_text(encoding="utf-8"))


def test_katalogdaki_her_e_senaryosu_protokolde_tanimli(ham):
    """senaryolar/egitim/ altindaki her E senaryosunun sapma beyani olmali.

    Konfig dosya adlari kucuk harfle baslar (e1_overfitting.yaml); senaryo
    kodlari buyuk harftir (E1). Kural senaryolar/veri/ altinda da aynidir.
    """
    konfigler = sorted(
        p.stem.split("_")[0].upper()
        for p in (protokol.PROTOKOL_YOLU.parent / "egitim").glob("*.yaml")
    )
    tanimli = set(ham["e_serisi"])
    eksik = [k for k in konfigler if k not in tanimli]
    assert not eksik, f"Bu E senaryolarinin sapma beyani yok: {eksik}"


@pytest.mark.parametrize("kod", E_KODLARI)
def test_her_senaryonun_aciklamasi_ve_kosu_ayari_var(kod):
    ayar = protokol.e_senaryo_ayarlari(kod)
    assert ayar.get("aciklama", "").strip(), f"{kod}: aciklama bos"
    assert "sapmalar" in ayar, f"{kod}: sapmalar anahtari yok (bos olabilir, eksik olamaz)"
    assert ayar.get("kosu_ayarlari"), f"{kod}: kosu_ayarlari bos"


@pytest.mark.parametrize("kod", E_KODLARI)
def test_sapmalar_protokolde_var_olan_alanlari_hedefler(kod):
    """Sapma, mevcut bir protokol alanini DEGISTIRMELI; yeni alan eklememeli.

    Yeni alan eklenmesi sessiz bir protokol genislemesi olurdu: D serisi
    kosulari o alani hic gormemis olacagi icin karsilastirma bozulurdu.
    """
    taban = protokol.egitim_kwargs()
    sapmalar = protokol.e_senaryo_ayarlari(kod)["sapmalar"] or {}
    bilinmeyen = sorted(set(sapmalar) - set(taban))
    assert not bilinmeyen, f"{kod}: protokolde olmayan alan(lar): {bilinmeyen}"


@pytest.mark.parametrize("kod", E_KODLARI)
def test_sapmalar_gercekten_farkli_deger_tasir(kod):
    """Protokolle ayni degeri tekrar yazan bir 'sapma' yaniltici olur."""
    taban = protokol.egitim_kwargs()
    sapmalar = protokol.e_senaryo_ayarlari(kod)["sapmalar"] or {}
    etkisiz = [k for k, v in sapmalar.items() if taban[k] == v]
    assert not etkisiz, f"{kod}: protokolle ayni degeri tasiyan sapmalar: {etkisiz}"


def test_egitim_kwargs_e_tabani_sapmayla_birlestirir():
    taban = protokol.egitim_kwargs()
    e1 = protokol.egitim_kwargs_e("E1")
    assert e1["mosaic"] == 0.0 and taban["mosaic"] == 0.5
    assert e1["fliplr"] == 0.0 and taban["fliplr"] == 0.5
    # Sapmada gecmeyen alanlar protokolden aynen gelmeli
    assert e1["lr0"] == taban["lr0"]
    assert e1["cos_lr"] == taban["cos_lr"]


def test_e_serisi_d_serisi_protokolunu_kirletmez():
    """egitim_kwargs() cagrisi E sapmalarindan etkilenmemelidir."""
    once = protokol.egitim_kwargs()
    protokol.egitim_kwargs_e("E1")
    protokol.egitim_kwargs_e("E3")
    assert protokol.egitim_kwargs() == once


def test_bilinmeyen_senaryo_kodu_reddedilir():
    with pytest.raises(KeyError, match="e_serisi"):
        protokol.e_senaryo_ayarlari("E9")


def test_e1_augmentasyonu_tamamen_kapatir():
    """E1'in tezi 'augmentasyon yoksa ezberler'; tek bir acik kalan bozar."""
    e1 = protokol.egitim_kwargs_e("E1")
    for alan in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                 "shear", "perspective", "flipud", "fliplr", "mosaic",
                 "mixup", "copy_paste"):
        assert e1[alan] == 0.0, f"E1'de {alan} acik kalmis: {e1[alan]}"


def test_e1_ve_e2_erken_durdurmayi_devre_disi_birakir():
    """Erken durdurma acikken E1 overfit'e, E2 de 5 epoch'a ulasamayabilir."""
    taban_patience = protokol.egitim_kwargs()["patience"]
    for kod, ayar in (("E1", protokol.e_senaryo_ayarlari("E1")),
                      ("E2", protokol.e_senaryo_ayarlari("E2"))):
        patience = protokol.egitim_kwargs_e(kod)["patience"]
        epochs = ayar["kosu_ayarlari"]["epochs"]
        assert patience > epochs > 0, (
            f"{kod}: patience ({patience}) epoch sayisindan ({epochs}) buyuk olmali; "
            f"protokol tabani {taban_patience}"
        )


def test_e3_iki_seed_ile_kosulur():
    """Kararsizlik tek kosuda gosterilemez; oynaklik iki seed arasinda olculur."""
    ayar = protokol.e_senaryo_ayarlari("E3")["kosu_ayarlari"]
    assert ayar.get("ikinci_seed"), "E3 ikinci bir seed tanimlamali"
    assert ayar["ikinci_seed"] != 42, "ikinci seed varsayilan seed ile ayni olamaz"


def test_e3_ogrenme_orani_belirgin_sekilde_yuksek():
    taban = protokol.egitim_kwargs()["lr0"]
    assert protokol.egitim_kwargs_e("E3")["lr0"] >= taban * 50


def test_e4_egitim_gerektirmez():
    """E4 cikarim tarafinda bir uyumsuzluktur; egitim protokolu bozulmaz."""
    ayar = protokol.e_senaryo_ayarlari("E4")
    assert not ayar["sapmalar"], "E4 egitim protokolunden sapmamali"
    olcum = ayar["kosu_ayarlari"]["olcum_imgsz"]
    egitim = ayar["kosu_ayarlari"]["egitim_imgsz"]
    assert egitim in olcum, "olcum listesi egitim cozunurlugunu de icermeli (kontrol noktasi)"
    assert len(olcum) >= 3, "tek bir uyumsuz deger egri cizmeye yetmez"

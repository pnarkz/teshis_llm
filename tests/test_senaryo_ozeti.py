"""Senaryo ozeti kaynak dosyalardan TURETILIYOR mu?

Demo, senaryo aciklamalarini 24 girdilik elle tutulan bir sozlukte tasiyordu
ve geride kaliyordu: D6a, D6b, v00n ve D1n eklendiginde demo onlari sessizce
eksik gosterdi. Ozetin bes bileseninden dordu artik turetiliyor; elle yazilan
tek alan senaryonun ne olctugu.
"""

import csv
from pathlib import Path

import pytest

from teshis.degerlendirme import senaryo_ozeti as so

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def senaryolar() -> list[str]:
    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        return [s["scenario"] for s in csv.DictReader(f)]


def test_her_kosu_icin_ozet_uretilebiliyor(senaryolar):
    basarisiz = []
    for s in senaryolar:
        try:
            so.ozet(s)
        except Exception as sorun:  # noqa: BLE001
            basarisiz.append(f"{s}: {type(sorun).__name__} {sorun}")
    assert not basarisiz, basarisiz


def test_her_kosunun_gozlem_ve_kanit_gucu_var(senaryolar):
    eksik = [
        s for s in senaryolar
        if not so.ozet(s)["ne_gozlendi"] or not so.ozet(s)["kanit_gucu"].get("seviye")
    ]
    assert not eksik, f"Gozlem veya kanit gucu uretilemeyen kosular: {eksik}"


def test_anlatim_dosyasi_her_senaryo_kodunu_kapsiyor(senaryolar):
    """Elle yazilan tek alan; eksik kalirsa senaryo sayfasi yarim gorunur."""
    eksik = sorted({
        so._kod(s) for s in senaryolar if not so.ozet(s)["ne_olcuyor"]
    })
    assert not eksik, (
        f"senaryolar/anlatim.yaml bu kodlari icermiyor: {eksik}"
    )


def test_veri_senaryolarinda_degisen_parametre_yazili():
    """'Ne degistirildi' sorusu konfigden cevaplanabilmeli."""
    for kod in ("D1", "D2a", "D2b", "D4", "D5"):
        d = so.ne_degisti(kod)
        assert d["tur"] == "veri", kod
        assert d["parametreler"], f"{kod}: degisen parametre bulunamadi"


def test_sabit_kalanlar_gercek_degerlendirme_setini_soyler(senaryolar):
    """Ozet, kosunun GERCEKTEN olculdugu kumeyi yazmali.

    GERCEK HATA: bu satir "val_diagnostic (kilitli, hic degismez)" diye SABIT
    yaziliyordu. D6a ise kasitli olarak sizintili kume uzerinde olculmustur,
    dolayisiyla sayfa kendi kendisiyle celisiyordu: ust kutuda "kilitli set
    hic degismez", alt kutuda "baska sette olculdu". Ustelik eski test tam
    da bu celiskiyi DOGRULUYORDU - her ozette "val_diagnostic" ariyordu.
    """
    import csv

    from teshis.degerlendirme.senaryo_ozeti import RESULTS_CSV

    with RESULTS_CSV.open(encoding="utf-8") as f:
        defter = {r["scenario"]: r for r in csv.DictReader(f)}

    for s in senaryolar:
        sabitler = " ".join(so.ne_sabit_kaldi(s))
        kume = defter[s]["evaluation_set"]
        assert kume in sabitler, f"{s}: ozette '{kume}' gecmiyor"
        if kume != "val_diagnostic":
            assert "kilitli set DEĞİL" in sabitler, (
                f"{s} kilitli set disinda olculdu ama ozet bunu soylemiyor"
            )
        else:
            assert "kilitli" in sabitler, s


def test_d6a_ozeti_kendisiyle_celismiyor():
    """Somut regresyon: D6a'nin sabitleri ve sinirlamalari ayni seyi soylemeli."""
    sabitler = " ".join(so.ne_sabit_kaldi("D6a"))
    sinirlar = " ".join(so.sinirlamalar("D6a"))
    assert "v08_sizintili_kume" in sabitler
    assert "v08_sizintili_kume" in sinirlar
    assert "val_diagnostic (kilitli" not in sabitler


def test_gurultu_icinde_kalan_senaryo_boyle_isaretlenir():
    """D6b hicbir genel metrikte esigi asmiyor; ozet bunu soylemeli."""
    guc = so.kanit_gucu("D6b")
    assert guc["seviye"] == "gurultu icinde", guc
    assert "iddia kurulamaz" in guc["aciklama"]


def test_guclu_senaryo_boyle_isaretlenir():
    """D2a birden fazla metrikte esigi asiyor."""
    assert so.kanit_gucu("D2a")["seviye"] == "guclu"


def test_farkli_kumede_olculen_kosu_sinirlamada_uyariyor():
    """D6a sizintili kumede olculdu; ozet bunu sinir olarak yazmali."""
    sinirlar = " ".join(so.sinirlamalar("D6a"))
    assert "kilitli set yerine" in sinirlar


def test_last_pt_satiri_sinirlamada_uyariyor():
    sinirlar = " ".join(so.sinirlamalar("D5 last_pt"))
    assert "last.pt" in sinirlar


def test_ozet_elle_tutulan_senaryo_sozlugu_gerektirmiyor():
    """demo/app.py'deki 24 girdilik scenario_info sozlugu geri gelmemeli."""
    kaynak = (ROOT / "teshis/degerlendirme/senaryo_ozeti.py").read_text(encoding="utf-8")
    assert "scenario_info" not in kaynak
    # Anlatim disindaki her sey turetilmeli: modulde senaryo koduna gore
    # sabitlenmis metin bloklari olmamali.
    assert kaynak.count('"D1"') == 0, "modul senaryo koduna gore metin tutuyor"


def test_yukselen_metrik_bozulma_kaniti_sayilmiyor():
    """Esigi asan bir YUKSELIS, esigi asan bir dususle ayni sey degildir.

    D1'de mAP50_95 referansa gore +0.0240 YUKSELDI ve esik 0.0201'di. Kural
    mutlak degere baktigi icin bu, uc ayri yerde bozulma kaniti gibi
    gorundu: hipotez tablosunda "kismen desteklendi", etki haritasinda
    esigi asan renkli hucre, ve burada "zayif bulgu" + "mAP50_95 gurultu
    esigini asiyor" cumlesi.

    Kural artik tek kaynakta yone gore ayriliyor.
    """
    g = so.ne_gozlendi("D1")
    assert g["metrikler"]["mAP50_95"]["fark"] > 0, "D1 mAP50_95 dusmus olmali degil"
    assert g["metrikler"]["mAP50_95"]["asiyor"], "buyukluk hala esigi asiyor"
    assert g["metrikler"]["mAP50_95"]["yon"] == "yukselis"

    assert "mAP50_95" in g["asan_metrikler"], (
        "asan_metrikler BUYUKLUK sorusudur, degismemeli"
    )
    assert "mAP50_95" not in g["asan_dusen"], (
        "yukselen bir metrik 'esigi asan dusus' listesine giremez"
    )
    assert "mAP50_95" in g["asan_yukselen"]

    guc = so.kanit_gucu("D1")
    assert guc["seviye"] == "gurultu icinde", (
        f"D1'de esigi asan hicbir DUSUS yok; seviye 'gurultu icinde' olmali, "
        f"'{guc['seviye']}' gelmis"
    )
    assert "YÜKSELİŞ" in guc["aciklama"], (
        "beklenmedik yondeki etki aciklamada gorunmeli, gizlenmemeli"
    )


def test_dusen_senaryolar_etkilenmiyor():
    """Duzeltme yalnizca yukselisi ayirir; gercek dususleri zayiflatmaz."""
    for senaryo, beklenen in (("D2a", "guclu"), ("D4", "guclu")):
        g = so.ne_gozlendi(senaryo)
        assert g["asan_dusen"], f"{senaryo}: esigi asan dusus kaybolmus"
        assert g["asan_dusen"] == [m for m in g["asan_metrikler"]
                                   if g["metrikler"][m]["yon"] == "dusus"]
        assert so.kanit_gucu(senaryo)["seviye"] == beklenen


def test_kontrol_kosusu_yukselisi_de_derecelendirilmiyor():
    """C2 seed21'in iki metrigi YUKSELEREK esigi asiyor - yine de bulgu degil."""
    g = so.ne_gozlendi("C2 seed21")
    assert g["asan_yukselen"] and not g["asan_dusen"]
    assert so.kanit_gucu("C2 seed21")["seviye"] == "kontrol kosusu"

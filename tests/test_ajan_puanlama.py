"""teshis/ajan/puanlama.py icin testler."""

from teshis.ajan import puanlama


def test_dogru_teshis_tam_puan_alir():
    cevap = {"diagnosis": "sinif yetersizligi nedeniyle recall dustu"}
    puan, etiket = puanlama.teshis_puani("sinif_yetersizligi", cevap)
    assert puan == 1.0
    assert etiket == "sinif yetersizligi"


def test_yanlis_teshis_sifir_puan_alir():
    cevap = {"diagnosis": "model asiri uyum (overfitting) gosteriyor"}
    puan, _ = puanlama.teshis_puani("sinif_yetersizligi", cevap)
    assert puan == 0.0


def test_eksik_etikette_kismi_puan_precision_recall_bahsiyle():
    cevap = {"diagnosis": "belirsiz bir durum ama precision ve recall degisti"}
    puan, _ = puanlama.teshis_puani("eksik_etiket", cevap)
    assert puan == 0.5


def test_bilinmeyen_beklenen_etiket_icin_degerlendirilemez():
    puan, etiket = puanlama.teshis_puani("henuz_tanimsiz_senaryo", {"diagnosis": "her sey"})
    assert puan == 0.0
    assert etiket is None


def test_puanlama_kosu_numarasindan_bagimsizdir():
    """Ayni cevap, kosu_NN numarasi degisse de ayni puani almalidir.

    kosu_NN numaralari results.csv satir sirasina bagli oldugu icin araya yeni
    bir kosu eklendiginde kayabilir; puanlama bu kaymadan etkilenmemelidir.
    """
    cevap = {
        "diagnosis": "eksik etiket",
        "evidence": ["precision 0.81", "recall 0.90"],
        "limitations": ["UAP n=15, UAI n=17"],
    }
    beklenen = {"expected": "eksik_etiket"}
    ilk = puanlama.kosuyu_puanla("kosu_04", beklenen, cevap)
    ikinci = puanlama.kosuyu_puanla("kosu_09", beklenen, cevap)
    assert ilk["total"] == ikinci["total"] == 1.0


def test_her_senaryonun_puanlama_kalibi_vardir():
    """SENARYO_BEKLENEN'deki her beklenen etiket icin bir regex kalibi olmalidir."""
    eksik = [
        beklenen for beklenen in puanlama.SENARYO_BEKLENEN.values()
        if beklenen not in puanlama.ANAHTAR_KALIPLAR
    ]
    assert not eksik, f"Bu beklenen teshisler icin kalip yok: {eksik}"


def test_kanit_puani_iki_sayisal_kanit_ister():
    assert puanlama.kaniti_puanla({"evidence": ["recall 0.72", "precision 0.81"]}) == 1.0
    assert puanlama.kaniti_puanla({"evidence": ["recall dustu"]}) == 0.0
    assert puanlama.kaniti_puanla({"evidence": ["recall dustu", "precision dustu"]}) == 0.0
    assert puanlama.kaniti_puanla({}) == 0.0


def test_sinir_puani_uap_uai_bbox_sayisini_arar():
    assert puanlama.siniri_puanla({"limitations": ["UAP n=15, UAI n=17 dusuk, genelleme sinirli"]}) == 1.0
    assert puanlama.siniri_puanla({"limitations": ["genel olarak veri az"]}) == 0.0


def test_paketi_puanla_ortalamayi_hesaplar():
    cevaplar = [
        {"run_id": "kosu_01", "diagnosis": "saglikli referans", "evidence": ["mAP50 0.93", "recall 0.88"], "limitations": ["UAP n=15, UAI n=17"]},
        {"run_id": "kosu_02", "diagnosis": "alakasiz bir aciklama", "evidence": [], "limitations": []},
    ]
    anahtar = {
        "kosu_01": {"expected": "saglikli_referans"},
        "kosu_02": {"expected": "sinif_yetersizligi"},
    }
    sonuc = puanlama.paketi_puanla(cevaplar, anahtar)
    assert sonuc["runs"][0]["total"] == 1.0
    assert sonuc["runs"][1]["total"] == 0.0
    assert sonuc["mean_score"] == 0.5


def test_paketi_puanla_eksik_cevabi_missing_sayar():
    sonuc = puanlama.paketi_puanla([], {"kosu_01": {"expected": "saglikli_referans"}})
    assert sonuc["runs"][0]["model_diagnosis"] == "missing"
    assert sonuc["mean_score"] == 0.0


def test_sinirlama_grup_adi_ve_sayi_ister():
    """Mugla belirsizlik ifadeleri puan almamali; grup adi + sayi sart."""
    assert puanlama.siniri_puanla({"limitations": ["UAP (n=15) ve UAI (n=17) azdir."]}) == 1.0
    assert puanlama.siniri_puanla({"limitations": ["kaynak_d bbox sayisi 106 ile sinirlidir."]}) == 1.0
    for mugla in ([], ["Belirsizlik var."], ["Az ornek var."], ["Veri sinirli."],
                  ["UAP ve UAI siniflarinda ornek azdir."]):
        assert puanlama.siniri_puanla({"limitations": mugla}) == 0.0, mugla


def test_eksik_etikette_turkce_terimler_kismi_puan_alir():
    """Model Turkce cevapladiginda 'hassasiyet/duyarlilik' da kabul edilmeli."""
    puan, _ = puanlama.teshis_puani(
        "eksik_etiket", {"diagnosis": "belirsiz ama hassasiyet ve duyarlilik degisti"}
    )
    assert puan == 0.5


def test_tespit_edilemeyen_kosuda_degisim_yok_cevabi_kabul_edilir():
    """Bozulmanin kanitta iz birakmadigi kosuda 'anlamli degisim yok' dogru sayilmali."""
    cevap = {"diagnosis": "performans_stabil_hafif_iyilesme",
             "evidence": ["mAP50 0.92", "recall 0.87"],
             "limitations": ["UAP (n=15) ve UAI (n=17) azdir."]}
    beklenen = {"expected": "sinif_yetersizligi", "hidden_role": "D1"}
    sonuc = puanlama.kosuyu_puanla("kosu_02", beklenen, cevap)
    assert sonuc["diagnosis_score"] == 0.0          # kati puan: bozulma adi soylenmedi
    assert sonuc["diagnosis_score_tespit"] == 1.0   # tespit-farkindalikli: dogru
    assert sonuc["tespit_edilebilir"] is False
    assert sonuc["tespit_notu"]


def test_tespit_edilebilir_kosuda_iki_puan_ayni():
    """Bozulmanin gorundugu kosuda tespit-farkindalikli puan avantaj saglamamali."""
    cevap = {"diagnosis": "alakasiz bir aciklama", "evidence": [], "limitations": []}
    beklenen = {"expected": "kucuk_nesne_sinyal_kaybi", "hidden_role": "D4"}
    sonuc = puanlama.kosuyu_puanla("kosu_08", beklenen, cevap)
    assert sonuc["diagnosis_score"] == sonuc["diagnosis_score_tespit"] == 0.0
    assert sonuc["tespit_edilebilir"] is True


def test_paket_iki_ortalama_dondurur():
    anahtar = {"kosu_02": {"expected": "sinif_yetersizligi", "hidden_role": "D1"}}
    cevaplar = [{"run_id": "kosu_02", "diagnosis": "stabil",
                 "evidence": ["a 1", "b 2"], "limitations": ["UAP (n=15)"]}]
    sonuc = puanlama.paketi_puanla(cevaplar, anahtar)
    assert "mean_score" in sonuc and "mean_score_tespit" in sonuc
    assert sonuc["mean_score_tespit"] >= sonuc["mean_score"]


def test_snake_case_teshis_eslesir():
    """Model snake_case yazar; kaliplar dogal dilde. Normallestirme sart.

    Gercek bir kosuda ajan D1 icin "bozulma_tespit_edilmedi" dedi - bizim
    analizimize gore DOGRU cevap - ama kalip "bozulma yok" aradigi icin
    sifir aliyordu.
    """
    assert puanlama.teshis_puani("anlamli_degisim_yok",
                                 {"diagnosis": "bozulma_tespit_edilmedi"})[0] == 1.0
    assert puanlama.teshis_puani("kucuk_nesne_sinyal_kaybi",
                                 {"diagnosis": "cok_kucuk_nesne_tespit_kaybi"})[0] == 1.0
    assert puanlama.teshis_puani("eksik_etiket",
                                 {"diagnosis": "yuksek-yanlis-pozitif-orani"})[0] == 1.0


def test_normallestirme_yanlis_cevabi_dogru_yapmaz():
    """Normallestirme yalnizca ayirici karakterleri duzeltmeli; anlam eklememeli."""
    assert puanlama.teshis_puani("anlamli_degisim_yok",
                                 {"diagnosis": "tamamen_alakasiz_bir_aciklama"})[0] == 0.0

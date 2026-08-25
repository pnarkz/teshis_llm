"""teshis/ajan/semalar.py::teshis_dogrula icin sozlesme testleri."""

from teshis.ajan import semalar

GECERLI_CEVAP = {
    "diagnosis": "sinif yetersizligi",
    "evidence": ["insan recall 0.72 (baseline 0.82)", "insan AP50 -0.034"],
    "confidence": "orta",
    "limitations": ["UAP bbox n=15, UAI bbox n=17, genelleme sinirli"],
    "next_measurement": "D2b eksik etiket senaryosu",
}


def test_gecerli_cevap_hatasiz_gecer():
    assert semalar.teshis_dogrula(GECERLI_CEVAP) == []


def test_eksik_alan_yakalanir():
    eksik = {key: value for key, value in GECERLI_CEVAP.items() if key != "confidence"}
    hatalar = semalar.teshis_dogrula(eksik)
    assert "eksik_alan:confidence" in hatalar


def test_gecersiz_confidence_yakalanir():
    bozuk = {**GECERLI_CEVAP, "confidence": "cok_yuksek"}
    assert "confidence_gecersiz_deger" in semalar.teshis_dogrula(bozuk)


def test_tek_kanit_yetersiz_sayilir():
    bozuk = {**GECERLI_CEVAP, "evidence": ["sadece bir kanit"]}
    assert "evidence_en_az_iki_ogeli_liste_olmali" in semalar.teshis_dogrula(bozuk)


def test_sayisal_olmayan_kanitlar_yakalanir():
    bozuk = {**GECERLI_CEVAP, "evidence": ["recall dustu", "precision dustu"]}
    assert "evidence_en_az_bir_sayisal_kanit_icermeli" in semalar.teshis_dogrula(bozuk)


def test_limitations_liste_olmalidir():
    bozuk = {**GECERLI_CEVAP, "limitations": "UAP/UAI dusuk"}
    assert "limitations_liste_olmali" in semalar.teshis_dogrula(bozuk)


def test_arac_bildirimleri_isimleri_araclar_moduluyle_eslesir():
    from teshis.ajan import ajan

    bildirilen_adlar = {bildirim["name"] for bildirim in semalar.ARAC_BILDIRIMLERI}
    uygulanan_adlar = set(ajan.ARAC_UYGULAMALARI)
    assert bildirilen_adlar == uygulanan_adlar

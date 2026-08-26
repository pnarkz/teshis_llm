"""Function-calling ajan dongusunun API gerektirmeyen parcalari icin testler.

Canli API cagrisi yapan teshis_uret burada test edilmez; JSON ayiklama,
arac dispatch'i, arac kullanim ozeti ve cikti yolu sozlesmeleri test edilir.
Bunlar gercek hatalardan dogdu: onceki surum ham json.loads cagiriyordu
(markdown blogunda cokerdi) ve ciktisini tek atislik denemenin dosyasinin
uzerine yaziyordu.
"""

import json

import pytest

from teshis.ajan import ajan, semalar


# --- JSON ayiklama ---------------------------------------------------------

def test_duz_json_ayiklanir():
    assert ajan.json_ayikla('{"diagnosis": "x"}') == {"diagnosis": "x"}


def test_markdown_blogundaki_json_ayiklanir():
    metin = '```json\n{"diagnosis": "kucuk_nesne"}\n```'
    assert ajan.json_ayikla(metin)["diagnosis"] == "kucuk_nesne"


def test_dil_etiketsiz_blok_ayiklanir():
    assert ajan.json_ayikla('```\n{"a": 1}\n```') == {"a": 1}


def test_aciklama_metniyle_gelen_json_ayiklanir():
    metin = 'Analiz tamamlandi:\n{"diagnosis": "y", "confidence": "orta"}\nUmarim yardimci olur.'
    assert ajan.json_ayikla(metin)["diagnosis"] == "y"


def test_bos_cevap_acik_hata_verir():
    with pytest.raises(ValueError, match="bos cevap"):
        ajan.json_ayikla("   ")


def test_jsonsuz_cevap_acik_hata_verir():
    with pytest.raises(ValueError, match="gecerli JSON bulunamadi"):
        ajan.json_ayikla("burada hic json yok")


# --- Arac dispatch ---------------------------------------------------------

def test_bilinmeyen_arac_hata_sozlugu_dondurur():
    assert "hata" in ajan._arac_cagrisini_calistir("olmayan_arac", {})


def test_gecersiz_arguman_ajani_dusurmez():
    sonuc = ajan._arac_cagrisini_calistir("kosu_metriklerini_getir", {"kosu_id": "kosu_yok"})
    assert "hata" in sonuc


def test_liste_donen_arac_sozluge_sarilir():
    """FunctionResponse.response sozluk olmali; liste donen arac sarilmalidir."""
    sonuc = ajan._arac_cagrisini_calistir("kosu_listesini_getir", {})
    assert isinstance(sonuc, dict)
    assert isinstance(sonuc["sonuc"], list)


def test_bildirim_ve_uygulama_birebir():
    assert {b["name"] for b in semalar.ARAC_BILDIRIMLERI} == set(ajan.ARAC_UYGULAMALARI)


# --- Cikti yolu sozlesmesi -------------------------------------------------

def test_ajan_ciktisi_tek_atislik_denemeyi_ezmez():
    """Ajan ciktisi, run_gemini_trial.py'nin dosyasindan FARKLI olmalidir.

    Onceki surum ikisini de gemini_response.json'a yaziyordu; iki deneme
    birbirini siliyor ve karsilastirilamaz hale geliyordu.
    """
    assert ajan.VARSAYILAN_CIKTI.name != "gemini_response.json"
    assert ajan.VARSAYILAN_CIKTI != ajan.VARSAYILAN_LOG


# --- Arac kullanim ozeti ---------------------------------------------------

def _kayit(*araclar_):
    return {"arac_cagrilari": [{"tur": 1, "arac": a, "argumanlar": {}, "hata": None} for a in araclar_]}


def test_arac_ozeti_sayimlari_dogru():
    kayitlar = {
        "kosu_01": _kayit("kosu_metriklerini_getir"),
        "kosu_02": _kayit("kosu_metriklerini_getir", "boyut_bazli_recall_getir"),
    }
    ozet = ajan.arac_kullanim_ozeti(kayitlar)
    assert ozet["arac_cagri_sayisi"]["kosu_metriklerini_getir"] == 2
    assert ozet["kosu_basina_cagri"] == {"kosu_01": 1, "kosu_02": 2}


def test_ozet_kirilim_araci_kullanimini_olcer():
    """Projenin merkezi sorusu: ajan dogru kanita yonelebiliyor mu?"""
    kayitlar = {
        "kosu_01": _kayit("kosu_metriklerini_getir"),
        "kosu_02": _kayit("sinif_karisikligini_getir"),
        "kosu_03": _kayit("kaynak_bazli_recall_getir"),
        "kosu_04": _kayit("baseline_farkini_getir"),
    }
    ozet = ajan.arac_kullanim_ozeti(kayitlar)
    assert ozet["kirilim_araci_kullanan_kosular"] == ["kosu_02", "kosu_03"]
    assert ozet["kirilim_araci_kullanim_orani"] == 0.5


def test_ozet_bos_kayitta_patlamaz():
    ozet = ajan.arac_kullanim_ozeti({})
    assert ozet["kirilim_araci_kullanim_orani"] == 0.0


def test_sistem_talimati_kirilim_araclarini_anlatir():
    """Talimat, toplam metriklerin bozulmayi gizleyebilecegini soylemelidir."""
    talimat = ajan.SISTEM_TALIMATI
    for arac in ("boyut_bazli_recall_getir", "kaynak_bazli_recall_getir",
                 "sinif_karisikligini_getir"):
        assert arac in talimat, f"{arac} sistem talimatinda anilmiyor"
    assert "GIZLEYEBILIR" in talimat

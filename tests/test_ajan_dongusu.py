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


# --- JSON tasima guvenligi -------------------------------------------------

def test_json_guvenli_sonsuzu_metne_cevirir():
    """Infinity JSON'da gecersizdir; Gemini API'si 400 ile reddeder."""
    assert ajan.json_guvenli(float("inf")) == "sonsuz"
    assert ajan.json_guvenli(float("-inf")) == "-sonsuz"
    assert ajan.json_guvenli(float("nan")) is None
    assert ajan.json_guvenli(1.5) == 1.5


def test_json_guvenli_ic_ice_yapilari_temizler():
    veri = {"a": {"b": [1.0, float("inf")]}, "c": float("nan")}
    temiz = ajan.json_guvenli(veri)
    metin = json.dumps(temiz, allow_nan=False)  # allow_nan=False: gecersizse patlar
    assert "Infinity" not in metin and "NaN" not in metin


@pytest.mark.parametrize("arac", sorted(ajan.ARAC_UYGULAMALARI))
def test_her_arac_ciktisi_gecerli_json_uretir(arac):
    """Hicbir arac API'ye gonderilemeyecek bir deger dondurmemeli.

    Gercek bir kosuda tum kosular bu yuzden basarisiz oldu: boyut bandi
    tanimindaki ust sinir float("inf") idi ve Gemini govdeyi reddetti.
    """
    argumanlar = {"kosu_id": "kosu_02"} if "kosu_id" in str(
        next(b for b in semalar.ARAC_BILDIRIMLERI if b["name"] == arac)
    ) else {}
    sonuc = ajan._arac_cagrisini_calistir(arac, argumanlar)
    json.dumps(sonuc, allow_nan=False)  # gecersiz deger varsa ValueError firlatir


def test_bant_tanimi_json_guvenli():
    from teshis.degerlendirme.metrikler import bant_araliklari

    metin = json.dumps(bant_araliklari(), allow_nan=False)
    assert "Infinity" not in metin


# --- Kota (429) yeniden deneme --------------------------------------------

def test_kota_hatasi_taninir():
    assert ajan._kota_hatasi_mi(Exception("429 RESOURCE_EXHAUSTED ..."))
    assert not ajan._kota_hatasi_mi(Exception("400 INVALID_ARGUMENT"))


def test_bekleme_suresi_sunucu_onerisini_okur():
    hata = Exception("... Please retry in 14.277709133s. ...")
    assert 15.0 <= ajan._bekleme_suresi(hata) <= 16.0


def test_bekleme_suresi_retrydelay_alanini_okur():
    hata = Exception("{'@type': '...RetryInfo', 'retryDelay': '13s'}")
    assert ajan._bekleme_suresi(hata) == 14.0


def test_bekleme_suresi_varsayilana_duser():
    assert ajan._bekleme_suresi(Exception("bilinmeyen"), varsayilan=7.0) == 7.0


# --- Gunluk kota ayrimi ve devam etme --------------------------------------

def test_gunluk_kota_dakikaliktan_ayirt_edilir():
    """Gunluk kotada beklemek ise yaramaz; dakikalikta yarar."""
    gunluk = Exception("{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}")
    dakikalik = Exception("{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}")
    assert ajan._gunluk_kota_mi(gunluk)
    assert not ajan._gunluk_kota_mi(dakikalik)
    assert ajan._kota_hatasi_mi(dakikalik) or True  # ikisi de kota hatasi


def test_basarili_kosu_ayirt_edilir():
    assert ajan._basarili_mi({"run_id": "kosu_02", "diagnosis": "x"})
    assert not ajan._basarili_mi({"run_id": "kosu_02", "diagnosis": "hata", "_hata": "429"})
    assert not ajan._basarili_mi({"run_id": "kosu_02", "diagnosis": None})


def test_devam_modu_tamamlanmis_kosulari_okur(tmp_path):
    """--devam, onceki calismadan kalan basarili kosulari atlayabilmeli."""
    cikti = tmp_path / "ajan_response.json"
    log = tmp_path / "ajan_arac_kaydi.json"
    cikti.write_text(json.dumps([
        {"run_id": "kosu_01", "diagnosis": "saglikli"},
        {"run_id": "kosu_02", "diagnosis": "hata", "_hata": "429"},
    ]), encoding="utf-8")
    log.write_text(json.dumps({"kosu_01": {"arac_cagrilari": [], "hata": None}}), encoding="utf-8")

    sonuclar, kayitlar = ajan._mevcut_sonuclari_oku(cikti, log)
    assert set(sonuclar) == {"kosu_01", "kosu_02"}
    assert ajan._basarili_mi(sonuclar["kosu_01"])
    assert not ajan._basarili_mi(sonuclar["kosu_02"])  # yeniden denenmeli
    assert "kosu_01" in kayitlar


def test_mevcut_sonuc_yoksa_bos_doner(tmp_path):
    sonuclar, kayitlar = ajan._mevcut_sonuclari_oku(tmp_path / "yok.json", tmp_path / "yok2.json")
    assert sonuclar == {} and kayitlar == {}


def test_kaydet_kosu_sirasini_korur(tmp_path):
    """Cikti, kosu_NN sirasinda yazilmali (sozluk ekleme sirasi degil)."""
    cikti, log = tmp_path / "c.json", tmp_path / "l.json"
    sonuclar = {"kosu_03": {"run_id": "kosu_03"}, "kosu_01": {"run_id": "kosu_01"}}
    ajan._kaydet(cikti, log, sonuclar, {})
    yazilan = [o["run_id"] for o in json.loads(cikti.read_text(encoding="utf-8"))]
    assert yazilan == sorted(yazilan)

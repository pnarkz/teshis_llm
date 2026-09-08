"""Canli ajan: korluk, on kontrol ve hata dayanikliligi.

Sunumda canli ajan mentorlerin onunde calisacak. Uc sey garanti edilmeli:

1. **Korluk bozulmuyor** - cevap anahtari ve senaryo adi ajana gitmiyor.
2. **On kontrol dogru soyluyor** - eksik kosul varsa hangisi oldugu belli.
3. **Basarisiz cagri uygulamayi cokertmiyor** - her hata turu ayri ayri
   yorumlaniyor ve kayitli moda gecis oneriliyor.

Bu testlerin hicbiri API cagrisi yapmaz; anahtar gerektirmez.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

import ajan_katmani  # noqa: E402


# --- Korluk -----------------------------------------------------------------

def test_arac_ciktilarinda_senaryo_adi_gecmiyor():
    """Araclarin dondurdugu hicbir deger gercek senaryo adini tasimamali."""
    import csv
    import json

    from teshis.ajan import araclar

    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        adlar = {r["scenario"] for r in csv.DictReader(f)}
    # Kod harfleri (D1, E4...) metrik adlarinda gecmez ama tam senaryo
    # adlari ("v00_saglikli", "E4 imgsz512") gecmemeli.
    yasak = {a for a in adlar if len(a) > 4}

    kosu_id = "kosu_08"
    ciktilar = []
    for arac in ("kosu_metriklerini_getir", "baseline_farkini_getir",
                 "boyut_bazli_recall_getir", "kaynak_bazli_recall_getir"):
        ciktilar.append(json.dumps(getattr(araclar, arac)(kosu_id),
                                   ensure_ascii=False))
    ciktilar.append(json.dumps(araclar.baseline_metriklerini_getir(),
                               ensure_ascii=False))
    metin = " ".join(ciktilar)
    sizan = sorted(a for a in yasak if a in metin)
    assert not sizan, f"Arac ciktisinda senaryo adi sizmis: {sizan}"


def test_cevap_anahtari_arac_katmaninda_yok():
    """Cevap anahtari ajan cevabindan SONRA, ayri bir yerel islemle kullanilir."""
    kaynak = (ROOT / "teshis/ajan/araclar.py").read_text(encoding="utf-8")
    assert "answer_key" not in kaynak
    assert "SENARYO_BEKLENEN" not in kaynak


def test_kaynak_adlari_takma_adla_veriliyor():
    """Kaynak grubu gercek adiyla verilse senaryo tahmini kolaylasirdi."""
    from teshis.ajan import araclar

    kaynaklar = araclar.kaynak_bazli_recall_getir("kosu_08")["kaynaklar"]
    for ad in kaynaklar:
        assert ad.startswith("kaynak_"), ad


def test_arayuz_secicisi_gercek_adi_gostermiyor():
    """Kosu secicisi yalnizca kosu_NN gostermeli; ad 'gercegi goster' ile acilir."""
    kaynak = (ROOT / "demo/bolumler/ajan.py").read_text(encoding="utf-8")
    secici = kaynak[kaynak.index('st.selectbox(\n            "Koşu"'):]
    secici = secici[: secici.index(")\n    with b:")]
    assert "senaryo" not in secici.lower(), (
        "Kosu secicisinde gercek senaryo adi kullaniliyor olabilir"
    )
    # GERCEK HATA: oneri aciklamalari bir sure secicide yaziyordu
    # ("belirgin bir bozulma...") ve senaryo adi acilmadan izleyiciye
    # bozulma olup olmadigini sizdiriyordu. Korluk yalnizca ajan icin
    # degil salondaki herkes icin gecerli olmali.
    for etiket in ajan_katmani.ONERILEN_CANLI.values():
        assert etiket not in secici, (
            "Oneri aciklamasi secicide gorunuyor; gercegi sizdirir"
        )


def test_onerilen_kosu_etiketleri_senaryo_adi_icermiyor():
    """Oneri etiketleri gercegi ele vermemeli - izleyici de kor kalmali."""
    import csv

    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        adlar = {r["scenario"] for r in csv.DictReader(f)}
    for kosu_id, etiket in ajan_katmani.ONERILEN_CANLI.items():
        assert kosu_id.startswith("kosu_")
        for ad in adlar:
            if len(ad) > 4:
                assert ad not in etiket, f"{kosu_id} etiketi '{ad}' iceriyor"


# --- On kontrol -------------------------------------------------------------

def test_on_kontrol_anahtarsiz_da_calisiyor(monkeypatch):
    """On kontrol API cagrisi yapmaz; anahtar yoksa da sonuc uretir."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    kontroller = ajan_katmani.on_kontrol("kosu_08")
    assert kontroller
    anahtar = next(k for k in kontroller if k["ad"] == "API anahtarı")
    assert anahtar["tamam"] is False
    assert not ajan_katmani.hazir_mi(kontroller)


def test_on_kontrol_anahtar_degerini_sizdirmiyor(monkeypatch):
    """Anahtarin KENDISI hicbir metne girmemeli."""
    monkeypatch.setenv("GEMINI_API_KEY", "gizli-anahtar-degeri-12345")
    metin = " ".join(
        f"{k['ad']} {k['not']}" for k in ajan_katmani.on_kontrol("kosu_08")
    )
    assert "gizli-anahtar-degeri-12345" not in metin


def test_bilinmeyen_kosu_on_kontrolde_yakalaniyor():
    kontroller = ajan_katmani.on_kontrol("kosu_9999")
    kimlik = next(k for k in kontroller if k["ad"] == "Koşu kimliği")
    assert kimlik["tamam"] is False


def test_model_adi_ajan_modulununkiyle_ayni():
    """Iki yerde farkli model adi olursa on kontrol yanlis sey dogrular."""
    kaynak = (ROOT / "teshis/ajan/ajan.py").read_text(encoding="utf-8")
    assert f'"{ajan_katmani.VARSAYILAN_MODEL}"' in kaynak


def test_model_adi_ortam_degiskeniyle_degistirilebiliyor(monkeypatch):
    monkeypatch.setenv("TESHIS_AJAN_MODEL", "baska-model")
    assert ajan_katmani.model_adi() == "baska-model"


# --- Hata dayanikliligi -----------------------------------------------------

@pytest.mark.parametrize(("mesaj", "beklenen"), [
    ("429 RESOURCE_EXHAUSTED quota exceeded", "KOTA"),
    ("503 UNAVAILABLE: model is overloaded", "GECICI"),
    ("Read timed out after 60s", "ZAMAN_ASIMI"),
    ("getaddrinfo failed", "BAGLANTI"),
    ("cevapta gecerli JSON bulunamadi", "GECERSIZ_JSON"),
    ("UNAUTHENTICATED: invalid api key", "YETKI"),
    ("bir sey patladi", "BILINMEYEN"),
])
def test_hata_turleri_ayirt_ediliyor(mesaj: str, beklenen: str):
    """GERCEK HATA: 503 bir donem "kota bitti" diye raporlaniyordu ve
    kullaniciya "yarin tekrar dene" deniyordu - oysa birkac saniye beklemek
    yetiyordu."""
    assert ajan_katmani.hata_turu(RuntimeError(mesaj))["tur"] == beklenen


def test_her_hata_turu_kullaniciya_ne_yapacagini_soyluyor():
    for mesaj in ("429 quota", "503 unavailable", "timed out",
                  "connection refused", "invalid json", "boom"):
        bilgi = ajan_katmani.hata_turu(RuntimeError(mesaj))
        assert bilgi["baslik"] and bilgi["mesaj"]
        assert bilgi["seviye"] in ("error", "warning")


def test_sema_dogrulamasi_eksik_alani_yakaliyor():
    """Canli cevap yapisal olarak dogrulanir."""
    eksik = ajan_katmani.sonuc_dogrula({"diagnosis": "x"})
    assert eksik, "eksik alanli cevap uyari uretmeliydi"


def test_canli_hata_kayitli_moda_yonlendiriyor():
    """Basarisiz canli cagri sonrasi kullanici cikmaza girmemeli."""
    kaynak = (ROOT / "demo/bolumler/ajan.py").read_text(encoding="utf-8")
    assert "Kayıtlı koşu" in kaynak
    assert "canlı sonuç değildir" in kaynak

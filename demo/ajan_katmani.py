"""Canli ajan icin saglik kontrolu, hata siniflandirmasi ve saglayici ayari.

Neden ayri dosya
----------------
Sunumda canli ajan mentorlerin onunde calisacak. O anda "anahtar yok",
"paket kurulu degil", "kota bitti", "503" gibi durumlarin hepsi ayni
gorunurse hicbiri anlatilamaz. Bu modul her durumu AYIRIR ve arayuz her
biri icin ne soylemesi gerektigini bilir.

API anahtari hicbir yerde - ekranda, logda, hata mesajinda - GORUNMEZ.
Yalnizca "tanimli / tanimli degil" bilgisi disari cikar.

Saglayici ve model adi ortam degiskeniyle degistirilebilir; saglayiciya
ozel kod `teshis/ajan/ajan.py` icinde kalir, burasi yalnizca on kontrol ve
hata yorumlamasi yapar.
"""

from __future__ import annotations

import os
from typing import Any

# Saglayici ayari: kod degistirmeden model degistirebilmek icin.
# Ajan katmaninin kendi varsayilaniyla AYNI olmali; iki yerde
# farkli model adi olursa on kontrol "gecerli" der ama calisan
# baska bir modeldir.
VARSAYILAN_MODEL = "gemini-3.6-flash"
ANAHTAR_DEGISKENI = "GEMINI_API_KEY"


def model_adi() -> str:
    return os.environ.get("TESHIS_AJAN_MODEL") or VARSAYILAN_MODEL


def anahtar_var_mi() -> bool:
    """Yalnizca VARLIK bilgisi doner - degerin kendisi asla disari cikmaz."""
    return bool(os.environ.get(ANAHTAR_DEGISKENI))


def on_kontrol(kosu_id: str) -> list[dict[str, Any]]:
    """Canli calistirmadan ONCE sessizce dogrulanan kosullar.

    Her satir: {ad, tamam, not}. Hepsi tamamsa canli mod guvenle
    baslatilabilir; degilse hangi kosulun eksik oldugu tek bakista gorunur.
    """
    kontroller: list[dict[str, Any]] = []

    kontroller.append({
        "ad": "API anahtarı",
        "tamam": anahtar_var_mi(),
        "not": ("ortam değişkeninde tanımlı" if anahtar_var_mi()
                else f"{ANAHTAR_DEGISKENI} tanımlı değil"),
    })

    try:
        import google.genai  # noqa: F401
        paket, paket_notu = True, "google-genai kurulu"
    except Exception as hata:  # noqa: BLE001
        paket, paket_notu = False, f"google-genai yüklenemedi ({type(hata).__name__})"
    kontroller.append({"ad": "Sağlayıcı paketi", "tamam": paket, "not": paket_notu})

    kontroller.append({
        "ad": "Model ayarı",
        "tamam": bool(model_adi()),
        "not": f"`{model_adi()}` (TESHIS_AJAN_MODEL ile değiştirilebilir)",
    })

    try:
        from teshis.ajan import araclar

        harita = araclar.anonim_kosu_haritasi()
        var = kosu_id in harita or kosu_id == "kosu_01"
        kontroller.append({
            "ad": "Koşu kimliği",
            "tamam": var,
            "not": (f"`{kosu_id}` araç katmanında bulundu" if var
                    else f"`{kosu_id}` ajana verilen koşular arasında yok"),
        })

        taban = araclar.baseline_metriklerini_getir()
        kontroller.append({
            "ad": "Referans metrikleri",
            "tamam": bool(taban) and "hata" not in taban,
            "not": "sağlıklı referans okunabiliyor",
        })

        eksik = []
        for arac, arg in (
            ("kosu_metriklerini_getir", {"kosu_id": kosu_id}),
            ("boyut_bazli_recall_getir", {"kosu_id": kosu_id}),
            ("kaynak_bazli_recall_getir", {"kosu_id": kosu_id}),
        ):
            try:
                sonuc = getattr(araclar, arac)(**arg)
                if isinstance(sonuc, dict) and sonuc.get("hata"):
                    eksik.append(arac)
            except Exception:  # noqa: BLE001
                eksik.append(arac)
        kontroller.append({
            "ad": "Kırılım dosyaları",
            "tamam": not eksik,
            "not": ("hepsi okunabiliyor" if not eksik
                    else "eksik: " + ", ".join(eksik)),
        })
    except Exception as hata:  # noqa: BLE001
        kontroller.append({
            "ad": "Araç katmanı",
            "tamam": False,
            "not": f"{type(hata).__name__}: {hata}",
        })
    return kontroller


def hazir_mi(kontroller: list[dict[str, Any]]) -> bool:
    return all(k["tamam"] for k in kontroller)


# Hata turleri BILEREK ayristirilir. Bir donem 503 (gecici sunucu hatasi)
# "kota bitti" diye raporlaniyordu ve kullaniciya "yarin tekrar dene"
# deniyordu - oysa birkac saniye beklemek yetiyordu.
def hata_turu(hata: BaseException) -> dict[str, str]:
    metin = f"{type(hata).__name__}: {hata}".lower()
    if any(k in metin for k in ("quota", "resource_exhausted", "429")):
        return {
            "tur": "KOTA",
            "baslik": "Günlük kota bitti",
            "mesaj": ("429 RESOURCE_EXHAUSTED — günlük istek kotası aşıldı. "
                      "Ücretsiz katman 20 istek/gün ve 5 istek/dk ile "
                      "sınırlıdır. Kayıtlı koşu modu çalışmaya devam eder."),
            "seviye": "error",
        }
    if any(k in metin for k in ("503", "unavailable", "high demand", "overloaded")):
        return {
            "tur": "GECICI",
            "baslik": "Geçici sunucu hatası",
            "mesaj": ("503 UNAVAILABLE — sağlayıcıda geçici yoğunluk. Bu hata "
                      "kotayla ilgili DEĞİLDİR; birkaç saniye sonra yeniden "
                      "denenebilir."),
            "seviye": "warning",
        }
    if any(k in metin for k in ("timeout", "timed out", "deadline")):
        return {
            "tur": "ZAMAN_ASIMI",
            "baslik": "Zaman aşımı",
            "mesaj": ("Sağlayıcı ayrılan süre içinde cevap vermedi. Kayıtlı "
                      "koşu moduna geçilebilir."),
            "seviye": "warning",
        }
    if any(k in metin for k in ("connection", "network", "dns", "ssl",
                                "unreachable", "getaddrinfo")):
        return {
            "tur": "BAGLANTI",
            "baslik": "Bağlantı hatası",
            "mesaj": ("Sağlayıcıya ulaşılamadı. İnternet bağlantısını kontrol "
                      "edin; kayıtlı koşu modu çevrimdışı çalışır."),
            "seviye": "warning",
        }
    if any(k in metin for k in ("json", "gecerli json", "decode")):
        return {
            "tur": "GECERSIZ_JSON",
            "baslik": "Cevap ayrıştırılamadı",
            "mesaj": ("Model geçerli JSON döndürmedi. Bu bir ölçüm hatası "
                      "değil, biçim hatasıdır; yeniden denemek genellikle "
                      "yeterlidir."),
            "seviye": "warning",
        }
    if "api_key" in metin or "api key" in metin or "unauthenticated" in metin:
        return {
            "tur": "YETKI",
            "baslik": "Kimlik doğrulama hatası",
            "mesaj": ("Sağlayıcı isteği kabul etmedi. Ortam değişkenindeki "
                      "anahtarın geçerli olduğunu kontrol edin."),
            "seviye": "error",
        }
    return {
        "tur": "BILINMEYEN",
        "baslik": "Başarısız",
        "mesaj": f"{type(hata).__name__}: {hata}",
        "seviye": "error",
    }


# Sunum icin onerilen canli test kosulari. Etiketler yalnizca ARAYUZDE
# gorunur; ajana gonderilen istekte yer almaz - yoksa korluk bozulurdu.
ONERILEN_CANLI = {
    "kosu_08": "belirgin bir bozulma — ajanın doğru teşhis ettiği koşu",
    "kosu_11": "kontrol koşusu — hiçbir bozulma yok",
    "kosu_06": "sınırlı kanıtlı, daha zor koşu",
}


def sonuc_dogrula(cevap: dict) -> list[str]:
    """Canli cevabin yapisal dogrulamasi - semalar.py'nin kurallariyla."""
    try:
        from teshis.ajan import semalar

        return semalar.teshis_dogrula(cevap) or []
    except Exception as hata:  # noqa: BLE001
        return [f"şema doğrulaması çalıştırılamadı: {type(hata).__name__}"]

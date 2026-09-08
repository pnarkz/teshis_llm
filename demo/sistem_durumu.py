"""Kenar cubugundaki kucuk sistem durumu: hangi cikti var, hangisi yok.

Amac sunum guvenligi. Sunumdan once "veri raporu var mi, hata galerileri
uretildi mi, API anahtari tanimli mi" sorularinin cevabini aramak yerine
tek bakista gormek gerekir. Bir sey eksikse sayfa sessizce bos kalmaz.

API anahtari HICBIR ZAMAN gosterilmez - yalnizca "tanimli / tanimli degil".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

KOK = Path(__file__).resolve().parents[1]


def _var(yol: str) -> bool:
    hedef = KOK / yol
    if hedef.is_dir():
        return any(hedef.iterdir())
    return hedef.is_file()


def durumlar() -> list[dict[str, Any]]:
    """Her satir: (ad, tamam mi, kisa aciklama)."""
    from data_loader import error_galleries, load_results
    from gorseller import durum as gorsel_durumu

    satirlar: list[dict[str, Any]] = []

    satirlar.append({
        "ad": "Veri raporu",
        "tamam": _var("reports/veri_raporu.json"),
        "not": "reports/veri_raporu.json",
    })

    try:
        n = len(load_results())
    except Exception:  # noqa: BLE001 - durum paneli akisi durdurmamali
        n = 0
    satirlar.append({
        "ad": "Deney kayıtları",
        "tamam": n > 0,
        "not": f"{n} koşu (results.csv)" if n else "results.csv okunamadı",
    })

    try:
        g = len(error_galleries())
    except Exception:  # noqa: BLE001
        g = 0
    satirlar.append({
        "ad": "Hata galerileri",
        "tamam": g > 0,
        "not": f"{g} galeri" if g else "üretilmemiş",
    })

    gd = gorsel_durumu()
    satirlar.append({
        "ad": "Etiketli görseller",
        "tamam": gd["kaynak"] != "yok",
        "not": {"tam": f"kilitli set, {gd['görüntü']} görüntü",
                "yedek": f"taşınabilir alt küme, {gd['görüntü']} görüntü",
                "yok": "bulunamadı"}[gd["kaynak"]],
    })

    satirlar.append({
        "ad": "Ajan denemesi",
        "tamam": _var("reports/ajan_denemesi/ajan_response.json"),
        "not": "kayıtlı koşular okunabiliyor",
    })

    anahtar = bool(os.environ.get("GEMINI_API_KEY"))
    satirlar.append({
        "ad": "LLM API anahtarı",
        "tamam": anahtar,
        "not": "tanımlı" if anahtar else "tanımlı değil (kayıtlı mod çalışır)",
    })

    try:
        import google.genai  # noqa: F401
        paket = True
    except Exception:  # noqa: BLE001 - kurulu degilse canli mod kapali
        paket = False
    satirlar.append({
        "ad": "Canlı ajan",
        "tamam": anahtar and paket,
        "not": ("kullanılabilir" if (anahtar and paket)
                else "google-genai yok" if not paket
                else "anahtar yok"),
    })
    return satirlar


def goster(st) -> None:
    """Kenar cubugunda sade bir liste; ana icerigin onune gecmez."""
    import stil

    with st.sidebar.expander("Sistem durumu", expanded=False):
        for d in durumlar():
            renk = stil.GUCLU if d["tamam"] else stil.UYARI
            isaret = "●" if d["tamam"] else "○"
            st.markdown(
                f'<div style="font-size:.8rem;line-height:1.7">'
                f'<span style="color:{renk}">{isaret}</span> '
                f'<span style="color:{stil.METIN}">{d["ad"]}</span> '
                f'<span style="color:{stil.METIN_SOLUK}">— {d["not"]}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

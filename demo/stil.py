"""Konsolun ortak gorsel dili ve kucuk gosterim yardimcilari.

Tasarim kararlari
-----------------
Onceki surum fosforlu terminal estetigi kullaniyordu (koyu zemin, neon yesil,
buyuk sayaclar). Bu, bir arastirma panelinden cok bir gosteriye benziyordu.
Yeni dil sakin ve akademik:

- kirik beyaz zemin, koyu gri metin
- TEK vurgu rengi; uyari ve olumlu durum icin ayri iki renk
- emoji yok, animasyon yok, pazarlama dili yok
- sayilar her zaman "deger + referansa fark" olarak
- her grafigin altinda tek cumlelik yorum

Renk anlamlari sabittir ve butun sayfalarda ayni kalir:
    VURGU   incelenen kosu
    NOTR    referans (v00) - her grafikte ayni renk
    UYARI   gurultu icinde kalan / zayif kanit
    OLUMLU  gurultu esigini belirgin asan kanit
"""

from __future__ import annotations

import streamlit as st

VURGU = "#1f4e79"     # incelenen kosu
NOTR = "#8a8a8a"      # referans
UYARI = "#a8611c"     # zayif / gurultu icinde
OLUMLU = "#2f6b4f"    # guclu kanit
ZEMIN = "#fbfaf8"
METIN = "#2b2b2b"
CIZGI = "#e3e0da"

CSS = f"""
<style>
  .stApp {{ background: {ZEMIN}; }}
  html, body, [class*="css"] {{ color: {METIN}; }}
  h1, h2, h3 {{ font-weight: 600; letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.55rem; }}
  h2 {{ font-size: 1.2rem; margin-top: 1.6rem; }}
  h3 {{ font-size: 1.02rem; margin-top: 1.1rem; }}
  hr {{ border: none; border-top: 1px solid {CIZGI}; margin: 1.2rem 0; }}
  .ust {{ font-size: .78rem; letter-spacing: .08em; text-transform: uppercase;
          color: #6f6f6f; margin-bottom: .15rem; }}
  .kutu {{ border: 1px solid {CIZGI}; border-radius: 3px; padding: .7rem .85rem;
           background: #fff; }}
  .kutu p {{ margin: 0; }}
  .rozet {{ display: inline-block; font-size: .74rem; padding: .12rem .5rem;
            border-radius: 2px; border: 1px solid; margin-right: .3rem; }}
  .r-uyari {{ color: {UYARI}; border-color: {UYARI}; background: #fdf6ef; }}
  .r-olumlu {{ color: {OLUMLU}; border-color: {OLUMLU}; background: #f1f6f3; }}
  .r-notr {{ color: #5f5f5f; border-color: {CIZGI}; background: #f7f6f4; }}
  .yorum {{ font-size: .84rem; color: #5f5f5f; margin-top: .3rem; }}
  .stDataFrame {{ font-size: .87rem; }}
  section[data-testid="stSidebar"] {{ background: #f4f2ee;
                                      border-right: 1px solid {CIZGI}; }}
</style>
"""


def uygula() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def ust_baslik(etiket: str) -> None:
    """Kucuk, buyuk harfli bolum etiketi."""
    st.markdown(f'<div class="ust">{etiket}</div>', unsafe_allow_html=True)


def rozet(metin: str, tur: str = "notr") -> str:
    """Durum rozeti; `tur`: olumlu | uyari | notr."""
    return f'<span class="rozet r-{tur}">{metin}</span>'


def kutu(icerik: str) -> None:
    st.markdown(f'<div class="kutu">{icerik}</div>', unsafe_allow_html=True)


def yorum(metin: str) -> None:
    """Grafik altina tek cumlelik okuma notu."""
    st.markdown(f'<div class="yorum">{metin}</div>', unsafe_allow_html=True)


def fark_metni(deger: float, fark: float, basamak: int = 4) -> str:
    """Sayilari her zaman 'deger (referansa fark)' olarak gosterir."""
    return f"{deger:.{basamak}f}  ({fark:+.{basamak}f})"


def guc_rozeti(seviye: str) -> str:
    tur = {"guclu": "olumlu", "zayif": "uyari", "gurultu icinde": "uyari"}.get(
        seviye, "notr"
    )
    return rozet(seviye, tur)

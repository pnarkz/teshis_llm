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


# --- Grafik yardimcilari ----------------------------------------------------

# Etki haritasinda renk olcegi bu degerde KIRPILIR. Kirpmadan cizildiginde
# birkac uc deger (orn. D5 last_pt'nin band orani ~97) butun olcegi eziyor ve
# geri kalan hucreler ayni acik tonda gorunuyordu. Kirpma yalnizca RENGI
# etkiler; gercek oran her zaman tooltip'te ve tablolarda tam degeriyle durur.
HARITA_RENK_TAVANI = 10.0


def etki_haritasi(veri, x: str, y: str, deger: str, baslik: str = ""):
    """Senaryo x metrik etki haritasi.

    Renk, farkin YONUNU degil BUYUKLUGUNU tasir: gurultu bandina orani.
    Ham farki renklendirmek yaniltici olurdu - kucuk bir grupta buyuk gorunen
    fark, o grubun dogal yayilimi icinde olabilir.
    """
    import altair as alt

    veri = veri.copy()
    veri["renk"] = veri[deger].clip(upper=HARITA_RENK_TAVANI)
    return (
        alt.Chart(veri)
        .mark_rect(stroke="#ffffff", strokeWidth=1)
        .encode(
            x=alt.X(f"{x}:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y}:N", title=None, sort=None),
            color=alt.Color(
                "renk:Q",
                title=f"band oranı (≥{HARITA_RENK_TAVANI:.0f} aynı ton)",
                scale=alt.Scale(scheme="oranges", domain=[0, HARITA_RENK_TAVANI]),
            ),
            tooltip=[c for c in veri.columns if c != "renk"],
        )
        .properties(height=max(240, 24 * veri[y].nunique()), title=baslik)
    )


def gurultu_bandi_grafigi(veri, senaryo: str = "senaryo", fark: str = "fark",
                          band: str = "band"):
    """Her senaryonun farkini, gurultu bandi kusagiyla birlikte cizer.

    Bandin icinde kalan noktalar gorsel olarak ayrisir; okuyucu "bu fark
    buyuk mu" sorusunu tabloya bakmadan cevaplar.
    """
    import altair as alt

    kusak = (
        alt.Chart(veri)
        .mark_area(opacity=0.22, color=NOTR)
        .encode(
            y=alt.Y(f"{senaryo}:N", title=None, sort=None),
            x=alt.X("band_alt:Q", title="referansa fark"),
            x2="band_ust:Q",
        )
    )
    noktalar = (
        alt.Chart(veri)
        .mark_point(size=90, filled=True)
        .encode(
            y=alt.Y(f"{senaryo}:N", title=None, sort=None),
            x=alt.X(f"{fark}:Q"),
            color=alt.Color(
                "asiyor:N",
                title=None,
                scale=alt.Scale(domain=["evet", "hayir"], range=[VURGU, UYARI]),
            ),
            tooltip=list(veri.columns),
        )
    )
    sifir = alt.Chart(veri).mark_rule(color="#bbb").encode(x=alt.datum(0))
    return (kusak + sifir + noktalar).properties(
        height=max(240, 24 * veri[senaryo].nunique())
    )


def guven_rozeti(deger: str) -> str:
    """Ajanin OZ-BILDIRDIGI guven duzeyi.

    Bilerek rozet, gauge degil: "yuksek" kalibre edilmis bir olasilik degil,
    modelin kendi beyanidir. Yarim daire bir gauge, olculmus bir guven
    yuzdesi izlenimi yaratirdi.
    """
    tur = {"yüksek": "olumlu", "yuksek": "olumlu",
           "orta": "notr", "düşük": "uyari", "dusuk": "uyari"}.get(
        str(deger).lower(), "notr"
    )
    return rozet(f"öz-bildirim: {deger}", tur)

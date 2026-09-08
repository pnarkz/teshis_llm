"""Butun grafiklerin tek kaynagi: tema, renk sozlesmesi, hover kurallari.

Neden ayri dosya
----------------
Grafikler bolumlere dagilmisken her sayfa kendi eksen/renk/tooltip
ayarlarini tekrar yaziyordu ve ayrisiyorlardi - "referans" bir sayfada
gri, digerinde maviydi. Bu projede tekrarlayan hata oruntusu tam olarak
budur: ayni kural iki yerde yasarsa biri geride kalir.

Kurallar
--------
- Renk semantiktir (stil.py'deki sozlesme), dekoratif degil.
- Ayrim asla YALNIZCA renge birakilmaz; sekil veya metin ikinci kanaldir.
- Her grafikte hover ile kesin deger okunabilir.
- Eksen kirpilmaz. Renk olceginde kirpma varsa gercek deger tooltip'te
  tam degeriyle durur ve baslikta kirpma acikca yazar.
"""

from __future__ import annotations

import altair as alt

import stil

# Renk olcegi bu degerde KIRPILIR. Kirpmadan cizildiginde birkac uc deger
# (orn. D5 last_pt'nin band orani ~97) butun olcegi eziyor ve geri kalan
# hucreler ayni tonda gorunuyordu. Kirpma yalnizca RENGI etkiler.
HARITA_RENK_TAVANI = 10.0

_EKSEN = alt.Axis(
    labelColor=stil.METIN_SOLUK, titleColor=stil.METIN_SOLUK,
    gridColor=stil.CIZGI, domainColor=stil.CIZGI, tickColor=stil.CIZGI,
    labelFontSize=11, titleFontSize=11,
)
_EFSANE = alt.Legend(
    labelColor=stil.METIN, titleColor=stil.METIN_SOLUK,
    labelFontSize=11, titleFontSize=11,
)


def _tema(grafik, yukseklik: int | None = None):
    """Ortak arka plan, yazi rengi ve olculer."""
    grafik = grafik.configure_view(
        strokeWidth=0, fill=stil.YUZEY,
    ).configure(background=stil.YUZEY).configure_title(
        color=stil.METIN, fontSize=13, anchor="start", dy=-6,
    )
    return grafik


def _katman_tema(grafik):
    """Katmanli/birlesik grafikler icin - configure yalnizca en ustte olur."""
    return grafik.configure_view(strokeWidth=0, fill=stil.YUZEY).configure(
        background=stil.YUZEY
    ).configure_title(color=stil.METIN, fontSize=13, anchor="start", dy=-6)


# --- Etki haritasi ----------------------------------------------------------

def etki_haritasi(veri, x: str, y: str, deger: str, baslik: str = ""):
    """Senaryo x metrik etki haritasi.

    Renk, farkin YONUNU degil BUYUKLUGUNU tasir: gurultu bandina orani.
    Ham farki renklendirmek yaniltici olurdu - kucuk bir grupta buyuk
    gorunen fark, o grubun dogal yayilimi icinde olabilir.
    """
    veri = veri.copy()
    veri["renk"] = veri[deger].clip(upper=HARITA_RENK_TAVANI)
    grafik = (
        alt.Chart(veri)
        .mark_rect(stroke=stil.ZEMIN, strokeWidth=2, cornerRadius=2)
        .encode(
            x=alt.X(f"{x}:N", title=None, axis=alt.Axis(
                labelAngle=0, labelColor=stil.METIN, domainColor=stil.CIZGI,
                tickColor=stil.CIZGI, labelFontSize=11)),
            y=alt.Y(f"{y}:N", title=None, sort=None, axis=alt.Axis(
                labelColor=stil.METIN, domainColor=stil.CIZGI,
                tickColor=stil.CIZGI, labelFontSize=11)),
            color=alt.Color(
                "renk:Q",
                title=f"band oranı (≥{HARITA_RENK_TAVANI:.0f} aynı ton)",
                scale=alt.Scale(
                    range=[stil.YUZEY_2, stil.ADAY, stil.UYARI, stil.KRITIK],
                    domain=[0, HARITA_RENK_TAVANI], type="linear",
                ),
                legend=_EFSANE,
            ),
            tooltip=[alt.Tooltip(c, title=c) for c in veri.columns if c != "renk"],
        )
        .properties(height=max(240, 26 * veri[y].nunique()), title=baslik)
    )
    return _tema(grafik)


# --- Gurultu bandi ----------------------------------------------------------

def gurultu_bandi_grafigi(veri, senaryo: str = "senaryo", fark: str = "fark"):
    """Her senaryonun farkini, gurultu bandi kusagiyla birlikte cizer.

    Bandin icinde kalan noktalar hem RENK hem SEKIL olarak ayrisir; okuyucu
    "bu fark buyuk mu" sorusunu tabloya bakmadan cevaplar.
    """
    kusak = (
        alt.Chart(veri)
        .mark_area(opacity=0.3, color=stil.NOTR)
        .encode(
            y=alt.Y(f"{senaryo}:N", title=None, sort=None, axis=_EKSEN),
            x=alt.X("band_alt:Q", title="referansa fark", axis=_EKSEN),
            x2="band_ust:Q",
        )
    )
    sifir = alt.Chart(veri).mark_rule(color=stil.CIZGI, strokeWidth=1.5).encode(
        x=alt.datum(0)
    )
    noktalar = (
        alt.Chart(veri)
        .mark_point(size=130, filled=True, strokeWidth=2)
        .encode(
            y=alt.Y(f"{senaryo}:N", title=None, sort=None, axis=_EKSEN),
            x=alt.X(f"{fark}:Q", axis=_EKSEN),
            color=alt.Color(
                "asiyor:N", title="eşiği aşıyor",
                scale=alt.Scale(domain=["evet", "hayir"],
                                range=[stil.ADAY, stil.UYARI]),
                legend=_EFSANE,
            ),
            shape=alt.Shape(
                "asiyor:N", title="eşiği aşıyor",
                scale=alt.Scale(domain=["evet", "hayir"],
                                range=["square", "circle"]),
                legend=_EFSANE,
            ),
            # Bandin ICINDE kalan noktalar acik dolgulu ama CIZGILI kalir.
            # Dolgu cok saydam yapildiginda o noktalar gri kusagin icinde
            # gorunmez oldu - "gurultu icinde" senaryolar ekrandan silinmis
            # gibi duruyordu.
            fillOpacity=alt.condition(
                "datum.asiyor == 'evet'", alt.value(1.0), alt.value(0.4)
            ),
            tooltip=[alt.Tooltip(c) for c in veri.columns],
        )
    )
    return _katman_tema(
        (kusak + sifir + noktalar).properties(
            height=max(240, 26 * veri[senaryo].nunique())
        )
    )


# --- Karsilastirma ----------------------------------------------------------

def gruplu_bar(veri, kategori: str, deger: str, seri: str,
               baslik: str = "", yatay: bool = False, alan_adi: str = ""):
    """Referans ve aday(lar) yan yana. Referans rengi HER SAYFADA aynidir."""
    seriler = list(dict.fromkeys(veri[seri].tolist()))
    renkler = _seri_renkleri(seriler)
    x = alt.X(f"{deger}:Q", title=alan_adi or None, axis=_EKSEN)
    y = alt.Y(f"{kategori}:N", title=None, sort=None, axis=_EKSEN)
    if not yatay:
        x, y = (alt.X(f"{kategori}:N", title=None, sort=None,
                      axis=alt.Axis(labelAngle=0, labelColor=stil.METIN_SOLUK,
                                    domainColor=stil.CIZGI, tickColor=stil.CIZGI)),
                alt.Y(f"{deger}:Q", title=alan_adi or None, axis=_EKSEN))
    grafik = (
        alt.Chart(veri)
        .mark_bar(cornerRadius=2)
        .encode(
            x=x, y=y,
            color=alt.Color(f"{seri}:N", title=None,
                            scale=alt.Scale(domain=seriler, range=renkler),
                            legend=_EFSANE),
            xOffset=f"{seri}:N" if not yatay else alt.Undefined,
            yOffset=f"{seri}:N" if yatay else alt.Undefined,
            tooltip=[alt.Tooltip(c) for c in veri.columns],
        )
        .properties(height=max(240, 26 * veri[kategori].nunique()) if yatay else 300,
                    title=baslik)
    )
    return _tema(grafik)


def _seri_renkleri(seriler: list[str]) -> list[str]:
    """Referans serisi NOTR, digerleri sirayla vurgu renkleri."""
    dongu = [stil.ADAY, stil.IKINCIL, stil.UYARI, stil.KRITIK]
    renkler, i = [], 0
    for s in seriler:
        if "referans" in str(s).lower() or str(s).startswith("v00"):
            renkler.append(stil.REFERANS)
        else:
            renkler.append(dongu[i % len(dongu)])
            i += 1
    return renkler


def fark_bar(veri, kategori: str, deger: str, esik_alani: str | None = None,
             baslik: str = ""):
    """Referansa gore fark; esigi asanlar vurgu, asmayanlar uyari rengi."""
    kosul = (f"abs(datum.{deger}) > datum.{esik_alani}" if esik_alani
             else f"datum.{deger} < 0")
    grafik = (
        alt.Chart(veri)
        .mark_bar(cornerRadius=2)
        .encode(
            y=alt.Y(f"{kategori}:N", title=None, sort="x", axis=_EKSEN),
            x=alt.X(f"{deger}:Q", title="referansa fark", axis=_EKSEN),
            color=alt.condition(kosul, alt.value(stil.ADAY), alt.value(stil.NOTR)),
            tooltip=[alt.Tooltip(c) for c in veri.columns],
        )
        .properties(height=max(200, 26 * veri[kategori].nunique()), title=baslik)
    )
    return _tema(grafik)


# --- Veri seti grafikleri ---------------------------------------------------

def yatay_bar(veri, kategori: str, deger: str, baslik: str = "",
              renk: str | None = None, alan_adi: str = ""):
    grafik = (
        alt.Chart(veri)
        .mark_bar(cornerRadius=2, color=renk or stil.ADAY)
        .encode(
            y=alt.Y(f"{kategori}:N", title=None, sort="-x", axis=_EKSEN),
            x=alt.X(f"{deger}:Q", title=alan_adi or None, axis=_EKSEN),
            tooltip=[alt.Tooltip(c) for c in veri.columns],
        )
        .properties(height=max(160, 30 * len(veri)), title=baslik)
    )
    return _tema(grafik)


def isi_haritasi(veri, x: str, y: str, deger: str, baslik: str = ""):
    grafik = (
        alt.Chart(veri)
        .mark_rect(stroke=stil.ZEMIN, strokeWidth=2, cornerRadius=2)
        .encode(
            x=alt.X(f"{x}:N", title=None, axis=alt.Axis(
                labelAngle=0, labelColor=stil.METIN, domainColor=stil.CIZGI,
                tickColor=stil.CIZGI)),
            y=alt.Y(f"{y}:N", title=None, axis=alt.Axis(
                labelColor=stil.METIN, domainColor=stil.CIZGI,
                tickColor=stil.CIZGI)),
            color=alt.Color(f"{deger}:Q", title=deger, legend=_EFSANE,
                            scale=alt.Scale(range=[stil.YUZEY_2, stil.ADAY,
                                                   stil.IKINCIL])),
            tooltip=[alt.Tooltip(c) for c in veri.columns],
        )
        .properties(height=max(200, 34 * veri[y].nunique()), title=baslik)
    )
    return _tema(grafik)


def cizgi(veri, x: str, y: str, seri: str | None = None, baslik: str = "",
          alan_adi: str = ""):
    """Egitim egrisi gibi zaman/epoch serileri."""
    kodlama = dict(
        x=alt.X(f"{x}:Q", title=x, axis=_EKSEN),
        y=alt.Y(f"{y}:Q", title=alan_adi or y, axis=_EKSEN),
        tooltip=[alt.Tooltip(c) for c in veri.columns],
    )
    if seri:
        seriler = list(dict.fromkeys(veri[seri].tolist()))
        kodlama["color"] = alt.Color(
            f"{seri}:N", title=None, legend=_EFSANE,
            scale=alt.Scale(domain=seriler, range=_seri_renkleri(seriler)),
        )
    grafik = (
        alt.Chart(veri).mark_line(strokeWidth=2, point=False)
        .encode(**kodlama).properties(height=280, title=baslik)
    )
    return _tema(grafik)

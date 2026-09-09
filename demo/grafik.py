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

def etki_haritasi(veri, x: str, y: str, deger: str, baslik: str = "",
                  yon: str | None = None):
    """Senaryo x metrik etki haritasi.

    Renk BUYUKLUGU tasir: gurultu bandina oran. Ham farki renklendirmek
    yaniltici olurdu - kucuk bir grupta buyuk gorunen bir fark, o grubun
    dogal yayilimi icinde olabilir.

    `yon` verilirse (farkin isaretini tasiyan sutun) YUKSELISLER ayri bir
    tonda cizilir. Bu bir dogruluk meselesi: harita 48 hucrenin 10'unda
    yukselis gosteriyor ve biri (D1 mAP50_95, +0.0240) esigi asiyor.
    Mutlak deger kullanildiginda o hucre, ayni buyuklukteki bir DUSUSLE
    ayni rengi aliyor ve "bozulma kaniti" gibi okunuyordu - hipotez
    tablosunda kapatilan D1 hatasinin haritadaki ikizi.
    """
    veri = veri.copy()
    veri["renk"] = veri[deger].clip(upper=HARITA_RENK_TAVANI)
    if yon is not None:
        # Yukselisler negatif tarafa tasinir; olcek iki yonlu olur.
        veri["renk"] = veri.apply(
            lambda r: r["renk"] if r[yon] < 0 else -r["renk"], axis=1)
        return _yonlu_harita(veri, x, y, baslik)
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

def _yonlu_harita(veri, x: str, y: str, baslik: str = ""):
    """Iki yonlu etki haritasi: dusus sicak, yukselis soguk tonda.

    Sifir civari notr yuzey rengidir; boylece gurultu icinde kalan hucreler
    (oran < 1) her iki yonde de sonuk gorunur ve goz once esigi asanlara
    gider.
    """
    return _tema(
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
                title="← yükseliş   band oranı   düşüş →",
                scale=alt.Scale(
                    range=[stil.IKINCIL, stil.YUZEY_2, stil.YUZEY_2,
                           stil.UYARI, stil.KRITIK],
                    domain=[-HARITA_RENK_TAVANI, -1, 1,
                            HARITA_RENK_TAVANI / 2, HARITA_RENK_TAVANI],
                    type="linear",
                ),
                # Yon isareti olcegin ICINDE tasinir ama efsanede BUYUKLUK
                # gorunur. Ham haliyle "-10" yaziyordu ve "eksi band orani"
                # diye okunuyordu - oysa oran her zaman pozitiftir, eksi
                # yalnizca "yukselis" demek.
                legend=alt.Legend(
                    labelColor=stil.METIN, titleColor=stil.METIN_SOLUK,
                    labelFontSize=11, titleFontSize=11,
                    labelExpr="abs(datum.value)",
                ),
            ),
            tooltip=[alt.Tooltip(c, title=c) for c in veri.columns if c != "renk"],
        )
        .properties(height=max(240, 26 * veri[y].nunique()), title=baslik)
    )


def gurultu_bandi_grafigi(veri, senaryo: str = "senaryo", fark: str = "fark"):
    """Her senaryonun farkini, gurultu bandi kusagiyla birlikte cizer.

    Bandin icinde kalan noktalar hem RENK hem SEKIL olarak ayrisir; okuyucu
    "bu fark buyuk mu" sorusunu tabloya bakmadan cevaplar.
    """
    # Kusak SATIR BASINA bir bar olarak cizilir. `mark_area` denendiginde
    # kategoriler arasi bir poligon olusuyordu: cok satirda dikey bir serit
    # gibi gorunuyor ama dort metrikli bir grafikte elmas seklinde bir alan
    # cikiyordu. Bar her satirin kendi bandini gosterir.
    kusak = (
        alt.Chart(veri)
        .mark_bar(opacity=0.35, color=stil.NOTR, height=26, cornerRadius=2)
        .encode(
            y=alt.Y(f"{senaryo}:N", title=None, sort=None, axis=_EKSEN),
            x=alt.X("band_alt:Q", title="referansa fark", axis=_EKSEN),
            x2="band_ust:Q",
            tooltip=[alt.Tooltip(c) for c in veri.columns],
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
              renk: str | None = None, alan_adi: str = "",
              etiket: str | bool = False, sirala: str | None = "-x"):
    """Yatay cubuk. `etiket` verilirse deger cubugun ucuna YAZILIR.

    Etiketli cubuk, yanina ayni sayilari tekrar eden bir tablo koyma
    ihtiyacini ortadan kaldirir. Bu sayfada grafik+tablo cifti uc kez
    yan yana duruyordu ve ikisi de ayni seyi soyluyordu; goz once cubugu
    okuyup sonra tabloda ayni sayiyi ariyordu.

    `etiket=True` deger sutununu ham haliyle yazar. Binlik ayraci onemliyse
    ONCEDEN bicimlenmis bir sutun adi verin: Vega'nin ",.0f" bicimi
    "131,700" yazar, bu konsolun geri kalani "131.700" kullanir ve ayni
    ekranda iki ayrac bulunmamalidir.
    """
    temel = alt.Chart(veri).encode(
        y=alt.Y(f"{kategori}:N", title=None, sort=sirala, axis=_EKSEN),
        x=alt.X(f"{deger}:Q", title=alan_adi or None, axis=_EKSEN),
        tooltip=[alt.Tooltip(c) for c in veri.columns],
    )
    cubuk = temel.mark_bar(cornerRadius=2, color=renk or stil.ADAY)
    if etiket:
        alan = deger if etiket is True else etiket
        cubuk = cubuk + temel.mark_text(
            align="left", dx=5, fontSize=11, color=stil.METIN,
        ).encode(text=alt.Text(f"{alan}:N" if alan != deger
                               else f"{alan}:Q", format=""
                               if alan != deger else ",.0f"))
    return _katman_tema(
        cubuk.properties(height=max(160, 32 * len(veri)), title=baslik)
    ) if etiket else _tema(
        cubuk.properties(height=max(160, 30 * len(veri)), title=baslik)
    )


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


# --- Karsilastirma grafikleri: grafik ANA ANLATICI --------------------------
#
# Bu bolumdeki grafikler metnin yerine gecer, ona eslik etmez. Her biri tek
# bir soruyu cevaplar ve altinda tek cumlelik bir okuma notu durur.

def degerli_gruplu_bar(veri, kategori: str, deger: str, seri: str,
                       baslik: str = "", alan_adi: str = "",
                       basamak: int = 3):
    """Referans ve aday yan yana; DEGERLER cubuklarin uzerinde yazili.

    Y ekseni sifirdan baslar (`zero=True`). Kirpilmis bir eksen kucuk
    farklari olduklarindan cok daha dramatik gosterir; bu sayfanin butun
    amaci farkin BUYUKLUGUNU dogru okutmak oldugu icin kabul edilemez.
    """
    seriler = list(dict.fromkeys(veri[seri].tolist()))
    renkler = _seri_renkleri(seriler)
    temel = alt.Chart(veri).encode(
        x=alt.X(f"{kategori}:N", title=None, sort=None,
                axis=alt.Axis(labelAngle=0, labelColor=stil.METIN,
                              domainColor=stil.CIZGI, tickColor=stil.CIZGI,
                              labelFontSize=12)),
        y=alt.Y(f"{deger}:Q", title=alan_adi or None, axis=_EKSEN,
                scale=alt.Scale(zero=True)),
        xOffset=alt.XOffset(f"{seri}:N", sort=seriler),
    )
    cubuk = temel.mark_bar(cornerRadius=2).encode(
        color=alt.Color(f"{seri}:N", title=None, legend=_EFSANE,
                        scale=alt.Scale(domain=seriler, range=renkler)),
        tooltip=[alt.Tooltip(c) for c in veri.columns],
    )
    etiket = temel.mark_text(dy=-8, color=stil.METIN, fontSize=11).encode(
        text=alt.Text(f"{deger}:Q", format=f".{basamak}f")
    )
    return _katman_tema((cubuk + etiket).properties(height=300, title=baslik))


def fark_profili(veri, kategori: str, fark: str, esik: str | None = None,
                 baslik: str = ""):
    """Sifir merkezli yatay fark grafigi - senaryonun METRIK IMZASI.

    Sol taraf dusus, sag taraf artis. Gurultu esigini asan cubuklar dolu,
    icinde kalanlar soluk cizilir; boylece "fark ne kadar" ve "fark anlamli
    mi" ayni bakista okunur.

    D2b icin ozellikle guclu: eksik etiket senaryosunda precision duserken
    recall ARTAR. Bu ters yon, bozulmanin karakterini dogrudan gosterir -
    model etiketsiz kalan nesneleri arka plan sanmiyor, fazladan kutu
    uretiyor.
    """
    kosul = f"abs(datum['{fark}']) > datum['{esik}']" if esik else "true"
    cubuk = (
        alt.Chart(veri)
        .mark_bar(cornerRadius=2, height=20)
        .encode(
            y=alt.Y(f"{kategori}:N", title=None, sort=None, axis=_EKSEN),
            x=alt.X(f"{fark}:Q", title="referansa fark", axis=_EKSEN,
                    scale=alt.Scale(zero=True)),
            color=alt.condition(f"datum['{fark}'] < 0",
                                alt.value(stil.KRITIK), alt.value(stil.GUCLU)),
            fillOpacity=alt.condition(kosul, alt.value(0.95), alt.value(0.3)),
            tooltip=[alt.Tooltip(c) for c in veri.columns],
        )
    )
    sifir = alt.Chart(veri).mark_rule(
        color=stil.METIN_SOLUK, strokeWidth=1.5
    ).encode(x=alt.datum(0))
    return _katman_tema(
        (cubuk + sifir).properties(
            height=max(200, 44 * veri[kategori].nunique()), title=baslik)
    )


def dumbbell(veri, kategori: str, referans: str, aday: str,
             baslik: str = "", alan_adi: str = ""):
    """Iki nokta ve aralarindaki cizgi - fark MESAFE olarak okunur.

    Gruplu cubuk grafiginde iki cubugun boyunu gozle kiyaslamak gerekir;
    dumbbell'da aradaki bosluk dogrudan farkin kendisidir. D4'te yalnizca
    <16 px bandindaki ayrisma bu yuzden aninda gorunur ve "toplam mAP neden
    sorunu gizledi?" sorusunun gorsel cevabi olur.

    Daire = saglikli referans, kare = senaryo (renk korlugu icin sekil de
    ayirt edici).
    """
    ipucu = [alt.Tooltip(c) for c in veri.columns]
    eksen_y = alt.Y(f"{kategori}:N", title=None, sort=None, axis=_EKSEN)
    cizgi_ = (
        alt.Chart(veri).mark_rule(strokeWidth=3, color=stil.CIZGI)
        .encode(y=eksen_y,
                x=alt.X(f"{referans}:Q", title=alan_adi or None, axis=_EKSEN,
                        scale=alt.Scale(zero=False, nice=True)),
                x2=f"{aday}:Q", tooltip=ipucu)
    )
    ref_nokta = (
        alt.Chart(veri)
        .mark_point(size=150, filled=True, color=stil.REFERANS, shape="circle")
        .encode(y=eksen_y, x=f"{referans}:Q", tooltip=ipucu)
    )
    aday_nokta = (
        alt.Chart(veri)
        .mark_point(size=160, filled=True, color=stil.ADAY, shape="square")
        .encode(y=eksen_y, x=f"{aday}:Q", tooltip=ipucu)
    )
    return _katman_tema(
        (cizgi_ + ref_nokta + aday_nokta).properties(
            height=max(180, 44 * veri[kategori].nunique()), title=baslik)
    )


def karisiklik_farki(veri, baslik: str = ""):
    """Senaryo eksi saglikli: hangi hucre arttı, hangisi azaldi.

    Iki matrisi yan yana koymak karsilastirmayi izleyiciye birakir; fark
    matrisi soruyu dogrudan cevaplar. Kirmizi = azalan, yesil = artan.

    Ultralytics'in PNG'sinden DEGIL, kirilim olcumundeki sayisal
    `karisiklik_matrisi` alanindan cizilir - o PNG bir kez kendi
    raporladigi recall ile celismisti (D3b).
    """
    return _tema(
        alt.Chart(veri)
        .mark_rect(stroke=stil.ZEMIN, strokeWidth=2, cornerRadius=2)
        .encode(
            x=alt.X("tahmin:N", title="model ne dedi", sort=None,
                    axis=alt.Axis(labelAngle=0, labelColor=stil.METIN,
                                  domainColor=stil.CIZGI,
                                  tickColor=stil.CIZGI)),
            y=alt.Y("gercek:N", title="gerçekte ne vardı", sort=None,
                    axis=alt.Axis(labelColor=stil.METIN,
                                  domainColor=stil.CIZGI,
                                  tickColor=stil.CIZGI)),
            color=alt.Color(
                "fark:Q", title="senaryo − sağlıklı", legend=_EFSANE,
                scale=alt.Scale(domainMid=0,
                                range=[stil.KRITIK, stil.YUZEY, stil.GUCLU]),
            ),
            tooltip=[alt.Tooltip(c) for c in veri.columns],
        )
        .properties(height=max(200, 48 * veri["gercek"].nunique()),
                    title=baslik)
    )


def egri_isaretli(veri, x: str, y: str, seri: str, isaretler,
                  baslik: str = "", alan_adi: str = ""):
    """Egri + uzerinde isaretli epoch'lar (orn. best.pt ve last.pt).

    E1'in anlatimi tam olarak buna dayanir: en iyi checkpoint saglikli
    gorunurken son checkpoint arizayi gosterir. Iki noktanin egri uzerinde
    NEREDE oldugunu gormek, iki sayiyi yan yana koymaktan daha anlasilir.

    `isaretler`: [{"epoch": n, "etiket": "best.pt"}, ...]
    """
    import pandas as _pd

    seriler = list(dict.fromkeys(veri[seri].tolist()))
    govde = (
        alt.Chart(veri).mark_line(strokeWidth=2)
        .encode(
            x=alt.X(f"{x}:Q", title=x, axis=_EKSEN),
            y=alt.Y(f"{y}:Q", title=alan_adi or y, axis=_EKSEN),
            color=alt.Color(f"{seri}:N", title=None, legend=_EFSANE,
                            scale=alt.Scale(domain=seriler,
                                            range=_seri_renkleri(seriler))),
            tooltip=[alt.Tooltip(c) for c in veri.columns],
        )
    )
    if not isaretler:
        return _katman_tema(govde.properties(height=300, title=baslik))

    nokta = _pd.DataFrame(isaretler)
    kural = (
        alt.Chart(nokta)
        .mark_rule(strokeDash=[4, 4], strokeWidth=1.5, color=stil.IKINCIL)
        .encode(x=alt.X(f"{x}:Q"), tooltip=[alt.Tooltip("etiket")])
    )
    yazi = (
        alt.Chart(nokta)
        .mark_text(align="left", dx=5, dy=-4, color=stil.IKINCIL, fontSize=11)
        .encode(x=alt.X(f"{x}:Q"), y=alt.value(10), text="etiket:N")
    )
    return _katman_tema(
        (govde + kural + yazi).properties(height=300, title=baslik)
    )


def dumbbell_bantli(veri, kategori: str, referans: str, aday: str,
                    band: str, vurgu: str, alan_adi: str = "",
                    etiket: str | None = None):
    """Gurultu bandi SERIT olarak cizilmis dumbbell.

    Duz dumbbell "aday referanstan ne kadar uzak" sorusunu cevaplar ama
    "bu uzaklik anlamli mi" sorusunu izleyiciye birakir. Bu projede ikinci
    soru birincisinden onemli: bir farki bozulmaya baglamadan once, hicbir
    bozulma icermeyen kosular arasindaki yayilimi asmasi gerekir.

    Serit = referans +/- o grubun gurultu bandi. Seridin ICINDE kalan bir
    nokta, saf rastgelelikten ayirt edilemez. Boylece "genel skor kaybi
    gizleyebilir" ve "once gurultuyu olctuk" ayni karede okunur.

    `vurgu` sutunu True olan satir tam renkte, digerleri soluk cizilir:
    ana bulgu one cikarken kiyas satirlari da gorunur kalir.
    """
    ipucu = [alt.Tooltip(c) for c in veri.columns if c != vurgu]
    eksen_y = alt.Y(f"{kategori}:N", title=None, sort=None, axis=_EKSEN)
    veri = veri.copy()
    veri["_alt"] = veri[referans] - veri[band]
    veri["_ust"] = veri[referans] + veri[band]

    # Serit once cizilir; noktalar ustune biner.
    # Eksen: veriye gore daraltilir ama oran alaninda 1.0'i ASMAZ ve
    # tik sayisi sinirlanir. Varsayilan olcek 0.05 adimlarla 19 etiket
    # basiyordu; okunacak sey noktalarin YERI, eksenin kendisi degil.
    en_dusuk = float(min(veri[aday].min(), veri["_alt"].min()))
    en_yuksek = float(max(veri[referans].max(), veri[aday].max(),
                          veri["_ust"].max()))
    pay = max((en_yuksek - en_dusuk) * 0.08, 0.01)
    alan = alt.Scale(domain=[max(0.0, en_dusuk - pay), min(1.0, en_yuksek + pay)],
                     zero=False, nice=False, clamp=True)
    eksen_x = alt.Axis(
        labelColor=stil.METIN_SOLUK, titleColor=stil.METIN_SOLUK,
        gridColor=stil.CIZGI, domainColor=stil.CIZGI, tickColor=stil.CIZGI,
        labelFontSize=11, titleFontSize=11, tickCount=6, format=".2f",
    )
    serit = (
        alt.Chart(veri).mark_bar(height=18, color=stil.CIZGI, opacity=0.85)
        .encode(y=eksen_y,
                x=alt.X("_alt:Q", title=alan_adi or None, axis=eksen_x,
                        scale=alan),
                x2="_ust:Q", tooltip=ipucu)
    )
    saydam = alt.condition(f"datum.{vurgu}", alt.value(1.0), alt.value(0.4))
    cizgi_ = (
        alt.Chart(veri).mark_rule(strokeWidth=3, color=stil.KRITIK)
        .encode(y=eksen_y, x=f"{referans}:Q", x2=f"{aday}:Q",
                opacity=saydam, tooltip=ipucu)
    )
    ref_nokta = (
        alt.Chart(veri)
        .mark_point(size=150, filled=True, color=stil.REFERANS, shape="circle")
        .encode(y=eksen_y, x=f"{referans}:Q", opacity=saydam, tooltip=ipucu)
    )
    aday_nokta = (
        alt.Chart(veri)
        .mark_point(size=170, filled=True, color=stil.ADAY, shape="square")
        .encode(y=eksen_y, x=f"{aday}:Q", opacity=saydam, tooltip=ipucu)
    )
    katmanlar = [serit, cizgi_, ref_nokta, aday_nokta]
    if etiket:
        katmanlar.append(
            alt.Chart(veri[veri[vurgu]])
            .mark_text(align="left", dx=12, dy=-16, fontSize=12,
                       fontWeight="bold", color=stil.KRITIK)
            .encode(y=eksen_y, x=f"{aday}:Q", text=f"{etiket}:N")
        )
    return _katman_tema(
        alt.layer(*katmanlar).properties(
            height=max(190, 46 * veri[kategori].nunique()))
    )

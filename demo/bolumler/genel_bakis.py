"""Genel Bakis: sunumun ilk dakikasi.

Bu sayfanin isi izleyiciyi UC seye hazirlamak: hangi soru soruluyor, sistem
nasil calisiyor, ve neden onemli. Bunun otesindeki her sey baska bir sayfaya
aittir.

Onceki hali bir rapor gibi kurgulanmisti: alti gosterge, alti adimli sema,
butun kosularin etki haritasi, kanit gucu dagilimi, derecelendirilmeyenler
tablosu ve uc ajan skoru - hepsi ayni gorsel agirlikta. Ilk grafige varmadan
once epey okuma gerekiyordu ve izleyici neye bakacagini bilemiyordu.

Simdi tek bir ana grafik var ve o grafik projenin iki iddiasini birden
tasiyor: kirilim gizli kaybi acar, ve bir fark ancak gurultuyu asarsa etkidir.

Sayfa iki olcekte okunur: once TEK bir ornek (D4, kirilim + gurultu bandi),
sonra BUTUN kosularin etki haritasi. Ozelden genele; harita tek basina
birakilsaydi izleyici neye bakacagini bilemezdi.

Tasinanlar: kanit gucu dagilimi -> Sonuclar, ajan puanlama olcutleri ->
Ajan. Uc "ana bulgu" ve derecelendirilmeyenler tablosu ise Sonuclar'da
zaten vardi; buradakiler kopyaydi ve silindi.

Butun sayilar kaynaktan turetilir. Sabit yazilmis tek sey yoktur - "24 kosu"
gibi bir sayi bir donem burada yaziyordu ve defter buyudukce geride kaldi.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import grafik
import katalog
import stil
from data_loader import ajan_deneyi, ajan_deneyi_kosu_bazli, load_results
from teshis.degerlendirme.karsilastirilabilirlik import bozulmasiz_mi, kimlik
from teshis.degerlendirme.senaryo_ozeti import ne_gozlendi

# Ana bulgu olarak D4 secildi: bozulma TEK bir alt grupta yogunlasiyor ve
# genel metrikte kaybolmuyor - ikisi ayni karede gorunuyor.
ANA_SENARYO = "D4"


def _kontrol_sayisi(sonuclar: pd.DataFrame) -> int:
    return sum(
        1 for _, r in sonuclar.iterrows()
        if str(r["scenario"]).startswith("C") and str(r["scenario"])[1:2].isdigit()
    )


# --- Ana grafik -------------------------------------------------------------

def _ana_bulgu_verisi(senaryo: str = ANA_SENARYO) -> pd.DataFrame | None:
    """Nesne boyutu bandi basina recall: saglikli referans, senaryo, gurultu bandi.

    Band, o grupta hicbir bozulma icermeyen kosular arasinda gozlenen en
    buyuk yayilimdir. Grafikte serit olarak cizilir; seridin icinde kalan
    bir nokta saf rastgelelikten ayirt edilemez.
    """
    import senaryo_grafikleri as sg
    from teshis.degerlendirme import gurultu

    veri = sg.boyut_verisi(senaryo)
    if veri is None or veri.empty:
        return None

    import veri_seti as vs

    ters = {ad: kod for kod, ad in vs.BANT_ADI.items()}
    satirlar = []
    for _, r in veri.iterrows():
        kod = ters.get(r["grup"], r["grup"])
        d = gurultu.fark_degerlendir("boyut_bandi_recall", kod, float(r["fark"]))
        band = d.get("band")
        if not band:
            continue
        oran = abs(float(r["fark"])) / band
        satirlar.append({
            "nesne boyutu": r["grup"],
            "sağlıklı referans": float(r["sağlıklı"]),
            senaryo: float(r["senaryo"]),
            "fark": float(r["fark"]),
            "gürültü bandı": band,
            "band oranı": round(oran, 2),
            "gerçek kutu": r.get("gerçek kutu"),
            "_vurgu": oran >= 1.0,
            "_etiket": f"gürültü bandının {oran:.0f} katı" if oran >= 1.0 else "",
        })
    return pd.DataFrame(satirlar) if satirlar else None


def _ana_grafik(senaryo: str = ANA_SENARYO) -> None:
    veri = _ana_bulgu_verisi(senaryo)
    if veri is None:
        return

    vurgulu = veri[veri["_vurgu"]]
    stil.ust_baslik(f"örnek bulgu · {senaryo}")
    st.markdown(
        "#### Genel skor, küçük nesnelerdeki kaybı gizleyebilir",
        help=None,
    )
    st.altair_chart(
        grafik.dumbbell_bantli(
            veri, "nesne boyutu", "sağlıklı referans", senaryo,
            "gürültü bandı", "_vurgu", alan_adi="recall (yakalama oranı)",
            etiket="_etiket",
        ),
        width="stretch",
    )

    if not vurgulu.empty:
        r = vurgulu.iloc[0]
        st.markdown(
            f"Gri şerit, o boyut grubunda **hiçbir bozulma içermeyen** koşular "
            f"arasında görülen yayılımdır. {senaryo}'te yalnızca "
            f"**{r['nesne boyutu']}** grubu şeridin dışına çıkıyor: recall "
            f"{r['sağlıklı referans']:.4f} → {r[senaryo]:.4f} "
            f"(**{r['band oranı']:.0f}×** bant). Diğer gruplar şeridin içinde, "
            "yani o farklar rastgelelikten ayırt edilemiyor."
        )
    stil.yorum(
        f"Bu tek senaryonun sonucudur, bütün senaryoların özeti değildir. "
        f"Senaryo bazlı sonuçlar Karşılaştırma ve Sonuçlar sayfalarında."
    )


METRIK_ADI = {
    "mAP50": "mAP50", "mAP50_95": "mAP50-95",
    "precision": "precision", "recall": "recall",
}


def _etki_verisi(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Senaryo x metrik: farkin gurultu bandina orani ve YONU.

    Ham fark yerine ORANA bakilir: kucuk bir grupta buyuk gorunen bir fark,
    o grubun dogal yayilimi icinde olabilir.

    "fark" sutunu isaretiyle birlikte tasinir ve harita onu renklendirmede
    kullanir. Onceden yalnizca mutlak deger vardi; 48 hucrenin 10'u yukselis
    ve biri (D1 mAP50_95, +0.0240, oran 1.19) esigi asiyordu - yani ayni
    buyuklukteki bir DUSUSLE ayni rengi alip bozulma kaniti gibi
    okunuyordu.
    """
    satirlar = []
    for _, r in sonuclar.iterrows():
        senaryo = str(r["scenario"])
        # Bozulmasiz kosular haritaya GIRMEZ. Girdiklerinde - ki bir sure
        # girdiler - saglikli referans ve kontrol kosulari "etki" gosteren
        # renkli hucreler olarak cikiyordu. Kontrol kosusunun band orani,
        # kendi bandindan cikarilmis olmasinin bir yan urunudur; bozulma
        # olcusu degildir.
        if bozulmasiz_mi(senaryo):
            continue
        g = ne_gozlendi(senaryo)
        if not g:
            continue
        for metrik, d in g["metrikler"].items():
            esik = d["gurultu_esigi"]
            if not esik or d["fark"] is None:
                continue
            satirlar.append({
                "senaryo": senaryo,
                "metrik": METRIK_ADI.get(metrik, metrik),
                "band oranı": round(abs(d["fark"]) / esik, 2),
                "yön": "düşüş" if d["fark"] < 0 else "yükseliş",
                "değer": d["deger"],
                "referans": d["referans"],
                "fark": d["fark"],
                "gürültü eşiği": esik,
                "kanıt": "eşiği aşıyor" if d["asiyor"] else "gürültü içinde",
            })
    return pd.DataFrame(satirlar)


def _etki_haritasi_bolumu(sonuclar: pd.DataFrame) -> None:
    etki = _etki_verisi(sonuclar)
    if etki.empty:
        return
    st.markdown("### Bütün koşuların etki haritası")
    st.markdown(
        "Her hücre, o senaryonun o metrikteki farkının **gürültü bandına "
        "oranıdır**. 1'in altı, farkın hiçbir bozulma içermeyen koşular "
        "arasında da görüldüğü anlamına gelir."
    )
    st.altair_chart(
        grafik.etki_haritasi(etki, x="metrik", y="senaryo",
                            deger="band oranı", yon="fark"),
        width="stretch",
    )
    yukselen = etki[(etki["yön"] == "yükseliş") & (etki["band oranı"] >= 1)]
    stil.yorum(
        "Kırmızıya doğru giden hücreler bandın belirgin üzerinde bir "
        "<b>düşüş</b>; camgöbeği hücreler bandı aşan bir <b>yükseliş</b>. "
        "Ortadaki sönük ton gürültü içinde kalan farklardır. Renk ölçeği "
        f"{grafik.HARITA_RENK_TAVANI:.0f}× oranında kırpılır — gerçek oran "
        "her zaman hover'da tam değeriyle durur."
        + (f"<br><b>Yükselen {len(yukselen)} hücre var</b> ("
           + ", ".join(f"{r['senaryo']} {r['metrik']}"
                       for _, r in yukselen.iterrows())
           + "): eşiği aşıyorlar ama bozulma kanıtı değiller — metrik "
             "beklenenin tersine yükselmiş." if not yukselen.empty else "")
    )


# --- Surec semasi -----------------------------------------------------------

_ASAMALAR = [
    ("1", "Koşulu değiştir", "Kontrollü arıza",
     "Veri veya eğitim ayarında <b>her seferinde tek değişken</b>. "
     "Protokolün geri kalanı sabit: aynı seed, aynı çözünürlük."),
    ("2", "Değişimi ölç", "Genel + kırılımlı metrikler",
     "Sağlıklı referansla karşılaştırılır. Kırılım: sınıf, nesne boyutu, "
     "veri kaynağı. Fark, <b>gürültü bandıyla birlikte</b> raporlanır."),
    ("3", "Nedeni araştır", "Kör LLM teşhisi",
     "Ajan araçları çağırıp kanıt toplar, sonra teşhisini yazar. "
     "Cevap anahtarıyla <b>sonradan</b>, ayrı bir işlemde puanlanır."),
]


def _asama_semasi() -> str:
    """Uc asama - SVG degil HTML: tema degisince birlikte doner.

    Onceki alti kutuluk sema, kutulara dagitilmis bir paragraf gibi
    okunuyordu. Uc asama, izleyicinin akilda tutabilecegi sayida.
    """
    kutular = []
    for no, etiket, baslik, metin in _ASAMALAR:
        ok = (f'<div style="align-self:center;color:{stil.ADAY};'
              f'font-size:1.3rem;padding:0 .2rem">→</div>' if no != "1" else "")
        kutular.append(
            f'<div style="display:flex;flex:1 1 260px;min-width:240px;gap:.4rem">'
            f"{ok}"
            f'<div style="flex:1;border:1px solid {stil.CIZGI};border-radius:10px;'
            f'background:{stil.YUZEY};padding:.75rem .85rem">'
            f'<div style="font-size:.7rem;color:{stil.ADAY};letter-spacing:.1em;'
            f'text-transform:uppercase">{no} · {etiket}</div>'
            f'<div style="font-weight:600;font-size:1rem;color:{stil.METIN};'
            f'margin:.25rem 0 .35rem">{baslik}</div>'
            f'<div style="font-size:.78rem;color:{stil.METIN_SOLUK};'
            f'line-height:1.5">{metin}</div></div></div>'
        )
    return ('<div style="display:flex;flex-wrap:wrap;gap:.5rem;align-items:stretch">'
            + "".join(kutular) + "</div>")


# --- Sayfa ------------------------------------------------------------------

def goster() -> None:
    import veri_seti as vs

    sonuclar = load_results()
    tani = vs.tani_seti()

    st.title("Termal Teşhis Ajanı")
    st.markdown(
        "Termal nesne tespitindeki performans değişimlerini **kontrollü "
        "deneylerle** ölçen ve nedenlerini **kör bir LLM ajanıyla** araştıran "
        "deney sistemi."
    )

    st.write("")
    olculebilir = sum(
        1 for _, r in sonuclar.iterrows() if kimlik(str(r["scenario"])) is not None
    )
    deney = ajan_deneyi()
    kartlar = [
        # Alt metin OLCULEBILEN sayiyi da verir: Karsilastirma seciciside
        # kosusu olmayan senaryolar gorunmez ve iki sayfa arasindaki fark
        # aciklanmadan kalirdi.
        ("Araştırma kapsamı", len(katalog.senaryolar()),
         f"senaryo · {len([x for x in katalog.senaryolar() if x['ana_kosu']])}"
         "'ü ölçülebildi"),
        ("Değerlendirme", olculebilir,
         f"koşu · {_kontrol_sayisi(sonuclar)} kontrol dahil"),
        ("Kilitli tanı seti",
         f"{tani['goruntu_sayisi']:,}".replace(",", ".")
         if tani.get("goruntu_sayisi") else "kayıtta yok",
         f"görüntü · {tani.get('bbox_sayisi', 0):,} nesne".replace(",", ".")
         if tani.get("bbox_sayisi") else "künye okunamadı"),
    ]
    if deney:
        p = deney["puan"]
        kartlar.append((
            "Ajan deneyi", p["gozlem"],
            f"gözlem · {p['kosu']} koşu × "
            f"{p['gozlem'] // max(p['kosu'], 1)} tekrar",
        ))
    stil.kpi_satiri(kartlar)
    stil.yorum(
        "Kapsam, ölçülen senaryo sayısını gösterir; hepsinin bulgu ürettiği "
        "anlamına gelmez. <b>Test seti hiç kullanılmadı</b> — bütün ölçümler "
        "kilitli tanı setinde yapıldı."
    )

    st.markdown("---")
    st.markdown(_asama_semasi(), unsafe_allow_html=True)
    stil.kutu(
        "<b>Ajan görüntülere bakmıyor.</b> Gördüğü tek şey ölçüm çıktıları: "
        "genel metrikler ve kırılım tabloları, anonim <code>koşu_NN</code> "
        "kimliğiyle. Senaryo adı, bozulma açıklaması ve cevap anahtarı ona "
        "hiçbir biçimde gönderilmez."
    )

    st.markdown("---")
    _ana_grafik()

    st.markdown("---")
    _etki_haritasi_bolumu(sonuclar)

    if deney:
        st.markdown("---")
        _kapanis(deney)


def _kapanis(deney: dict) -> None:
    """Iki sayi. Ucuncusu eklenirse hicbiri akilda kalmaz."""
    rol = (deney["puan"].get("rol_bazli") or {}).get("kontrol") or {}
    kosular = ajan_deneyi_kosu_bazli()
    tutarli = [k for k in kosular if k["hukum_tutarli"]]

    a, b = st.columns(2)
    with a:
        if rol:
            stil.kpi(
                "Kontrol koşularında uydurma",
                f"{rol['gozlem'] - round(rol['dogru_teshis'] * rol['gozlem'])}"
                f" / {rol['gozlem']}",
                "hiçbir bozulma içermeyen koşularda yanlış teşhis",
            )
    with b:
        if kosular:
            stil.kpi(
                "Tekrar tutarlılığı",
                f"{len(tutarli)} / {len(kosular)}",
                f"koşu, {kosular[0]['tekrar']} tekrarında da aynı hükmü verdi",
            )
    st.caption(
        "Ajanın senaryo bazlı başarısı tek bir orana indirgenmez: kontrol "
        "koşuları \"sorun uyduruyor mu\", bozulma senaryoları \"nedeni "
        "bulabiliyor mu\" sorusunu ölçer. Ayrıntı: LLM Teşhis Ajanı sayfası."
    )

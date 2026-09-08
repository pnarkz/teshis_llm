"""Genel Bakis: proje ne soruyor, nasil calisiyor, ne buldu.

Sayfa uzun bir README gibi degil, bir gozlem panosu gibi kurgulanmistir:
once ucten dorde sayi, sonra sistemin uctan uca sureci, sonra etki haritasi.

Skorlarin adlandirilmasina ozellikle dikkat edilir. `mean_score` bir rubrik
ortalamasidir ve iki bileseni (kanit, sinirlama) her kosuda tam puan aldigi
icin yuksek gorunur. Tek basina verilirse "ajan senaryolarin %83'unu dogru
bildi" diye okunur; gercek teshis dogrulugu %50'dir. Bu yuzden ayristirilarak
gosterilir.

Butun sayilar kaynaktan turetilir. Sabit yazilmis tek sey yoktur - "24 kosu"
gibi bir sayi bir donem burada yaziyordu ve defter buyudukce geride kaldi.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import grafik
import stil
from data_loader import ajan_kaydi, error_galleries, load_results
from teshis.degerlendirme.karsilastirilabilirlik import bozulmasiz_mi, kimlik
from teshis.degerlendirme.senaryo_ozeti import kanit_gucu, ne_gozlendi

METRIK_ADI = {
    "mAP50": "mAP50", "mAP50_95": "mAP50-95",
    "precision": "precision", "recall": "recall",
}


def _etki_verisi(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Senaryo × metrik: farkın gürültü bandına oranı.

    Ham fark yerine **orana** bakılır: küçük bir grupta büyük görünen bir
    fark, o grubun doğal yayılımı içinde olabilir.
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
                "değer": d["deger"],
                "referans": d["referans"],
                "fark": d["fark"],
                "gürültü eşiği": esik,
                "kanıt": "eşiği aşıyor" if d["asiyor"] else "gürültü içinde",
            })
    return pd.DataFrame(satirlar)


def _guc_dagilimi(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Yalnizca DERECELENDIRILEBILEN kosularin dagilimi.

    Kontrol kosulari, referanslar, eslenik olcumler ve esigi olmayan kosular
    bu dagilima girmez - girseydi "guclu bulgu" sayisi, hicbir bozulma
    icermeyen kosularla sisirilirdi.
    """
    sayim: dict[str, int] = {}
    for _, r in sonuclar.iterrows():
        seviye = kanit_gucu(str(r["scenario"]))["seviye"]
        if seviye in stil.DERECELENDIRILEN:
            sayim[seviye] = sayim.get(seviye, 0) + 1
    return pd.DataFrame(
        {"koşu": [sayim.get(s, 0) for s in stil.DERECELENDIRILEN]},
        index=[stil.seviye_adi(s) for s in stil.DERECELENDIRILEN],
    )


def _derecelendirilemeyen(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Derecelendirilmeyen kosular ve NEDEN derecelendirilmedikleri."""
    satirlar = []
    for _, r in sonuclar.iterrows():
        ad = str(r["scenario"])
        g = kanit_gucu(ad)
        if g["seviye"] in stil.DERECELENDIRILEN:
            continue
        o = ne_gozlendi(ad)
        satirlar.append({
            "koşu": ad,
            "durum": stil.seviye_adi(g["seviye"]),
            "referansı": o.get("referans_senaryo") or "—",
        })
    return pd.DataFrame(satirlar)


def _kontrol_sayisi(sonuclar: pd.DataFrame) -> int:
    return sum(
        1 for _, r in sonuclar.iterrows()
        if str(r["scenario"]).startswith("C") and str(r["scenario"])[1:2].isdigit()
    )


def _ajan_skorlari(ajan: dict) -> dict[str, float]:
    """Rubrik ortalamasini bilesenlerine ayirir.

    Tek sayi vermek yaniltici: kanit ve sinirlama bilesenleri her kosuda tam
    puan aliyor, dolayisiyla ortalama teshis dogrulugundan cok daha yuksek
    cikiyor.
    """
    kosular = list(ajan.get("puanlar", {}).values())
    if not kosular:
        return {}
    n = len(kosular)
    return {
        "teshis": sum(k["diagnosis_score"] for k in kosular) / n,
        "teshis_tespit": sum(k["diagnosis_score_tespit"] for k in kosular) / n,
        "kanit": sum(k["evidence_score"] for k in kosular) / n,
        "sinir": sum(k["limitation_score"] for k in kosular) / n,
        "rubrik": ajan.get("ozet", {}).get("mean_score"),
    }


# --- Surec semasi -----------------------------------------------------------

_ADIMLAR = [
    # Ilk adimin alt metni tani setinden TURETILIR; buradaki deger yalnizca
    # kunye hic okunamazsa gorunur ve sayi icermez.
    ("Termal veri", "kilitli tanı seti"),
    ("Kontrollü arıza", "veri veya eğitim ayarında<br>her seferinde TEK değişken"),
    ("YOLO eğitimi /<br>değerlendirmesi", "sabit protokol<br>aynı seed, aynı çözünürlük"),
    ("Genel + kırılımlı<br>metrikler", "sınıf, nesne boyutu,<br>veri kaynağı"),
    ("Ajan araçlarla<br>kanıt topluyor", "senaryo adını görmez,<br>yalnızca <code>kosu_NN</code>"),
    ("Teşhis + kanıt<br>+ sınırlama", "cevap anahtarıyla<br>SONRADAN puanlanır"),
]


def _surec_semasi(tani: dict) -> str:
    """Uctan uca akis - SVG degil, HTML kutular: tema degisince birlikte doner."""
    adimlar = list(_ADIMLAR)
    if tani.get("goruntu_sayisi"):
        adimlar[0] = (
            "Termal veri",
            f"{tani['goruntu_sayisi']:,}".replace(",", ".")
            + " görüntü, "
            + f"{tani.get('bbox_sayisi', 0):,}".replace(",", ".")
            + " bbox<br>kilitli tanı seti",
        )
    # Ok, kendinden SONRAKI kutuyla ayni ogenin icinde durur. Ayri bir oge
    # olsaydi satir sonunda bosa isaret eden bir ok kalirdi (flex-wrap).
    kutular = []
    for i, (baslik, alt) in enumerate(adimlar):
        ok = (f'<div style="align-self:center;color:{stil.ADAY};'
              f'font-size:1.05rem;padding:0 .1rem">→</div>' if i else "")
        kutular.append(
            f'<div style="display:flex;flex:1 1 190px;min-width:170px;gap:.35rem">'
            f"{ok}"
            f'<div style="flex:1;border:1px solid {stil.CIZGI};'
            f'border-radius:8px;background:{stil.YUZEY};padding:.6rem .7rem">'
            f'<div style="font-size:.68rem;color:{stil.ADAY};letter-spacing:.08em">'
            f'ADIM {i + 1}</div>'
            f'<div style="font-weight:600;font-size:.88rem;color:{stil.METIN};'
            f'line-height:1.3;margin:.15rem 0 .25rem">{baslik}</div>'
            f'<div style="font-size:.74rem;color:{stil.METIN_SOLUK};'
            f'line-height:1.45">{alt}</div></div></div>'
        )
    return ('<div style="display:flex;flex-wrap:wrap;gap:.5rem;align-items:stretch">'
            + "".join(kutular) + "</div>")


def goster() -> None:
    import veri_seti as vs

    sonuclar = load_results()
    ajan = ajan_kaydi()
    tani = vs.tani_seti()

    st.title("Termal Teşhis Ajanı")
    st.markdown(
        "Termal drone görüntüleriyle çalışan bir YOLO nesne tespit modeli "
        "**kontrollü biçimde bozulur**, bozulmanın metriklere nasıl yansıdığı "
        "ölçülür; sonra bir LLM ajanına bu ölçümler anonim olarak verilerek "
        "nedeni **kanıta dayalı** teşhis edip edemediği sınanır."
    )
    stil.kutu(
        "<b>Araştırma sorusu:</b> Bir dil modeli, yalnızca ölçüm çıktılarına "
        "bakarak bir nesne tespit modelindeki bozulmanın <i>nedenini</i> "
        "ayırt edebilir mi — ve ürettiği gerekçe savunulabilir mi?"
    )

    st.write("")
    olculebilir = sum(
        1 for _, r in sonuclar.iterrows() if kimlik(str(r["scenario"])) is not None
    )
    stil.kpi_satiri([
        ("Değerlendirme koşusu", olculebilir, "defterde kayıtlı"),
        ("Kontrol koşusu", _kontrol_sayisi(sonuclar), "yalnızca seed farklı"),
        ("Kilitli tanı seti",
         f"{tani['goruntu_sayisi']:,}".replace(",", ".")
         if tani.get("goruntu_sayisi") else "kayıtta yok",
         f"{tani.get('bbox_sayisi', 0):,} bbox".replace(",", ".")
         if tani.get("bbox_sayisi") else "künye okunamadı"),
        ("Hata galerisi", len(error_galleries()), "koşu başına örnek incelemesi"),
        ("Ajan denemesi", len(ajan.get("cevaplar") or {}), "kör teşhis koşusu"),
        ("Test seti kullanımı", "YOK", "final aşamasına kadar yasak"),
    ])

    st.markdown("---")
    st.markdown("## Sistem nasıl çalışıyor")
    st.markdown(_surec_semasi(tani), unsafe_allow_html=True)
    stil.yorum(
        "Zincirin kritik yeri 5. adım: ajana senaryo adı, bozulma açıklaması "
        "ve cevap anahtarı gönderilmez. Puanlama, ajan cevabını ürettikten "
        "SONRA ayrı bir yerel işlemle yapılır."
    )

    st.markdown("---")
    st.markdown("## Etki haritası")
    st.markdown(
        "Her hücre, o senaryonun o metrikteki farkının **gürültü bandına "
        "oranıdır**. 1'in altı, farkın hiçbir bozulma içermeyen koşular "
        "arasında da görüldüğü anlamına gelir."
    )
    etki = _etki_verisi(sonuclar)
    if not etki.empty:
        st.altair_chart(
            grafik.etki_haritasi(etki, x="metrik", y="senaryo", deger="band oranı"),
            width="stretch",
        )
        stil.yorum(
            "Koyu hücreler bandın belirgin üzerinde; açık hücreler gürültüden "
            "ayırt edilemiyor. Hücrenin üzerine gelince gerçek değer, "
            "referans, fark ve eşik görünür. Renk ölçeği "
            f"{grafik.HARITA_RENK_TAVANI:.0f}× oranında kırpılır — gerçek oran "
            "her zaman hover'da tam değeriyle durur."
        )

    e, f = st.columns([1, 2])
    with e:
        stil.ust_baslik("kanıt gücü dağılımı")
        st.bar_chart(_guc_dagilimi(sonuclar), height=210, color=stil.ADAY)
        derece_disi = _derecelendirilemeyen(sonuclar)
        stil.yorum(
            f"Yalnızca kendi ölçeğinde referansı VE gürültü eşiği olan "
            f"{int(_guc_dagilimi(sonuclar)['koşu'].sum())} koşu derecelendirilir. "
            f"Kalan {len(derece_disi)} koşu aşağıda, nedeniyle birlikte."
        )
        with st.expander("Derecelendirilmeyen koşular"):
            st.dataframe(derece_disi, hide_index=True, width="stretch")
    with f:
        stil.ust_baslik("üç ana bulgu")
        for baslik, metin in _ana_bulgular(sonuclar):
            with st.expander(baslik, expanded=False):
                st.markdown(metin)

    st.markdown("---")
    st.markdown("## Ajan")
    skor = _ajan_skorlari(ajan)
    if skor:
        g, h, i = st.columns(3)
        with g:
            stil.kpi("Doğru neden teşhisi", f"%{skor['teshis'] * 100:.0f}",
                     "asıl performans ölçüsü")
        with h:
            stil.kpi("Tespit-farkındalıklı",
                     f"%{skor['teshis_tespit'] * 100:.0f}",
                     "bozulmanın izi yoksa ceza yok")
        with i:
            stil.kpi("Rubrik ortalaması", f"%{skor['rubrik'] * 100:.0f}",
                     "üç bileşenin ortalaması")
        stil.kutu(
            "<b>Bu üç sayı aynı şeyi ölçmez.</b> Rubrik ortalaması üç "
            f"bileşenin ortalamasıdır ve ikisi doymuştur: kanıt "
            f"%{skor['kanit'] * 100:.0f}, sınırlama %{skor['sinir'] * 100:.0f} "
            "— her koşuda tam puan. Ayırt eden tek bileşen teşhistir. Yani "
            f"ajan senaryoların %{skor['rubrik'] * 100:.0f}'ini <i>bilmedi</i>; "
            f"doğru nedeni bulma oranı %{skor['teshis'] * 100:.0f}."
        )


def _ana_bulgular(sonuclar: pd.DataFrame) -> list[tuple[str, str]]:
    """Uc ana bulgu; sayilari olcumden gelir."""
    deger = {str(r["scenario"]): r for _, r in sonuclar.iterrows()}

    def m(ad, alan="mAP50"):
        return float(deger[ad][alan]) if ad in deger else None

    bulgular = []

    g = ne_gozlendi("E4 imgsz512")
    if g:
        bulgular.append((
            "Bozulmanın türü metrik imzasından okunabiliyor",
            "Çıkarım çözünürlüğü uyumsuzluğu recall'u çökertiyor "
            f"({g['metrikler']['recall']['fark']:+.4f}) ama etiket "
            "bozulmaları precision'ı da bozuyor. Hangi metriğin bozulduğu "
            "arızanın türünü söylüyor — bu, ajanın teşhis için kullandığı "
            "asıl sinyal.",
        ))

    if m("E1") is not None and m("v00_saglikli last_pt") is not None:
        bulgular.append((
            "Standart raporlama bir arızayı tamamen gizleyebiliyor",
            "E1'de 200 epoch süren ders kitabı niteliğinde bir aşırı uyum "
            "elde edildi. En iyi checkpoint ile raporlandığında model "
            f"sağlıklı görünüyor (mAP50 farkı {m('E1') - m('v00_saglikli'):+.4f}). "
            f"Arıza son checkpoint'te ortaya çıkıyor: E1 best'ten last'a "
            f"{m('E1 last_pt') - m('E1'):+.4f} düşerken sağlıklı referans "
            f"yalnızca {m('v00_saglikli last_pt') - m('v00_saglikli'):+.4f} "
            "düşüyor. Yani düşüşün kendisi değil, tabandan ne kadar ayrıldığı "
            "anlamlı.",
        ))

    g = ne_gozlendi("D1")
    if g:
        bulgular.append((
            "Gürültü ölçülmeden \"etki\" denemez",
            "Aynı veri ve protokolle, yalnızca rastgelelik tohumu "
            "değiştirilerek eğitilen modeller arasında bile belirgin fark "
            f"var: recall'da {g['metrikler']['recall']['gurultu_esigi']:.4f}, "
            f"mAP50'de {g['metrikler']['mAP50']['gurultu_esigi']:.4f}. Bu "
            "taban ölçülünce bir dizi iddia zayıfladı ve bir senaryo (D6b) "
            "bulgu olmaktan çıktı.",
        ))
    return bulgular

"""Genel Bakış: proje ne soruyor, ne buldu, hangi bulgu ne kadar sağlam.

Metin yığını yerine bir **etki haritası** ile açılır: hangi senaryonun hangi
metrikte gürültü bandını aştığı bir bakışta görünür.

Skorların adlandırılmasına özellikle dikkat edilir. `mean_score` bir rubrik
ortalamasıdır ve iki bileşeni (kanıt, sınırlama) her koşuda tam puan aldığı
için yüksek görünür. Tek başına verilirse "ajan senaryoların %83'ünü doğru
bildi" diye okunur; gerçek teşhis doğruluğu %50'dir. Bu yüzden ayrıştırılarak
gösterilir.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import stil
from data_loader import ajan_kaydi, error_galleries, load_results
from teshis.degerlendirme.senaryo_ozeti import kanit_gucu, ne_gozlendi

METRIK_ADI = {
    "mAP50": "mAP50",
    "mAP50_95": "mAP50-95",
    "precision": "precision",
    "recall": "recall",
}
GUC_ETIKET = {"guclu": "güçlü", "zayif": "zayıf", "gurultu icinde": "gürültü içinde"}


def _etki_verisi(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Senaryo × metrik: farkın gürültü bandına oranı.

    Ham fark yerine **orana** bakılır: küçük bir grupta büyük görünen bir
    fark, o grubun doğal yayılımı içinde olabilir.
    """
    satirlar = []
    for _, r in sonuclar.iterrows():
        senaryo = str(r["scenario"])
        g = ne_gozlendi(senaryo)
        if not g:
            continue
        for metrik, d in g["metrikler"].items():
            esik = d["gurultu_esigi"]
            if not esik:
                continue
            satirlar.append({
                "senaryo": senaryo,
                "metrik": METRIK_ADI.get(metrik, metrik),
                "band oranı": round(abs(d["fark"]) / esik, 2),
                "fark": d["fark"],
                "eşik": esik,
            })
    return pd.DataFrame(satirlar)


def _guc_dagilimi(sonuclar: pd.DataFrame) -> pd.DataFrame:
    sayim: dict[str, int] = {}
    for _, r in sonuclar.iterrows():
        seviye = kanit_gucu(str(r["scenario"]))["seviye"]
        if seviye != "olcum yok":
            sayim[seviye] = sayim.get(seviye, 0) + 1
    sira = ["guclu", "zayif", "gurultu icinde"]
    return pd.DataFrame(
        {"koşu": [sayim.get(s, 0) for s in sira]},
        index=[GUC_ETIKET[s] for s in sira],
    )


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


def goster() -> None:
    sonuclar = load_results()
    ajan = ajan_kaydi()

    st.title("Termal Teşhis Ajanı")
    st.markdown(
        "Termal drone görüntüleriyle çalışan bir YOLO nesne tespit modeli "
        "**kontrollü biçimde bozulur**, bozulmanın metriklere nasıl yansıdığı "
        "ölçülür; sonra bir LLM ajanına bu ölçümler verilerek nedeni **kanıta "
        "dayalı** teşhis edip edemediği sınanır."
    )

    a, b, c, d = st.columns(4)
    a.metric("Koşu", len(sonuclar))
    b.metric("Kontrol koşusu", _kontrol_sayisi(sonuclar))
    c.metric("Hata galerisi", len(error_galleries()))
    d.metric("Test seti kullanımı", "YOK")
    stil.yorum(
        "Test seti final aşamasına kadar yasaktır ve bugüne kadar hiç "
        "kullanılmadı; bütün ölçümler kilitli tanı seti (1.056 görüntü, "
        "4.014 bbox) üzerinde yapıldı."
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
            stil.etki_haritasi(etki, x="metrik", y="senaryo", deger="band oranı"),
            use_container_width=True,
        )
        stil.yorum(
            "Koyu hücreler bandın belirgin üzerinde; açık hücreler gürültüden "
            "ayırt edilemiyor. Ayrıntı için 'Senaryolar' bölümüne bakın."
        )

    e, f = st.columns([1, 2])
    with e:
        stil.ust_baslik("kanıt gücü dağılımı")
        st.bar_chart(_guc_dagilimi(sonuclar), height=200)
    with f:
        stil.ust_baslik("üç ana bulgu")
        for baslik, metin in (
            ("Bozulmanın türü metrik imzasından okunabiliyor",
             "Çıkarım çözünürlüğü uyumsuzluğu recall'u çökertirken precision'a "
             "dokunmuyor; etiket bozulmaları precision'ı da bozuyor. Hangi "
             "metriğin bozulduğu arızanın türünü söylüyor."),
            ("Standart raporlama bir arızayı tamamen gizleyebiliyor",
             "E1'de 200 epoch süren ders kitabı niteliğinde bir aşırı uyum elde "
             "edildi. En iyi checkpoint ile raporlandığında model sağlıklı "
             "görünüyor (mAP50 farkı −0.001); arıza yalnızca eğitim eğrisinde "
             "ve son checkpoint'te görülüyor (−0.088)."),
            ("Gürültü ölçülmeden \"etki\" denemez",
             "Aynı veri ve protokolle, yalnızca rastgelelik tohumu "
             "değiştirilerek eğitilen dört model arasında bile belirgin fark "
             "var. Bu taban ölçülünce beş iddia zayıfladı ve bir senaryo "
             "(D6b) bulgu olmaktan çıktı."),
        ):
            with st.expander(baslik, expanded=False):
                st.markdown(metin)

    st.markdown("---")
    st.markdown("## Ajan")
    skor = _ajan_skorlari(ajan)
    if skor:
        g, h, i = st.columns(3)
        g.metric("Doğru neden teşhisi", f"{skor['teshis']:.1%}")
        h.metric("Tespit-farkındalıklı teşhis", f"{skor['teshis_tespit']:.1%}")
        i.metric("Rubrik ortalaması", f"{skor['rubrik']:.1%}")
        stil.kutu(
            "<b>Bu üç sayı aynı şeyi ölçmez.</b> Rubrik ortalaması üç bileşenin "
            f"ortalamasıdır ve ikisi doymuştur: kanıt {skor['kanit']:.0%}, "
            f"sınırlama {skor['sinir']:.0%} — her koşuda tam puan. Ayırt eden "
            "tek bileşen teşhistir. Yani ajan senaryoların "
            f"%{skor['rubrik'] * 100:.0f}'ini <i>bilmedi</i>; doğru nedeni "
            f"bulma oranı %{skor['teshis'] * 100:.0f}."
        )
    stil.yorum(
        "Koşu başına tek deneme yapıldı; bunlar nokta tahminidir ve güven "
        "aralığı hesaplanamaz. Ajanın hata profili ve neyin söylenemeyeceği "
        "için 'Deney Tasarımı ve Sınırlar' bölümüne bakın."
    )

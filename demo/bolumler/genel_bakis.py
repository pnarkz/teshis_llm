"""Genel Bakis: proje ne soruyor, ne yapti, su an nerede.

Tek ekran. Amaci etkilemek degil, konuyu ve durumu hizla vermek; ayrintiya
gerektiginde diger bolumlerden inilir.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import stil
from data_loader import ajan_kaydi, error_galleries, load_results


def _kontrol_sayisi(sonuclar: pd.DataFrame) -> int:
    return sum(
        1 for _, r in sonuclar.iterrows()
        if str(r["scenario"]).startswith("C") and str(r["scenario"])[1:2].isdigit()
    )


def goster() -> None:
    sonuclar = load_results()
    ajan = ajan_kaydi()

    st.title("Termal Teshis Ajani")
    st.markdown(
        "Termal drone goruntuleriyle calisan bir YOLO nesne tespit modeli "
        "**kontrollu bicimde bozulur**, bozulmanin metriklere nasil yansidigi "
        "olculur; sonra bir LLM ajanina bu olcumler verilerek nedeni **kanita "
        "dayali** teshis edip edemedigi sinanir."
    )

    st.markdown("---")
    stil.ust_baslik("arastirma sorusu")
    stil.kutu(
        "Termal nesne tespit sistemi hangi veri, etiket ve dagilim kosullarinda "
        "bozulur; bir LLM bu bozulmayi yeterli kanitla teshis edebilir mi?"
    )

    st.markdown("### Sayilarla durum")
    a, b, c, d = st.columns(4)
    a.metric("Kosu", len(sonuclar))
    b.metric("Kontrol kosusu", _kontrol_sayisi(sonuclar))
    c.metric("Hata galerisi", len(error_galleries()))
    d.metric("Test seti kullanimi", "YOK")
    stil.yorum(
        "Test seti final asamasina kadar yasaktir ve bugune kadar hic "
        "kullanilmadi; butun olcumler kilitli tanı seti (1.056 goruntu, "
        "4.014 bbox) uzerinde yapildi."
    )

    st.markdown("### Uc ana bulgu")

    st.markdown("**1. Bozulmanin turu metrik imzasindan okunabiliyor**")
    st.markdown(
        "Cikarim cozunurlugu uyumsuzlugu recall'u cokertirken precision'a "
        "dokunmuyor; etiket bozulmalari precision'i da bozuyor. Yani \"model "
        "kotu calisiyor\" demek yetmiyor - hangi metrigin bozuldugu arizanin "
        "turunu soyluyor."
    )

    st.markdown("**2. Standart raporlama bir arizayi tamamen gizleyebiliyor**")
    st.markdown(
        "E1'de 200 epoch suren ders kitabi niteliginde bir asiri uyum elde "
        "edildi. En iyi checkpoint ile raporlandiginda model **saglikli** "
        "gorunuyor (mAP50 farki -0.001); ariza yalnizca egitim egrisinde ve "
        "son checkpoint'te goruluyor (-0.088)."
    )

    st.markdown("**3. Gurultu olculmeden \"etki\" denemez**")
    st.markdown(
        "Ayni veri ve ayni protokolle, yalnizca rastgelelik tohumu "
        "degistirilerek egitilen dort model arasinda bile belirgin fark var. "
        "Bu taban olculunce bes iddia zayifladi ve bir senaryo (D6b) bulgu "
        "olmaktan cikti."
    )

    st.markdown("### Ajan")
    ozet = ajan.get("ozet", {})
    e, f = st.columns(2)
    e.metric("Ortalama puan (kati)", ozet.get("mean_score", "-"))
    f.metric("Ortalama puan (tespit-farkindalikli)", ozet.get("mean_score_tespit", "-"))
    stil.yorum(
        "Bu bir nokta tahminidir; kosu basina tek deneme yapildigi icin guven "
        "araligi hesaplanamaz. Ayrintili degerlendirme ve neyin soylenemeyecegi "
        "icin 'Deney Tasarimi ve Sinirlar' bolumune bakin."
    )

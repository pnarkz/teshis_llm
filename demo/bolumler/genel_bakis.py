"""Genel Bakis: proje ne soruyor, ne yapti, su an nerede.

Tek ekran. Amaci etkilemek degil, konuyu ve durumu hizla vermek; ayrintiya
gerektiginde diger bolumlerden inilir.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import stil
from data_loader import ajan_kaydi, error_galleries, load_results


def _guc_dagilimi(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Kosularin kanit gucune gore dagilimi.

    Metin yiginini bir bakista okunabilir hale getirir: kac kosu gercekten
    gurultu esigini asiyor, kacinin iddiasi zayif.
    """
    from teshis.degerlendirme.senaryo_ozeti import kanit_gucu

    sayim: dict[str, int] = {}
    for _, r in sonuclar.iterrows():
        seviye = kanit_gucu(str(r["scenario"]))["seviye"]
        if seviye == "olcum yok":
            continue
        sayim[seviye] = sayim.get(seviye, 0) + 1
    sira = ["guclu", "zayif", "gurultu icinde"]
    return pd.DataFrame(
        {"kosu": [sayim.get(s, 0) for s in sira]},
        index=[s for s in sira],
    )


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

    st.markdown("### Kosularin kanit gucu")
    st.markdown(
        "Her kosu, saglikli referansa gore **olculmus gurultu tabanina karsi** "
        "tartilir. Esigi asmayan bir fark, hicbir bozulma icermeyen kosular "
        "arasinda da gorulmustur."
    )
    st.bar_chart(_guc_dagilimi(sonuclar), height=200)
    stil.yorum(
        "Ayrinti icin 'Karsilastirma ve Gurultu' bolumune bakin; her kosunun "
        "hangi metrikte esigi astigi orada listelenir."
    )

    st.markdown("### Uc ana bulgu")
    for baslik, metin in (
        ("Bozulmanin turu metrik imzasindan okunabiliyor",
         "Cikarim cozunurlugu uyumsuzlugu recall'u cokertirken precision'a "
         "dokunmuyor; etiket bozulmalari precision'i da bozuyor. Hangi "
         "metrigin bozuldugu arizanin turunu soyluyor."),
        ("Standart raporlama bir arizayi tamamen gizleyebiliyor",
         "E1'de 200 epoch suren ders kitabi niteliginde bir asiri uyum elde "
         "edildi. En iyi checkpoint ile raporlandiginda model saglikli "
         "gorunuyor (mAP50 farki -0.001); ariza yalnizca egitim egrisinde ve "
         "son checkpoint'te goruluyor (-0.088)."),
        ("Gurultu olculmeden \"etki\" denemez",
         "Ayni veri ve protokolle, yalnizca rastgelelik tohumu degistirilerek "
         "egitilen dort model arasinda bile belirgin fark var. Bu taban "
         "olculunce bes iddia zayifladi ve bir senaryo (D6b) bulgu olmaktan "
         "cikti."),
    ):
        with st.expander(baslik, expanded=False):
            st.markdown(metin)

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

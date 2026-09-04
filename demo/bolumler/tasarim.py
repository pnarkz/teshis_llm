"""Deney Tasarimi ve Sinirlar.

Bu bolum projenin en savunulabilir kismidir: neyi kontrol ettigimiz ve neyi
HALA soyleyemedigimiz. Sinirlari gizlemek yerine one koymak, bir savunmada
en guclu karttir.
"""

from __future__ import annotations

import streamlit as st

import stil
from data_loader import ajana_gizlenenler, load_results


def _gurultu_tablosu():
    from teshis.degerlendirme.gurultu import alt_grup_bandi

    band = alt_grup_bandi()
    satirlar = []
    for alan, gruplar in band.items():
        for grup, d in gruplar.items():
            satirlar.append({
                "kirilim": alan.replace("_recall", ""),
                "grup": grup,
                "bbox": d["bbox_n"],
                "band": d["band"],
                "std": d["std"],
                "kosu": d["n_kosu"],
            })
    return sorted(satirlar, key=lambda s: -s["band"])


def goster() -> None:
    st.title("Deney Tasarimi ve Sinirlar")

    st.markdown("## Kontrollü deney kurgusu")
    st.markdown(
        "Gercek hayatta \"model neden kotu calisiyor?\" sorusu cevaplanamaz, "
        "cunku ayni anda birden fazla sey yanlis olabilir. Burada tersi "
        "yapilir: saglikli bir referans egitilir, sonra **her seferinde tek "
        "bir sey** kasitli olarak bozulur."
    )

    a, b = st.columns(2)
    with a:
        stil.ust_baslik("değişen")
        stil.kutu(
            "<b>D serisi:</b> veri (etiket, dagilim, temsil)<br>"
            "<b>E serisi:</b> egitim veya cikarim ayari<br>"
            "<b>C serisi:</b> yalnizca rastgelelik tohumu"
        )
    with b:
        stil.ust_baslik("sabit tutulan")
        stil.kutu(
            "Kilitli tanı seti (val_diagnostic) - hic degismez<br>"
            "Egitim protokolu - tek dosyada beyan edilir<br>"
            "Baslangic modeli, seed, cozunurluk<br>"
            "Test seti - final asamasina kadar yasak"
        )

    st.markdown("### Protokol sapmaları beyan edilir")
    st.markdown(
        "E serisi protokolu **kasitli olarak** bozar. Sapmalar koda dagilmis "
        "bayraklarla degil, `senaryolar/egitim_protokolu.yaml` icinde beyan "
        "edilir; her kosu kendi sapmasini manifestinde tasir. Boylece hangi "
        "kosunun protokolden nerede ayrildigi tek yerden okunur."
    )
    stil.kutu(
        "Somut ornek: E3'un tezi \"ogrenme orani 100 kat yuksek\". Ultralytics "
        "<code>optimizer=auto</code> iken lr0'i <b>yok sayar</b>; bu fark "
        "edilmeseydi E3 sessizce saglikli bir kosuya donusur ve \"kararsizlik "
        "gozlenmedi\" diye raporlanirdi. E3 sapmasi artik optimizer'i da "
        "acikca yaziyor."
    )

    st.markdown("---")
    st.markdown("## Ajanın körleştirilmesi")
    st.markdown(
        "Ajan hangi kosunun hangi senaryo oldugunu bilmez. Filtreler yapisaldir "
        "ve testlidir; ad listesine dayanmaz."
    )
    gizli = ajana_gizlenenler()
    st.dataframe(
        [{"alan": k, "durum": v} for k, v in gizli.items()],
        hide_index=True, width="stretch",
    )
    stil.yorum(
        "Cevap anahtarinin gonderilmemesi kritik: puanlama ancak ajan cevabi "
        "tamamlandiktan sonra ayri bir yerel islemle yapilir."
    )

    st.markdown("---")
    st.markdown("## Gürültü tabanı")
    st.markdown(
        "Hicbir sey bozulmadan, yalnizca rastgelelik tohumu degistirilerek "
        "egitilen kosular arasindaki yayilim. Bir farkin bu bandin altinda "
        "kalmasi, o farkin **saf rastgelelikten ayirt edilemedigi** anlamina "
        "gelir - buyuklugu ne olursa olsun."
    )
    satirlar = _gurultu_tablosu()
    st.dataframe(satirlar, hide_index=True, width="stretch")
    stil.yorum(
        "Dikkat: bu yalnizca kucuk orneklem sorunu degil. termal grubu 858 "
        "bbox tasir ama bandi hituav'in (2.165 bbox) bandinin on katindan "
        "fazladir; bazi gruplar gercekten oynaktir."
    )

    st.markdown("---")
    st.markdown("## Neyi HENÜZ söyleyemiyoruz")
    st.markdown(
        "Projenin asil sorusu \"bir LLM bozulmayi teshis edebilir mi?\" idi. "
        "**Bu soruyu cevaplayacak orneklem henuz yok.**"
    )
    for madde in (
        "Tek model, kosu basina tek deneme, tekrar yok. Olculen skor bir "
        "nokta tahminidir; guven araligi hesaplanamaz.",
        "Ajanin \"sorun uydurmama\" orani icin verilebilecek aralik cok genis "
        "(dort saf kontrolun birinde uydurdu).",
        "Gurultu bandi dort kosudan hesaplandi; az gozlemle band gercek "
        "yayilimi oldugundan kucuk gosterir.",
        "Referans tek bir kosudur (v00) ve dort saglikli kosunun en zayifidir; "
        "daha saglam bir taban onlarin ortalamasi olurdu.",
        "D6b'nin grup bazli iddiasi last.pt uzerinden kuruldu; saglikli "
        "last.pt kontrolleri henuz olculmedi.",
    ):
        st.markdown(f"- {madde}")

    stil.kutu(
        "<b>Bu bolumun amaci:</b> bulgulari zayiflatmak degil, hangilerinin "
        "ne kadar dayanikli oldugunu acikca soylemek. Gurultu tabani "
        "olculdukten sonra bes iddia geri cekildi; bu, olcumun calistiginin "
        "kanitidir."
    )

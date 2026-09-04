"""Hata Analizi: metriklerin soylemedigi seyi gosteren gorsel kanit.

Metrikler bir kosunun NE KADAR bozuldugunu soyler; galeri NEYIN bozuldugunu
gosterir. Projede bir kez bir manset bulgu yanlis raporlandi (D3b) ve
duzeltilmesi ancak gorsel kontrolle mumkun oldu; bu sayfa o kontrolun kalici
halidir.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import stil
from data_loader import error_galleries, images_for, load_results

SIRALAMA = {
    "Skor": "score",
    "Yanlis negatif": "false_negatives",
    "Yanlis pozitif": "false_positives",
    "Dusuk IoU": "mean_iou",
}


def goster() -> None:
    st.title("Hata Analizi")
    galeriler = error_galleries()
    if not galeriler:
        st.info("Henuz hata galerisi uretilmemis.")
        return

    adlar = sorted(galeriler)
    senaryo = st.selectbox(
        "Kosu", adlar, index=adlar.index("D4") if "D4" in adlar else 0
    )
    galeri = galeriler[senaryo]
    kayitlar = galeri["entries"]

    a, b = st.columns([2, 1])
    with a:
        siralama = st.radio("Sıralama", list(SIRALAMA), horizontal=True)
    with b:
        adet = st.slider("Gösterilecek örnek", 3, 12, 6)

    anahtar = SIRALAMA[siralama]
    ters = anahtar != "mean_iou"          # dusuk IoU'da kucukten buyuge
    sirali = sorted(
        [k for k in kayitlar if anahtar in k],
        key=lambda k: k[anahtar], reverse=ters,
    )[:adet]

    if not sirali:
        st.warning("Bu galeride siralanabilir kayit bulunamadi.")
        return

    st.markdown("### Özet")
    st.dataframe(
        pd.DataFrame([{
            "goruntu": k.get("source", k.get("image", "")).split("/")[-1],
            "yanlis negatif": k.get("false_negatives"),
            "yanlis pozitif": k.get("false_positives"),
            "ortalama IoU": round(k["mean_iou"], 3) if k.get("mean_iou") else None,
            "skor": round(k["score"], 2) if k.get("score") else None,
        } for k in sirali]),
        hide_index=True, width="stretch",
    )
    stil.yorum(
        "Skor, yanlis negatif ve yanlis pozitif sayilariyla dusuk IoU'yu "
        "birlestiren siralama olcutudur; en sorunlu kareleri one cikarir."
    )

    st.markdown("### Örnekler")
    klasor = galeri["folder"]
    for k in sirali:
        yol = k.get("image")
        if not yol:
            continue
        tam = (klasor / yol) if klasor else None
        if tam and tam.is_file():
            st.image(
                str(tam),
                caption=(f"FN {k.get('false_negatives')} · "
                         f"FP {k.get('false_positives')} · "
                         f"IoU {k.get('mean_iou', 0):.2f}"),
                width="stretch",
            )

    gorseller = images_for(senaryo)
    matris = [g for g in gorseller if "confusion" in g.name.lower()]
    if matris:
        st.markdown("### Confusion matrix")
        st.image(str(matris[0]), width="stretch")
        stil.yorum(
            "Not: Ultralytics'in urettigi bu gorsel bir kez kendi raporladigi "
            "recall ile celisti (D3b). Projedeki karisiklik iddialari bagimsiz "
            "olcumle (metrikler.py) uretilir."
        )

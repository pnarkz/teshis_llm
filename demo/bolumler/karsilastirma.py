"""Karsilastirma ve Gurultu: butun kosular tek tabloda, gurultuye karsi tartilmis.

Bu sayfanin manseti bir sayi degil bir DUZELTMEDIR: gurultu tabani olculunce
bes iddia zayifladi ve bir senaryo bulgu olmaktan cikti. Bunu gizlemek yerine
one koymak, olcumun calistiginin kanitidir.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import stil
from data_loader import load_results
from teshis.degerlendirme.senaryo_ozeti import kanit_gucu, ne_gozlendi


def _tablo(sonuclar: pd.DataFrame) -> pd.DataFrame:
    satirlar = []
    for _, r in sonuclar.iterrows():
        senaryo = r["scenario"]
        g = ne_gozlendi(senaryo)
        if not g:
            continue
        m = g["metrikler"]
        satirlar.append({
            "senaryo": senaryo,
            "mAP50": m["mAP50"]["deger"],
            "Δ mAP50": m["mAP50"]["fark"],
            "Δ precision": m["precision"]["fark"],
            "Δ recall": m["recall"]["fark"],
            "esigi asan": ", ".join(g["asan_metrikler"]) or "-",
            "kanit": kanit_gucu(senaryo)["seviye"],
        })
    return pd.DataFrame(satirlar)


def goster() -> None:
    sonuclar = load_results()
    st.title("Karsilastirma ve Gurultu")

    st.markdown(
        "Butun kosular saglikli referansa (v00) gore, **olculmus gurultu "
        "tabanina karsi** tartilmistir. Esigi asmayan bir fark, hicbir bozulma "
        "icermeyen kosular arasinda da gorulmustur."
    )

    df = _tablo(sonuclar)
    sadece_asan = st.checkbox("Yalnizca gurultu esigini asanlari goster", value=False)
    gosterilen = df[df["esigi asan"] != "-"] if sadece_asan else df
    st.dataframe(gosterilen, hide_index=True, width="stretch", height=520)
    stil.yorum(
        "'kanit' sutunu uc seviye alir: guclu (birden fazla metrik esigi asiyor), "
        "zayif (tek metrik), gurultu icinde (hicbiri)."
    )

    st.markdown("---")
    st.markdown("## Gurultu tabani olculunce ne degisti")
    st.markdown(
        "Ilk olcum tek bir kontrol kosusuna dayaniyordu ve gurultuyu ciddi "
        "bicimde **kucuk** gosteriyordu. Uc kontrol kosusuna cikildiginda "
        "esikler buyudu:"
    )
    st.dataframe(
        pd.DataFrame([
            {"metrik": "mAP50", "n=1 esigi": 0.0015, "n=3 esigi": 0.0060, "kat": "4.0x"},
            {"metrik": "recall", "n=1 esigi": 0.0114, "n=3 esigi": 0.0430, "kat": "3.8x"},
            {"metrik": "AP tasit", "n=1 esigi": 0.0021, "n=3 esigi": 0.0216, "kat": "10.3x"},
            {"metrik": "mAP50-95", "n=1 esigi": 0.0129, "n=3 esigi": 0.0201, "kat": "1.6x"},
            {"metrik": "precision", "n=1 esigi": 0.0184, "n=3 esigi": 0.0184, "kat": "1.0x"},
        ]),
        hide_index=True, width="stretch",
    )

    st.markdown("### Zayiflayan bes iddia")
    st.dataframe(
        pd.DataFrame([
            {"senaryo": "D1", "kaybettigi": "mAP50, recall (yalnizca mAP50-95 kaldi)"},
            {"senaryo": "D2b", "kaybettigi": "mAP50-95, recall"},
            {"senaryo": "D3", "kaybettigi": "recall"},
            {"senaryo": "D4", "kaybettigi": "recall"},
            {"senaryo": "D6b", "kaybettigi": "mAP50 - geriye hicbir sey kalmadi"},
        ]),
        hide_index=True, width="stretch",
    )
    stil.yorum(
        "Genel oruntu: recall'a dayanan iddialar en kirilgan olanlar. Recall'un "
        "seed degiskenligi (0.043) precision'inkinin (0.018) iki katindan fazla."
    )

    st.markdown("### Erken durdurma noktasi da seed'e bagli")
    st.dataframe(
        pd.DataFrame([
            {"kosu": "v00", "seed": 42, "durdugu epoch": 11, "en iyi epoch": 1},
            {"kosu": "C2", "seed": 7, "durdugu epoch": 19, "en iyi epoch": 9},
            {"kosu": "C2 seed13", "seed": 13, "durdugu epoch": 30, "en iyi epoch": 11},
            {"kosu": "C2 seed21", "seed": 21, "durdugu epoch": 30, "en iyi epoch": 16},
        ]),
        hide_index=True, width="stretch",
    )
    stil.yorum(
        "Ayni veri, ayni protokol: egitim suresi 11 ile 30 epoch arasinda "
        "degisiyor. Gurultu yalnizca son metrikte degil surecin kendisinde de var."
    )

    st.markdown("---")
    st.markdown("## Checkpoint secimi bir kor nokta")
    st.dataframe(
        pd.DataFrame([
            {"kosu": "E1 best.pt", "mAP50": 0.9190, "Δ v00": -0.0010,
             "okuma": "saglikli gorunuyor"},
            {"kosu": "E1 last.pt", "mAP50": 0.8318, "Δ v00": -0.0881,
             "okuma": "asiri uyum goruluyor"},
            {"kosu": "D5 best.pt", "mAP50": 0.9092, "Δ v00": -0.0107,
             "okuma": "sinirli kayip"},
            {"kosu": "D5 last.pt", "mAP50": 0.3352, "Δ v00": -0.5848,
             "okuma": "felaket"},
        ]),
        hide_index=True, width="stretch",
    )
    stil.yorum(
        "Ayni kosunun iki checkpoint'i arasindaki fark, 'bu kosu saglikli mi' "
        "sorusunun cevabini degistiriyor. Bu yuzden defterde her ikisi de "
        "ayri satir olarak tutuluyor."
    )

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


def _erken_durdurma(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Bozulmasiz kosularin durdugu ve en iyi checkpoint'i verdigi epoch.

    Elle yazilmis bir tablo yerine kosu dizinlerinden okunur; yeni bir
    kontrol kosusu eklendiginde tablo kendiliginden buyur.
    """
    import csv as _csv

    from data_loader import run_dir_for

    satirlar = []
    for _, r in sonuclar.iterrows():
        ad = str(r["scenario"])
        bozulmasiz = ad == "v00_saglikli" or (ad.startswith("C") and ad[1:2].isdigit())
        if not bozulmasiz or not str(r.get("weights_path", "")).endswith("best.pt"):
            continue
        dizin = run_dir_for(r)
        yol = (dizin / "results.csv") if dizin else None
        if not yol or not yol.is_file():
            continue
        with yol.open(encoding="utf-8") as f:
            kayit = [{k.strip(): v for k, v in s.items()} for s in _csv.DictReader(f)]
        ciftler = [
            (int(float(s["epoch"])), float(s["metrics/mAP50(B)"]))
            for s in kayit if s.get("metrics/mAP50(B)")
        ]
        if not ciftler:
            continue
        en_iyi = max(ciftler, key=lambda c: c[1])
        satirlar.append({
            "kosu": ad,
            "seed": int(r["seed"]),
            "durdugu epoch": len(kayit),
            "en iyi epoch": en_iyi[0],
            "en iyi mAP50 (egitim val)": round(en_iyi[1], 4),
        })
    return pd.DataFrame(sorted(satirlar, key=lambda s: s["seed"]))


def _checkpoint_ciftleri(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Ayni kosunun best.pt / last.pt satirlarini yan yana koyar.

    Defterde her ikisi de ayri satir olarak durur; ciftleri elle listelemek
    yerine ad esleştirmesiyle bulunur.
    """
    from teshis.degerlendirme.senaryo_ozeti import ne_gozlendi

    adlar = set(sonuclar["scenario"])
    satirlar = []
    for ad in sorted(adlar):
        if not ad.endswith(" last_pt"):
            continue
        taban = ad[: -len(" last_pt")]
        if taban not in adlar:
            continue
        for etiket, senaryo in ((f"{taban} best.pt", taban), (f"{taban} last.pt", ad)):
            g = ne_gozlendi(senaryo)
            if not g:
                continue
            m = g["metrikler"]["mAP50"]
            satirlar.append({
                "kosu": etiket,
                "mAP50": m["deger"],
                "Δ v00": m["fark"],
                "esigi asiyor": "evet" if m["asiyor"] else "hayir",
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
    st.dataframe(_erken_durdurma(sonuclar), hide_index=True, width="stretch")
    stil.yorum(
        "Ayni veri, ayni protokol: egitim suresi 11 ile 30 epoch arasinda "
        "degisiyor. Gurultu yalnizca son metrikte degil surecin kendisinde de var."
    )

    st.markdown("---")
    st.markdown("## Checkpoint secimi bir kor nokta")
    st.dataframe(_checkpoint_ciftleri(sonuclar), hide_index=True, width="stretch")
    stil.yorum(
        "Ayni kosunun iki checkpoint'i arasindaki fark, 'bu kosu saglikli mi' "
        "sorusunun cevabini degistiriyor. Bu yuzden defterde her ikisi de "
        "ayri satir olarak tutuluyor."
    )
    stil.kutu(
        "<b>Dikkat - son checkpoint dususunun de bir tabani var.</b> "
        "Saglikli referansin kendisi best.pt'den last.pt'ye gecerken "
        "<b>-0.0280</b> dusuyor. Yani her last.pt dususu bozulma isareti "
        "degildir; D4 (-0.0329) ve D6b (-0.0330) bu tabana cok yakin. "
        "Ayrisanlar E1 (-0.0881) ve ozellikle D5 (-0.5848)."
    )

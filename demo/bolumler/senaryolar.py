"""Senaryolar: her kosunun deney ozeti ve kaniti.

Sayfanin yapisi bilerek bir arastirma raporunun kisa ozeti gibidir:

    Ne olcuyor? -> Ne degistirildi? -> Referansla ortak olan ne? ->
    Beklenen etki -> Ne gozlendi? -> Kanit ne kadar guclu? -> Sinirlama

"Referansla ortak olan" basligi bilerek boyle: sayfa bir donem "ne sabit
kaldi" diyip degerlendirme setini SABIT val_diagnostic yaziyordu, oysa D6a
kasitli olarak sizintili kumede olculur. Basligi karsilastirmaya baglamak
hem dogru hem de neyin neye gore olculdugunu ekranda gorunur kiliyor.

Icerigin bes bileseninden dordu turetilir (teshis/degerlendirme/senaryo_ozeti);
elle yazilan tek alan senaryonun ne olctugudur.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import stil
from data_loader import (
    curves_for,
    evidence_for,
    examples_for,
    images_for,
    load_results,
    training_curve,
)
from teshis.degerlendirme.senaryo_ozeti import ozet


def _metrik_tablosu(gozlem: dict) -> pd.DataFrame:
    """Metrik tablosu; referans sutunu KENDI referansinin adini tasir.

    Baslik "referans (v00)" diye sabit yaziliydi. Referans artik kosudan
    kosuya degisiyor - `D1n`'inki v00n, `D4 last_pt`'ninki v00'in last.pt'si.
    Sabit baslik, yanlis bir karsilastirma yapildigi izlenimi verirdi.
    """
    ref_ad = gozlem.get("referans_senaryo") or "referans yok"
    satirlar = []
    for ad, d in gozlem["metrikler"].items():
        satirlar.append({
            "metrik": ad,
            "değer": d["deger"],
            f"referans ({ref_ad})": d["referans"],
            "fark": d["fark"],
            "gürültü eşiği": d["gurultu_esigi"],
            "eşiği aşıyor": {True: "evet", False: "hayır"}.get(d["asiyor"], "—"),
        })
    return pd.DataFrame(satirlar)


def _kirilim_tablosu(senaryo: str, kosu_id: str | None):
    """Ajanın gördüğü kirilimi, gürültü bandiyla birlikte gösterir."""
    from teshis.ajan import araclar

    if kosu_id is None:
        return None, None
    try:
        kaynak = araclar.kaynak_bazli_recall_getir(kosu_id)["kaynaklar"]
        boyut = araclar.boyut_bazli_recall_getir(kosu_id)["bantlar"]
    except Exception:  # noqa: BLE001 - ajana verilmeyen kosular
        return None, None

    def cevir(d, ad):
        return pd.DataFrame([
            {ad: g, "recall": v["recall"], "referans": v["referans_recall"],
             "fark": v["fark"], "band": v["gurultu_bandi"],
             "band orani": v["band_orani"]}
            for g, v in d.items()
        ])
    return cevir(kaynak, "kaynak"), cevir(boyut, "boyut bandi")


def _kosu_id(senaryo: str) -> str | None:
    from data_loader import ajan_kosu_haritasi

    for kosu, ad in ajan_kosu_haritasi().items():
        if ad == senaryo:
            return kosu
    return None


def goster() -> None:
    sonuclar = load_results()
    adlar = sonuclar["scenario"].tolist()

    st.title("Senaryolar")
    senaryo = st.selectbox("Koşu", adlar, index=adlar.index("D4") if "D4" in adlar else 0)
    o = ozet(senaryo)

    st.markdown("## Deney özeti")
    if o["ne_olcuyor"]:
        stil.kutu(o["ne_olcuyor"].strip())

    a, b = st.columns(2)
    with a:
        stil.ust_baslik("ne değiştirildi")
        d = o["ne_degisti"]
        satir = f"<b>Tur:</b> {d.get('tur') or '-'}<br>"
        if d.get("parametreler"):
            for k, v in d["parametreler"].items():
                satir += f"<b>{k}:</b> {v}<br>"
        if d.get("veri_surumu"):
            satir += f"<b>Veri surumu:</b> <code>{d['veri_surumu']}</code>"
        stil.kutu(satir)
    with b:
        stil.ust_baslik("referansla ortak olan")
        stil.kutu("<br>".join(o["ne_sabit_kaldi"]))

    if o["beklenen_etki"]:
        stil.ust_baslik("beklenen etki")
        stil.kutu(str(o["beklenen_etki"]))

    st.markdown("## Ne gözlendi")
    gozlem = o["ne_gozlendi"]
    if gozlem:
        # Karsilastirmanin KIMLIGI once gelir: hangi kosuyla, hangi olcekte.
        # Bu satir olmadan asagidaki fark sutunu neyin farki oldugunu
        # soylemiyordu ve saglikli kosular "bozulmus" gorunuyordu.
        kimlik = gozlem.get("kimlik") or {}
        stil.ust_baslik("karşılaştırma ölçeği")
        stil.kutu(
            f"<b>Aday:</b> {senaryo} &nbsp;→&nbsp; "
            f"<b>Referans:</b> {gozlem.get('referans_senaryo') or '— yok —'}<br>"
            f"<b>Model:</b> {kimlik.get('model', '?')} · "
            f"<b>Küme:</b> {kimlik.get('degerlendirme_seti', '?')} · "
            f"<b>imgsz:</b> {kimlik.get('imgsz_eval', '?')} · "
            f"<b>Checkpoint:</b> {kimlik.get('checkpoint', '?')}.pt<br>"
            f"<span class='yorum'>{gozlem.get('karsilastirma_aciklamasi', '')}</span>"
        )
        st.dataframe(_metrik_tablosu(gozlem), hide_index=True, width="stretch")
        guc = o["kanit_gucu"]
        st.markdown(stil.guc_rozeti(guc["seviye"]) + " " + guc["aciklama"],
                    unsafe_allow_html=True)
        n = gozlem["kontrol_kosu_sayisi"]
        stil.yorum(
            f"Gürültü eşiği {n} kontrol koşusundan hesaplandı. Eşiğin altında "
            "kalan bir fark, saf rastgelelikten ayırt edilemez."
            if n else
            "Bu ölçekte kontrol koşusu yok, bu yüzden gürültü eşiği "
            "hesaplanamıyor; başka bir ölçeğin eşiği ödünç alınamaz."
        )

    kosu_id = _kosu_id(senaryo)
    kaynak_df, boyut_df = _kirilim_tablosu(senaryo, kosu_id)
    if kaynak_df is not None:
        st.markdown("## Kırılım")
        c, d2 = st.columns(2)
        with c:
            stil.ust_baslik("kaynak grubu")
            st.dataframe(kaynak_df, hide_index=True, width="stretch")
        with d2:
            stil.ust_baslik("nesne boyutu")
            st.dataframe(boyut_df, hide_index=True, width="stretch")
        stil.yorum(
            "'band oranı' = |fark| / o grubun gürültü bandı. 1'in altındaki "
            "bir oran, farkın bozulmasız koşular arasında da görüldüğü "
            "anlamına gelir."
        )

    egri = training_curve(sonuclar[sonuclar["scenario"] == senaryo].iloc[0])
    if egri is not None and not egri.empty:
        st.markdown("## Eğitim eğrisi")
        sutunlar = [s for s in ("train/cls_loss", "val/cls_loss",
                                "metrics/mAP50(B)") if s in egri.columns]
        if sutunlar:
            st.line_chart(egri.set_index("epoch")[sutunlar], height=240)
            stil.yorum(
                "Train ve val kaybı arasındaki farkın açılması aşırı uyum "
                "imzasıdır; metrikler sessizken eğri konuşabilir."
            )

    gorseller = images_for(senaryo)
    if gorseller:
        st.markdown("## Confusion matrix ve eğriler")
        secili = [g for g in gorseller if "confusion" in g.name.lower()][:1] or gorseller[:1]
        for g in secili:
            st.image(str(g), width="stretch")
        with st.expander("Diğer grafikler"):
            for g in gorseller:
                st.image(str(g), caption=g.name, width="stretch")

    ornekler = examples_for(senaryo)
    if ornekler:
        st.markdown("## Örnek tahminler")
        st.image([str(p) for p in ornekler[:3]], width="stretch")

    st.markdown("## Sınırlamalar")
    for s in o["sinirlamalar"]:
        st.markdown(f"- {s}")

    with st.expander("Ham metrik dosyası"):
        st.json(evidence_for(senaryo))

"""Canli Ajan: kor teshis akisi.

Iki mod var ve ikisi de AYNI ekrani uretir:

- **Kayitli kosu** (varsayilan): tamamlanmis denemeden okunur. API harcamaz,
  her zaman calisir. Sunumun guvenli yolu budur.
- **Canli kosu**: ajan o anda calistirilir. Ucretsiz katman 20 istek/gun ve
  5 istek/dk ile sinirlidir; kota veya gecici sunucu hatasi olabilir, bu
  yuzden varsayilan degildir.

Ajanin gordugu kanit her iki modda da yerelde yeniden uretilebilir: araclar
deterministiktir ve API harcamaz.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import streamlit as st

import stil
from data_loader import (
    ajan_araclarini_calistir,
    ajan_kaydi,
    ajan_kosu_haritasi,
    ajana_gizlenenler,
)


def _korluk_paneli(kosu_id: str, senaryo: str) -> None:
    stil.ust_baslik("kor teshis")
    st.markdown(
        f"<div class='kutu'><b>Sunucu gorunumu:</b> {senaryo}<br>"
        f"<b>Ajana gonderilen kimlik:</b> <code>{kosu_id}</code><br>"
        f"<b>Senaryo adi ajandan gizlendi:</b> EVET</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Ajana ne gidiyor, ne gitmiyor"):
        st.dataframe(
            [{"alan": k, "durum": v} for k, v in ajana_gizlenenler().items()],
            hide_index=True, width="stretch",
        )


def _arac_kayit_tablosu(cagrilar: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "tur": c.get("tur"),
            "arac": c.get("arac"),
            "arguman": json.dumps(c.get("argumanlar", {}), ensure_ascii=False),
            "hata": c.get("hata") or "-",
        }
        for c in cagrilar
    ])


def _teshis_goster(cevap: dict) -> None:
    st.markdown("### Teshis")
    stil.kutu(f"<b>{cevap.get('diagnosis', '-')}</b>")
    a, b = st.columns([1, 3])
    a.metric("Guven", cevap.get("confidence", "-"))
    with b:
        stil.ust_baslik("kanit")
        for k in cevap.get("evidence", []) or []:
            st.markdown(f"- {k}")
    if cevap.get("limitations"):
        stil.ust_baslik("sinirlamalar")
        for k in cevap["limitations"]:
            st.markdown(f"- {k}")
    if cevap.get("next_measurement"):
        stil.ust_baslik("onerilen sonraki olcum")
        st.markdown(cevap["next_measurement"])


def _puan_goster(puan: dict) -> None:
    st.markdown("### Cevap anahtariyla karsilastirma")
    st.markdown(
        "Cevap anahtari ajana **gonderilmedi**; puanlama cevap uretildikten "
        "sonra yerelde yapildi."
    )
    st.dataframe(
        pd.DataFrame([{
            "gercek senaryo (beklenen)": puan.get("expected"),
            "ajanin teshisi": puan.get("model_diagnosis"),
            "teshis puani": puan.get("diagnosis_score"),
            "kanit puani": puan.get("evidence_score"),
            "sinirlama puani": puan.get("limitation_score"),
            "toplam": puan.get("total"),
        }]),
        hide_index=True, width="stretch",
    )
    if puan.get("tespit_notu"):
        stil.yorum(
            "Bu kosuda bozulma kanitta anlamli iz birakmiyor; "
            f"tespit-farkindalikli puan {puan.get('diagnosis_score_tespit')}. "
            f"Gerekce: {puan['tespit_notu']}"
        )


def _canli_calistir(kosu_id: str) -> dict | None:
    """Ajani o anda calistirir. Hata turlerini ayirt ederek raporlar."""
    if not os.environ.get("GEMINI_API_KEY"):
        st.error(
            "GEMINI_API_KEY ortam degiskeni tanimli degil. Canli calistirma "
            "icin anahtar gerekir; kayitli kosu modu anahtarsiz calisir."
        )
        return None

    from teshis.ajan import ajan as ajan_modulu

    baslangic = time.time()
    with st.status(f"{kosu_id} icin canli teshis uretiliyor...", expanded=True) as durum:
        try:
            cevap, kayit = ajan_modulu.teshis_uret(kosu_id)
        except Exception as hata:  # noqa: BLE001
            metin = f"{type(hata).__name__}: {hata}".lower()
            if any(k in metin for k in ("quota", "resource_exhausted", "429")):
                durum.update(label="Gunluk kota bitti", state="error")
                st.error(
                    "429 RESOURCE_EXHAUSTED - gunluk istek kotasi asildi.\n\n"
                    "Ucretsiz katman 20 istek/gun. Kayitli kosu modu calismaya "
                    "devam eder."
                )
            elif any(k in metin for k in ("503", "unavailable", "high demand")):
                durum.update(label="Gecici sunucu hatasi", state="error")
                st.warning(
                    "503 UNAVAILABLE - saglayicida gecici yogunluk. Bu hata "
                    "kotayla ilgili degildir; birkac saniye sonra yeniden "
                    "denenebilir."
                )
            else:
                durum.update(label="Basarisiz", state="error")
                st.error(f"{type(hata).__name__}: {hata}")
            return None
        sure = time.time() - baslangic
        durum.update(label=f"Tamamlandi ({sure:.1f} sn, {len(kayit)} arac cagrisi)",
                     state="complete")
    return {"cevap": cevap, "kayit": kayit, "sure": sure}


def goster() -> None:
    st.title("Ajan")
    st.markdown(
        "Ajana yalnizca anonim metrikler ve kirilim araclari verilir; hangi "
        "kosunun hangi senaryo oldugunu bilmez. Teshisini kendi sectigi "
        "kanitla uretir."
    )

    kayit = ajan_kaydi()
    harita = ajan_kosu_haritasi()
    kosular = sorted(kayit["cevaplar"]) or sorted(harita)
    varsayilan = kosular.index("kosu_08") if "kosu_08" in kosular else 0
    kosu_id = st.selectbox(
        "Kosu", kosular, index=varsayilan,
        format_func=lambda k: f"{k}  —  {harita.get(k, '?')}",
    )
    senaryo = harita.get(kosu_id, "?")

    mod = st.radio(
        "Kaynak", ["Kayitli kosu", "Canli calistir"], horizontal=True,
        help=("Kayitli kosu API harcamaz ve her zaman calisir. Canli mod "
              "ucretsiz katman sinirlarina tabidir (20 istek/gun, 5 istek/dk)."),
    )

    st.markdown("---")
    _korluk_paneli(kosu_id, senaryo)

    st.markdown("### Ajana verilen kanit")
    st.markdown(
        "Araclar deterministiktir; asagidaki cikti yerelde yeniden uretildi ve "
        "API harcamadi. Ajanin gordugu kanit tam olarak budur."
    )
    kanit = ajan_araclarini_calistir(kosu_id)
    with st.expander("Arac ciktilari", expanded=False):
        for ad, deger in kanit.items():
            st.markdown(f"**{ad}**")
            st.json(deger, expanded=False)

    st.markdown("---")
    if mod == "Canli calistir":
        if st.button("Ajani calistir", type="primary"):
            sonuc = _canli_calistir(kosu_id)
            if sonuc:
                st.markdown("### Arac cagrilari")
                st.dataframe(_arac_kayit_tablosu(sonuc["kayit"]),
                             hide_index=True, width="stretch")
                _teshis_goster(sonuc["cevap"])
                with st.expander("Ham cevap (JSON)"):
                    st.json(sonuc["cevap"])
        else:
            st.info(
                "Canli mod secildi. 'Ajani calistir' dugmesine basildiginda "
                "saglayiciya gercek bir istek gonderilir."
            )
        return

    cevap = kayit["cevaplar"].get(kosu_id)
    if not cevap:
        st.warning(f"{kosu_id} icin kayitli cevap yok. Canli modu deneyebilirsiniz.")
        return

    cagrilar = (kayit["arac_kaydi"].get(kosu_id) or {}).get("arac_cagrilari", [])
    if cagrilar:
        st.markdown("### Arac cagrilari")
        st.dataframe(_arac_kayit_tablosu(cagrilar), hide_index=True,
                     width="stretch")
        stil.yorum(
            f"Ajan bu kosuda {len(cagrilar)} arac cagirdi. Hangi kaniti "
            "isteyecegine kendisi karar verdi."
        )

    _teshis_goster(cevap)
    puan = kayit["puanlar"].get(kosu_id)
    if puan:
        _puan_goster(puan)

    with st.expander("Ham cevap (JSON)"):
        st.json(cevap)

    st.markdown("---")
    st.markdown("### Denemenin butunu")
    ozet = kayit["ozet"]
    a, b = st.columns(2)
    a.metric("Ortalama puan (kati)", ozet.get("mean_score", "-"))
    b.metric("Ortalama puan (tespit-farkindalikli)", ozet.get("mean_score_tespit", "-"))
    stil.yorum(
        "Kosu basina tek deneme yapildi; bu bir nokta tahminidir ve guven "
        "araligi hesaplanamaz. Ajanin hata profili icin 'Deney Tasarimi ve "
        "Sinirlar' bolumune bakin."
    )

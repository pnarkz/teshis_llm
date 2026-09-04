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


def _korluk_paneli(kosu_id: str, senaryo: str, acik: bool) -> None:
    """Kor teshis paneli.

    Gercek senaryo BASLANGICTA GIZLIDIR - sunucu da ajanla ayni bilgiyle
    baslar. "Gercegi goster" ile acilir. Bu, kor tasarimi anlatmakla
    kalmayip izleyiciye YASATIR: once kanita bakilir, sonra cevap acilir.
    """
    stil.ust_baslik("kör teşhis")
    gercek = (
        f"<b>Gerçek senaryo:</b> {senaryo}" if acik
        else "<b>Gerçek senaryo:</b> <i>gizli — aşağıdan açabilirsiniz</i>"
    )
    st.markdown(
        f"<div class='kutu'>{gercek}<br>"
        f"<b>Ajana gönderilen kimlik:</b> <code>{kosu_id}</code><br>"
        f"<b>Senaryo adı ajandan gizlendi:</b> EVET</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Ajana ne gidiyor, ne gitmiyor"):
        st.dataframe(
            [{"alan": k, "durum": v} for k, v in ajana_gizlenenler().items()],
            hide_index=True, width="stretch",
        )


ARAC_ACIKLAMA = {
    "baseline_metriklerini_getir": "sağlıklı referansın metrikleri",
    "kosu_metriklerini_getir": "bu koşunun metrikleri",
    "baseline_farkini_getir": "referansa göre farklar",
    "bbox_sayilarini_getir": "sınıf başına örnek sayısı",
    "boyut_bazli_recall_getir": "nesne boyutu kırılımı",
    "kaynak_bazli_recall_getir": "kaynak grubu kırılımı",
    "sinif_karisikligini_getir": "sınıf karışıklığı matrisi",
}


def _arac_zaman_cizelgesi(cagrilar: list[dict]) -> pd.DataFrame:
    """Ajanın kanıt toplama sırası.

    Sıra bilgi taşır: ajan önce genel metriklere bakıp sonra hangi kırılımı
    sorguladığı, akıl yürütmesinin izidir. Tablo yerine sıralı bir çizelge
    bunu okunur kılar.
    """
    return pd.DataFrame([
        {
            "sıra": i + 1,
            "tur": c.get("tur"),
            "araç": c.get("arac"),
            "ne sorduğu": ARAC_ACIKLAMA.get(c.get("arac"), "-"),
            "hata": c.get("hata") or "-",
        }
        for i, c in enumerate(cagrilar)
    ])


def _teshis_goster(cevap: dict) -> None:
    st.markdown("### Teşhis")
    stil.kutu(f"<b>{cevap.get('diagnosis', '-')}</b>")
    st.markdown(stil.guven_rozeti(cevap.get("confidence", "-")),
                unsafe_allow_html=True)
    stil.yorum(
        "Bu değer ajanın kendi beyanıdır; kalibre edilmiş bir olasılık "
        "değildir. Doğrulukla ilişkisi ölçülmedi."
    )
    stil.ust_baslik("kanıt")
    for k in cevap.get("evidence", []) or []:
        st.markdown(f"- {k}")
    if cevap.get("limitations"):
        stil.ust_baslik("sınırlamalar")
        for k in cevap["limitations"]:
            st.markdown(f"- {k}")
    if cevap.get("next_measurement"):
        stil.ust_baslik("önerilen sonraki ölçüm")
        st.markdown(cevap["next_measurement"])


def _puan_goster(puan: dict) -> None:
    st.markdown("### Cevap anahtarıyla karşılaştırma")
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


def _senaryo_ozeti(senaryo: str) -> dict:
    from teshis.degerlendirme.senaryo_ozeti import ozet

    try:
        return ozet(senaryo)
    except Exception:  # noqa: BLE001 - ozet uretilemezse akis durmasin
        return {}


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
    # Secicide GERCEK AD GOSTERILMEZ: sunum sirasinda izleyici de sunucu da
    # ajanla ayni bilgiyle baslar. Ad, "gercegi goster" ile acilir.
    kosu_id = st.selectbox("Koşu", kosular, index=varsayilan)
    senaryo = harita.get(kosu_id, "?")

    anahtar = f"acik_{kosu_id}"
    if anahtar not in st.session_state:
        st.session_state[anahtar] = False
    acik = st.session_state[anahtar]

    mod = st.radio(
        "Kaynak", ["Kayıtlı koşu", "Canlı çalıştır"], horizontal=True,
        help=("Kayitli kosu API harcamaz ve her zaman calisir. Canli mod "
              "ucretsiz katman sinirlarina tabidir (20 istek/gun, 5 istek/dk)."),
    )

    st.markdown("---")
    _korluk_paneli(kosu_id, senaryo, acik)

    st.markdown("### Ajana verilen kanıt")
    st.markdown(
        "Araçlar deterministiktir; aşağıdaki çıktı yerelde yeniden üretildi ve "
        "API harcamadı. Ajanın gördüğü kanıt tam olarak budur."
    )
    kanit = ajan_araclarini_calistir(kosu_id)
    with st.expander("Araç çıktıları", expanded=False):
        for ad, deger in kanit.items():
            st.markdown(f"**{ad}**")
            st.json(deger, expanded=False)

    st.markdown("---")
    if mod == "Canlı çalıştır":
        if st.button("Ajanı çalıştır", type="primary"):
            sonuc = _canli_calistir(kosu_id)
            if sonuc:
                st.markdown("### Araç çağrıları")
                st.dataframe(_arac_zaman_cizelgesi(sonuc["kayit"]),
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
        st.markdown("### Araç çağrıları")
        st.dataframe(_arac_zaman_cizelgesi(cagrilar), hide_index=True,
                     width="stretch")
        stil.yorum(
            f"Ajan bu kosuda {len(cagrilar)} arac cagirdi. Hangi kaniti "
            "isteyecegine kendisi karar verdi."
        )

    _teshis_goster(cevap)

    st.markdown("---")
    if not acik:
        stil.kutu(
            "Ajanın teşhisi yukarıda. <b>Gerçek senaryo hâlâ gizli.</b> "
            "Kanıta bakıp kendi kararınızı verdikten sonra açın."
        )
        if st.button("Gerçeği göster", type="primary"):
            st.session_state[anahtar] = True
            st.rerun()
    else:
        ozet = _senaryo_ozeti(senaryo)
        st.markdown("### Gerçek senaryo")
        stil.kutu(
            f"<b>{senaryo}</b>"
            + (f"<br>{ozet['ne_olcuyor'].strip()}" if ozet.get("ne_olcuyor") else "")
        )
        puan = kayit["puanlar"].get(kosu_id)
        if puan:
            _puan_goster(puan)
        if st.button("Yeniden gizle"):
            st.session_state[anahtar] = False
            st.rerun()

    with st.expander("Ham cevap (JSON)"):
        st.json(cevap)

    st.markdown("---")
    st.markdown("### Denemenin bütünü")
    puanlar = list(kayit["puanlar"].values())
    if puanlar:
        n = len(puanlar)
        teshis = sum(p["diagnosis_score"] for p in puanlar) / n
        tespit = sum(p["diagnosis_score_tespit"] for p in puanlar) / n
        rubrik = kayit["ozet"].get("mean_score")
        a, b, c = st.columns(3)
        a.metric("Doğru neden teşhisi", f"{teshis:.1%}")
        b.metric("Tespit-farkındalıklı", f"{tespit:.1%}")
        c.metric("Rubrik ortalaması", f"{rubrik:.1%}" if rubrik else "-")
        stil.kutu(
            "Rubrik ortalaması üç bileşenin ortalamasıdır ve ikisi (kanıt, "
            "sınırlama) her koşuda tam puan alır. Ayırt eden tek bileşen "
            "teşhistir; ajanın <b>doğru nedeni bulma oranı "
            f"%{teshis * 100:.0f}</b>'dir."
        )
    stil.yorum(
        "Koşu başına tek deneme yapıldı; bunlar nokta tahminidir ve güven "
        "aralığı hesaplanamaz. Ajanın hata profili için 'Deney Tasarımı ve "
        "Sınırlar' bölümüne bakın."
    )

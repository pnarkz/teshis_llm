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
        "Cevap anahtarı ajana **gönderilmedi**; puanlama cevap üretildikten "
        "sonra yerelde yapıldı."
    )
    st.dataframe(
        pd.DataFrame([{
            "gerçek senaryo (beklenen)": puan.get("expected"),
            "ajanın teşhisi": puan.get("model_diagnosis"),
            "teşhis puanı": puan.get("diagnosis_score"),
            "kanıt puanı": puan.get("evidence_score"),
            "sınırlama puanı": puan.get("limitation_score"),
            "toplam": puan.get("total"),
        }]),
        hide_index=True, width="stretch",
    )
    if puan.get("tespit_notu"):
        stil.yorum(
            "Bu koşuda bozulma kanıtta anlamlı iz bırakmıyor; "
            f"tespit-farkındalıklı puan {puan.get('diagnosis_score_tespit')}. "
            f"Gerekçe: {puan['tespit_notu']}"
        )


def _canli_calistir(kosu_id: str) -> dict | None:
    """Ajani o anda calistirir. Hata turlerini ayırt ederek raporlar."""
    if not os.environ.get("GEMINI_API_KEY"):
        st.error(
            "GEMINI_API_KEY ortam değişkeni tanımlı değil. Canlı çalıştırma "
            "için anahtar gerekir; kayıtlı koşu modu anahtarsız çalışır."
        )
        return None

    from teshis.ajan import ajan as ajan_modulu

    baslangic = time.time()
    with st.status(f"{kosu_id} için canlı teşhis üretiliyor...", expanded=True) as durum:
        try:
            cevap, kayit = ajan_modulu.teshis_uret(kosu_id)
        except Exception as hata:  # noqa: BLE001
            metin = f"{type(hata).__name__}: {hata}".lower()
            if any(k in metin for k in ("quota", "resource_exhausted", "429")):
                durum.update(label="Günlük kota bitti", state="error")
                st.error(
                    "429 RESOURCE_EXHAUSTED — günlük istek kotası aşıldı.\n\n"
                    "Ücretsiz katman 20 istek/gün. Kayıtlı koşu modu çalışmaya "
                    "devam eder."
                )
            elif any(k in metin for k in ("503", "unavailable", "high demand")):
                durum.update(label="Geçici sunucu hatası", state="error")
                st.warning(
                    "503 UNAVAILABLE — sağlayıcıda geçici yoğunluk. Bu hata "
                    "kotayla ilgili değildir; birkaç saniye sonra yeniden "
                    "denenebilir."
                )
            else:
                durum.update(label="Başarısız", state="error")
                st.error(f"{type(hata).__name__}: {hata}")
            return None
        sure = time.time() - baslangic
        durum.update(label=f"Tamamlandı ({sure:.1f} sn, {len(kayit)} araç çağrısı)",
                     state="complete")
    return {"cevap": cevap, "kayit": kayit, "sure": sure}


def _senaryo_ozeti(senaryo: str) -> dict:
    from teshis.degerlendirme.senaryo_ozeti import ozet

    try:
        return ozet(senaryo)
    except Exception:  # noqa: BLE001 - ozet uretilemezse akis durmasin
        return {}


def _kanit_goster(kosu_id: str, kayit: dict) -> None:
    """Ajanin gordugu kanit: kayittan mi, yeniden mi uretildi?

    Bu ayrimi yapmak zorunlu. Kirilim araclarina sonradan gurultu bandi
    alanlari eklendi; dolayisiyla ESKI bir kaydin cevabini bugunku arac
    ciktisiyla yan yana koyup "ajanin gordugu kanit tam olarak budur" demek
    YANLIS olur. Ajan o alanlari hic gormemis olabilir.

    Kayit araç cevaplarini iceriyorsa (snapshot) onlar gosterilir; icermiyorsa
    cikti bugun yeniden uretilir ve bu acikca soylenir.
    """
    kosu_kaydi = (kayit.get("arac_kaydi") or {}).get(kosu_id) or {}
    cagrilar = kosu_kaydi.get("arac_cagrilari") or []
    snapshot = [c for c in cagrilar if "cevap" in c]

    if snapshot:
        st.markdown(
            "Aşağıdaki çıktılar **denemenin kendi kaydından** geliyor: ajanın "
            "o an gördüğü değerlerin birebir kopyası."
        )
        surum = kosu_kaydi.get("arac_surumu")
        if surum:
            stil.yorum(f"Araç sürümü parmak izi: `{surum}`")
        with st.expander("Araç çıktıları (kayıttan)", expanded=False):
            for c in snapshot:
                st.markdown(f"**{c.get('arac')}**")
                st.json(c.get("cevap"), expanded=False)
        return

    st.warning(
        "Bu koşunun kaydında araç **cevapları** saklanmamış — yalnızca hangi "
        "aracın çağrıldığı var. Aşağıdaki çıktı bugünün araçlarıyla yeniden "
        "üretildi. Araçlar deterministiktir ve API harcamaz, ancak sonradan "
        "gürültü bandı alanları eklendiği için ajanın o an gördüğü sürüm "
        "bundan farklı olabilir: bu, **yaklaşık** bir yeniden üretimdir."
    )
    kanit = ajan_araclarini_calistir(kosu_id)
    with st.expander("Araç çıktıları (bugün yeniden üretildi)", expanded=False):
        for ad, deger in kanit.items():
            st.markdown(f"**{ad}**")
            st.json(deger, expanded=False)


def goster() -> None:
    st.title("Ajan")
    st.markdown(
        "Ajana yalnızca anonim metrikler ve kırılım araçları verilir; hangi "
        "koşunun hangi senaryo olduğunu bilmez. Teşhisini kendi seçtiği "
        "kanıtla üretir."
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
        help=("Kayıtlı koşu API harcamaz ve her zaman çalışır. Canlı mod "
              "ücretsiz katman sınırlarına tabidir (20 istek/gün, 5 istek/dk)."),
    )

    st.markdown("---")
    _korluk_paneli(kosu_id, senaryo, acik)

    st.markdown("### Ajana verilen kanıt")
    _kanit_goster(kosu_id, kayit)

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
                "Canlı mod seçildi. 'Ajanı çalıştır' düğmesine basıldığında "
                "sağlayıcıya gerçek bir istek gönderilir."
            )
        return

    cevap = kayit["cevaplar"].get(kosu_id)
    if not cevap:
        st.warning(f"{kosu_id} için kayıtlı cevap yok. Canlı modu deneyebilirsiniz.")
        return

    cagrilar = (kayit["arac_kaydi"].get(kosu_id) or {}).get("arac_cagrilari", [])
    if cagrilar:
        st.markdown("### Araç çağrıları")
        st.dataframe(_arac_zaman_cizelgesi(cagrilar), hide_index=True,
                     width="stretch")
        stil.yorum(
            f"Ajan bu koşuda {len(cagrilar)} araç çağırdı. Hangi kanıtı "
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

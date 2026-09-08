"""Termal Teshis Konsolu — arastirma ve teshis paneli.

Tasarim kararlari
-----------------
**Serbest gezinme, slayt degil.** Bolumler birbirinden bagimsizdir; hicbir
sira dayatilmaz, "sunumu baslat / ileri / geri" yoktur. Sunum sirasinda gelen
soruya gore istenen bolume dogrudan atlanir. Menudeki sira yalnizca bir
oneridir.

**Bolumler ayri dosyalarda.** `demo/bolumler/` altinda her bolum kendi
modulunde durur; bir bolumu degistirmek digerlerine dokunmayi gerektirmez.

**Katmanlar ayri.** Gorsel dil `stil.py`, grafikler `grafik.py`, veri okuma
`data_loader.py` / `veri_seti.py` / `gorseller.py`. Hicbir bolum kendi
grafik temasini veya kendi metrik hesabini yazmaz - ayni kural iki yerde
yasarsa biri geride kalir.

**Konsol olcum yapmaz.** Yalnizca `reports/`, `experiments/`, `val_diagnostic/`
ve `results.csv` icindeki mevcut ciktilari okur. Tek istisna Ajan bolumunun
acikca isaretlenmis "canli calistir" dugmesidir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
sys.path.insert(0, str(KOK / "demo"))

import sistem_durumu  # noqa: E402
import stil  # noqa: E402
from bolumler import (  # noqa: E402
    ajan,
    genel_bakis,
    hata_analizi,
    karsilastirma,
    senaryolar,
    sonuclar,
    veri_ve_model,
)

st.set_page_config(
    page_title="Termal Teşhis Konsolu",
    layout="wide",
    initial_sidebar_state="expanded",
)
stil.uygula()

BOLUMLER = {
    "Genel Bakış": genel_bakis.goster,
    "Veri ve Sağlıklı Model": veri_ve_model.goster,
    "Deney Senaryoları": senaryolar.goster,
    "Karşılaştırma ve Gürültü": karsilastirma.goster,
    "Hata Analizi": hata_analizi.goster,
    "LLM Teşhis Ajanı": ajan.goster,
    "Sonuçlar ve Sınırlamalar": sonuclar.goster,
}

st.sidebar.markdown(
    f'<div style="font-size:1.05rem;font-weight:600;color:{stil.METIN};'
    f'letter-spacing:-.01em">Termal Teşhis Konsolu</div>'
    f'<div style="font-size:.76rem;color:{stil.METIN_SOLUK};margin-bottom:.6rem">'
    f"kontrollü bozulma · ölçüm · kör teşhis</div>",
    unsafe_allow_html=True,
)
secim = st.sidebar.radio("Bölüm", list(BOLUMLER), label_visibility="collapsed")

st.sidebar.markdown("---")
sistem_durumu.goster(st)
st.sidebar.markdown(
    f'<div class="yorum">Bu konsol mevcut ölçüm çıktılarını okur; eğitim veya '
    f"test çalıştırmaz. Tek istisna LLM Teşhis Ajanı bölümündeki "
    f'"canlı çalıştır" düğmesidir.</div>',
    unsafe_allow_html=True,
)

try:
    BOLUMLER[secim]()
except Exception as hata:  # noqa: BLE001
    # Sunum sirasinda tek bir bolumun hatasi butun konsolu goturmemeli.
    st.error(f"'{secim}' bölümü yüklenemedi: {type(hata).__name__}: {hata}")
    with st.expander("Ayrıntı"):
        import traceback

        st.code(traceback.format_exc())

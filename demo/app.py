"""Termal Teshis Konsolu — sunum ve inceleme paneli.

Tasarim kararlari
-----------------
**Serbest gezinme, slayt degil.** Bolumler birbirinden bagimsizdir; hicbir
sira dayatilmaz. Sunum sirasinda gelen soruya gore istenen bolume atlanir.

**Bolumler ayri dosyalarda.** `demo/bolumler/` altinda her bolum kendi
modulunde durur; bir bolumu degistirmek digerlerine dokunmayi gerektirmez.

**Sakin gorunum.** Onceki surum fosforlu terminal estetigi kullaniyordu; bu
bir arastirma panelinden cok gosteriye benziyordu. Yeni dil kirik beyaz zemin,
tek vurgu rengi, emoji ve animasyon yok (bkz. `demo/stil.py`).

**Konsol olcum yapmaz.** Yalnizca `reports/`, `experiments/` ve `results.csv`
icindeki mevcut ciktilari okur. Tek istisna Ajan bolumunun acikca isaretlenmis
"canli calistir" dugmesidir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
sys.path.insert(0, str(KOK / "demo"))

import stil  # noqa: E402
from bolumler import (  # noqa: E402
    ajan,
    genel_bakis,
    hata_analizi,
    karsilastirma,
    senaryolar,
    tasarim,
)

st.set_page_config(
    page_title="Termal Teshis Konsolu",
    layout="wide",
    initial_sidebar_state="expanded",
)
stil.uygula()

BOLUMLER = {
    "Genel Bakış": genel_bakis.goster,
    "Deney Tasarımı ve Sınırlar": tasarim.goster,
    "Senaryolar": senaryolar.goster,
    "Karşılaştırma ve Gürültü": karsilastirma.goster,
    "Hata Analizi": hata_analizi.goster,
    "Ajan": ajan.goster,
}

st.sidebar.markdown("### Termal Teşhis")
secim = st.sidebar.radio("Bölüm", list(BOLUMLER), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Bu konsol mevcut ölçüm çıktılarını okur; eğitim veya test çalıştırmaz. "
    "Tek istisna Ajan bölümündeki 'canlı çalıştır' düğmesidir."
)

try:
    BOLUMLER[secim]()
except Exception as hata:  # noqa: BLE001
    # Sunum sirasinda tek bir bolumun hatasi butun konsolu goturmemeli.
    st.error(f"'{secim}' bölümü yüklenemedi: {type(hata).__name__}: {hata}")
    with st.expander("Ayrıntı"):
        import traceback

        st.code(traceback.format_exc())

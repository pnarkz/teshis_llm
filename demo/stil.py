"""Konsolun gorsel dili: renk sozlesmesi, bilesenler, kucuk gosterim yardimcilari.

Tasarim kararlari
-----------------
Onceki surum kirik beyaz, rapor gorunumlu bir sayfaydi. Yeni dil koyu
lacivert bir **teknik gozlem konsolu**: veri one cikar, metin geri ceker.
Neon, terminal estetigi, animasyon ve emoji yoktur - bu bir arastirma
paneli, bir gosteri degil.

RENK SOZLESMESI (butun sayfalarda AYNI anlam)
---------------------------------------------
    REFERANS   sağlıklı referans koşusu - her grafikte aynı nötr gri-mavi
    ADAY       incelenen koşu - ana vurgu mavisi
    IKINCIL    ikinci karşılaştırma serisi - camgöbeği
    GUCLU      gürültü eşiğini birden fazla metrikte aşan kanıt - yeşil
    UYARI      gürültü içinde kalan / zayıf kanıt - amber
    KRITIK     ağır bozulma, ıraksama, veri kaybı - kırmızı
    NOTR       derecelendirilemeyen durum - gri

Renk asla tek basina anlam tasimaz: her yerde sekil, ikon veya metin
ikinci bir kanal olarak eklenir (renk korlugu).
"""

from __future__ import annotations

import streamlit as st

# --- Palet -----------------------------------------------------------------

ZEMIN = "#0b1220"          # ana arka plan: lacivert-siyah
YUZEY = "#131c2e"          # panel yuzeyi
YUZEY_2 = "#1a2438"        # ikincil yuzey (tablo basligi, hover)
CIZGI = "#243046"          # ince cerceve
METIN = "#e8ecf4"          # kirik beyaz
METIN_SOLUK = "#9aa7bd"    # aciklama grisi

ADAY = "#4c8dff"           # ana vurgu: parlak ama yormayan mavi
IKINCIL = "#2dd4bf"        # camgobegi
REFERANS = "#7d8aa3"       # notr gri-mavi: referans serisi
GUCLU = "#3fb984"          # yesil
UYARI = "#e0a33e"          # amber
KRITIK = "#e5544b"         # kirmizi
NOTR = "#6b7793"

# Kanit seviyesi -> (ekranda gorunen ad, rozet turu, kisa aciklama).
# Derecelendirilen uc seviye ile derecelendirilmeyen bes seviye BILEREK
# ayri tutulur: bir kontrol kosusunu "guclu" diye etiketlemek, projenin
# olcmeye calistigi hatanin ta kendisidir.
SEVIYE = {
    "guclu": ("güçlü", "guclu", "Birden fazla metrik gürültü eşiğini aşıyor"),
    "zayif": ("zayıf", "uyari", "Yalnızca tek metrik eşiği aşıyor"),
    "gurultu icinde": ("gürültü içinde", "uyari",
                       "Hiçbir metrik gürültü eşiğini aşmıyor"),
    "kontrol kosusu": ("kontrol koşusu", "notr",
                       "Bozulma içermez; gürültü tabanını ölçer"),
    "referans": ("referans", "referans", "Kendi ölçeğinin sağlıklı tabanı"),
    "eslenik olcum": ("eşlenik ölçüm", "ikincil",
                      "Aynı ağırlıklar, tek değişen çıkarım ayarı"),
    "esik yok": ("eşik yok", "uyari",
                 "Referansı var ama o ölçekte kontrol koşusu yok"),
    "karsilastirilamaz": ("karşılaştırılamaz", "kritik",
                          "O ölçekte sağlıklı referans hiç yok"),
    "olcum yok": ("ölçüm yok", "notr", "Defterde satırı yok"),
}
DERECELENDIRILEN = ("guclu", "zayif", "gurultu icinde")

ROZET_RENGI = {
    "guclu": GUCLU, "uyari": UYARI, "kritik": KRITIK,
    "ikincil": IKINCIL, "referans": REFERANS, "notr": NOTR, "aday": ADAY,
}

CSS = f"""
<style>
  /* Sunum projeksiyonda yapiliyor ve salonun arkasindan okunmasi gerekiyor.
     Kok font boyutu hic tanimli degildi, yani tarayici varsayilaninda (16px)
     kaliyordu. Buradaki butun olculer `rem` oldugu icin koku buyutmek
     hepsini ORANTILI buyutur - tek tek elle buyutmek yerine tek yerden. */
  html {{ font-size: 18px; }}
  .stApp {{ background: {ZEMIN}; }}
  html, body, [class*="css"], p, li, span, label {{ color: {METIN}; }}
  .stMarkdown p, .stMarkdown li {{ font-size: 1.02rem; line-height: 1.6; }}
  h1, h2, h3, h4 {{ color: {METIN}; font-weight: 600; letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.7rem; margin-bottom: .2rem; }}
  h2 {{ font-size: 1.22rem; margin-top: 1.8rem; }}
  h3 {{ font-size: 1.02rem; margin-top: 1.2rem; }}
  hr {{ border: none; border-top: 1px solid {CIZGI}; margin: 1.4rem 0; }}
  a {{ color: {ADAY}; }}
  code {{ background: {YUZEY_2}; color: {IKINCIL}; padding: .08rem .3rem;
          border-radius: 3px; font-size: .86em; }}

  .ust {{ font-size: .8rem; letter-spacing: .1em; text-transform: uppercase;
          color: {METIN_SOLUK}; margin: .2rem 0 .3rem; }}
  .kutu {{ border: 1px solid {CIZGI}; border-radius: 8px; padding: .85rem 1rem;
           background: {YUZEY}; box-shadow: 0 1px 3px rgba(0,0,0,.35); }}
  .kutu p {{ margin: 0; }}
  .kutu b {{ color: #ffffff; }}

  .kpi {{ border: 1px solid {CIZGI}; border-radius: 8px; padding: .7rem .9rem;
          background: {YUZEY}; height: 100%; }}
  .kpi .etiket {{ font-size: .8rem; letter-spacing: .06em;
                  text-transform: uppercase; color: {METIN_SOLUK}; }}
  .kpi .deger {{ font-size: 1.55rem; font-weight: 600; line-height: 1.25;
                 color: #ffffff; }}
  .kpi .alt {{ font-size: .82rem; color: {METIN_SOLUK}; }}

  .rozet {{ display: inline-block; font-size: .8rem; padding: .14rem .55rem;
            border-radius: 999px; border: 1px solid; margin: 0 .3rem .25rem 0;
            white-space: nowrap; }}
  .yorum {{ font-size: .92rem; color: {METIN_SOLUK}; margin-top: .35rem;
            line-height: 1.5; }}

  .stDataFrame {{ font-size: .92rem; }}
  div[data-testid="stDataFrame"] {{ border: 1px solid {CIZGI};
                                    border-radius: 8px; overflow: hidden; }}
  section[data-testid="stSidebar"] {{ background: #080d18;
                                      border-right: 1px solid {CIZGI}; }}
  section[data-testid="stSidebar"] .yorum {{ font-size: .78rem; }}
  div[data-testid="stExpander"] {{ border: 1px solid {CIZGI};
                                   border-radius: 8px; background: {YUZEY}; }}
  div[data-testid="stMetricValue"] {{ color: #ffffff; }}
  /* Streamlit'in kendi uyari kutulari koyu temada neredeyse okunmuyordu:
     acik zemin varsayimiyla gelen metin rengi arka planla catisiyor. Renk
     sozlesmesiyle AYNI renkler kullanilarak yeniden tanimlanir. */
  div[data-testid="stAlert"] {{ border-radius: 8px; border: 1px solid;
                                background: {YUZEY}; }}
  div[data-testid="stAlert"] p, div[data-testid="stAlert"] li,
  div[data-testid="stAlert"] span, div[data-testid="stAlert"] div {{
      color: {METIN} !important; }}
  div[data-testid="stAlert"] code {{ background: {YUZEY_2}; }}
  div[data-testid="stAlertContentInfo"] {{ border-color: {ADAY}; }}
  div[data-testid="stAlertContentWarning"] {{ border-color: {UYARI}; }}
  div[data-testid="stAlertContentSuccess"] {{ border-color: {GUCLU}; }}
  div[data-testid="stAlertContentError"] {{ border-color: {KRITIK}; }}
  /* Sekmeler. Onceki gap .2rem idi ve sekmelerin kendi ic bosluklari yoktu:
     "Veri seti Etiketli ornekler Saglikli referans model" tek bir cumle gibi
     akiyordu ve sunumda hangisinin sekme oldugu anlasilmiyordu. Ayrik
     durmalari icin hem aralik hem ic bosluk gerekiyor. */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 1.75rem; border-bottom: 1px solid {CIZGI}; margin-bottom: .35rem;
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {METIN_SOLUK}; padding: .35rem .1rem; font-size: 1rem;
    font-weight: 500; letter-spacing: .01em;
  }}
  .stTabs [data-baseweb="tab"]:hover {{ color: {METIN}; }}
  .stTabs [aria-selected="true"] {{ color: {METIN}; font-weight: 600; }}
  .stTabs [data-baseweb="tab-highlight"] {{ background-color: {ADAY}; }}
</style>
"""


def uygula() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --- Bilesenler -------------------------------------------------------------

def ust_baslik(etiket: str) -> None:
    """Kucuk, buyuk harfli bolum etiketi."""
    st.markdown(f'<div class="ust">{etiket}</div>', unsafe_allow_html=True)


def rozet(metin: str, tur: str = "notr") -> str:
    """Durum rozeti; `tur` RENK SOZLESMESI anahtarlarindan biri."""
    renk = ROZET_RENGI.get(tur, NOTR)
    return (f'<span class="rozet" style="color:{renk};border-color:{renk};'
            f'background:{renk}1a">{metin}</span>')


def kutu(icerik: str) -> None:
    st.markdown(f'<div class="kutu">{icerik}</div>', unsafe_allow_html=True)


def yorum(metin: str) -> None:
    """Grafik veya tablo altina okuma notu."""
    st.markdown(f'<div class="yorum">{metin}</div>', unsafe_allow_html=True)


def kpi(etiket: str, deger, alt: str = "") -> None:
    """Tek bir sayi karti. Sayilar HER ZAMAN kaynaktan turetilir."""
    st.markdown(
        f'<div class="kpi"><div class="etiket">{etiket}</div>'
        f'<div class="deger">{deger}</div>'
        f'<div class="alt">{alt}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_satiri(kartlar: list[tuple]) -> None:
    """Yan yana KPI kartlari: [(etiket, deger, alt), ...]"""
    if not kartlar:
        return
    for sutun, kart in zip(st.columns(len(kartlar)), kartlar):
        with sutun:
            kpi(kart[0], kart[1], kart[2] if len(kart) > 2 else "")


# --- Kanit seviyesi ---------------------------------------------------------

def seviye_adi(seviye: str) -> str:
    return SEVIYE.get(seviye, (seviye, "notr", ""))[0]


def seviye_aciklamasi(seviye: str) -> str:
    return SEVIYE.get(seviye, (seviye, "notr", ""))[2]


def guc_rozeti(seviye: str) -> str:
    ad, tur, _ = SEVIYE.get(seviye, (seviye, "notr", ""))
    return rozet(ad, tur)


def seviye_rengi(seviye: str) -> str:
    return ROZET_RENGI.get(SEVIYE.get(seviye, ("", "notr", ""))[1], NOTR)


def fark_metni(deger: float, fark: float, basamak: int = 4) -> str:
    """Sayilari her zaman 'deger (referansa fark)' olarak gösterir."""
    return f"{deger:.{basamak}f}  ({fark:+.{basamak}f})"


def guven_rozeti(deger: str) -> str:
    """Ajanin OZ-BILDIRDIGI guven duzeyi.

    Bilerek rozet, gauge degil: "yuksek" kalibre edilmis bir olasilik degil,
    modelin kendi beyanidir. Yarim daire bir gauge, olculmus bir guven
    yuzdesi izlenimi yaratirdi.
    """
    tur = {"yüksek": "guclu", "yuksek": "guclu",
           "orta": "notr", "düşük": "uyari", "dusuk": "uyari"}.get(
        str(deger).lower(), "notr"
    )
    return rozet(f"öz-bildirim: {deger}", tur)


# --- Grafikler (geriye donuk uyumluluk) -------------------------------------
#
# Grafik uretimi grafik.py'ye tasindi; bu iki ad testler ve eski cagrilar
# icin burada duruyor.

HARITA_RENK_TAVANI = 10.0


def etki_haritasi(veri, x: str, y: str, deger: str, baslik: str = ""):
    from grafik import etki_haritasi as _f

    return _f(veri, x, y, deger, baslik)


def gurultu_bandi_grafigi(veri, senaryo: str = "senaryo", fark: str = "fark",
                          band: str = "band"):
    from grafik import gurultu_bandi_grafigi as _f

    return _f(veri, senaryo, fark)

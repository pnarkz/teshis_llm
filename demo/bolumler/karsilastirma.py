"""Karsilastirma: SAGLIKLI MODEL ↔ secilen hata senaryosu.

Sayfanin mantigi
----------------
Onceki surumde bu sayfa butun kosulari listeleyen bir tabloyla aciliyordu
ve grafikler metne eslik ediyordu. Simdi tersi: **grafik ana anlatici**,
metin yalnizca grafigin ne soyledigini tek cumleyle aciklar. Kullanici bir
senaryo secer ve degisimin ne kadar oldugunu, hangi metrikte oldugunu,
gurultuyu gecip gecmedigini ve hangi kirilimda olustugunu GRAFIKLE gorur.

Butun kosularin toplu tablosu silinmedi; "Deney defteri" acilirina tasindi.

Referans NEDEN kullaniciya sectirilmiyor
----------------------------------------
Arastirma duzeninde karsilastirma tabani zaten belirlidir. Ama tabani
`v00_saglikli` diye SABITLEMEK de yanlis olurdu: `D1n`'in referansi
`v00n`, `D4 last_pt`'ninki `v00_saglikli last_pt`. Sabitleseydik hicbir
bozulma icermeyen kosular yeniden "bozulmus" gorunurdu - bu proje o hatayi
bir kez yapti. Referans OTOMATIK belirlenir (karsilastirilabilirlik.py) ve
basta acikca yazilir.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import grafik
import senaryo_grafikleri as sg
import stil
import veri_seti as vs
from data_loader import (
    error_galleries,
    gorsel_coz,
    load_results,
    referans_galerisi,
)
from teshis.degerlendirme.karsilastirilabilirlik import (
    METRIKLER,
    bozulmasiz_mi,
    kontrol_kosulari,
)
from teshis.degerlendirme.karsilastirilabilirlik import esikler as _grup_esikleri
from teshis.degerlendirme.karsilastirilabilirlik import (
    esik_yoklugu_aciklamasi,
    gurultu_esigi_gecerli_mi,
)
from teshis.degerlendirme.senaryo_ozeti import kanit_gucu, ne_gozlendi, ozet


# --- Ust bant ---------------------------------------------------------------

def _ne_degisti(o: dict) -> str:
    """Tek satirlik "ne degistirildi" - parametrelerden turetilir.

    "Veri azaltildi" gibi belirsiz ifade yerine gercek oran yazilir.
    """
    p = (o["ne_degisti"].get("parametreler") or {})
    if not p:
        return "Bu koşu için parametre kaydı yok."
    return " · ".join(f"{k}: <b>{v}</b>" for k, v in p.items())


def _ust_bant(senaryo: str, o: dict, gozlem: dict) -> None:
    ref = gozlem.get("referans_senaryo") or "— referans yok —"
    k = gozlem.get("kimlik") or {}
    stil.kutu(
        f'<div style="font-size:1.06rem;font-weight:600;color:#fff">'
        f"Sağlıklı Model &nbsp;↔&nbsp; {senaryo}</div>"
        f'<div style="margin-top:.45rem;font-size:.86rem">'
        f"<b>Değişen:</b> {_ne_degisti(o)}</div>"
        f'<div class="yorum" style="margin-top:.35rem">'
        f"Referans: <code>{ref}</code> · sabit tutulanlar: "
        f"<code>{k.get('model', '?')}</code>, "
        f"<code>{k.get('degerlendirme_seti', '?')}</code>, "
        f"<code>{k.get('imgsz_eval', '?')} px</code>, "
        f"<code>{k.get('checkpoint', '?')}.pt</code></div>"
    )


# --- Ust sira: genel metrikler + gurultu bandi ------------------------------

def _genel_metrik_verisi(senaryo: str, gozlem: dict) -> pd.DataFrame:
    ref_ad = gozlem.get("referans_senaryo") or "referans"
    satirlar = []
    for ad, d in gozlem["metrikler"].items():
        if d["referans"] is None:
            continue
        satirlar.append({"metrik": ad, "seri": "sağlıklı referans",
                         "değer": d["referans"], "fark": 0.0})
        satirlar.append({"metrik": ad, "seri": senaryo,
                         "değer": d["deger"], "fark": d["fark"]})
    return pd.DataFrame(satirlar)


def _fark_bandi_verisi(gozlem: dict) -> pd.DataFrame:
    satirlar = []
    for ad, d in gozlem["metrikler"].items():
        esik = d["gurultu_esigi"]
        if d["fark"] is None or esik is None:
            continue
        satirlar.append({
            "metrik": ad, "fark": d["fark"],
            "band_alt": -esik, "band_ust": esik,
            "gürültü eşiği": esik,
            "asiyor": "evet" if d["asiyor"] else "hayir",
        })
    return pd.DataFrame(satirlar)


def _sonuc_cumlesi(senaryo: str, gozlem: dict, yildiz_notu: str) -> str:
    """Sayfanin tek cumlelik hukmu - olcumden turetilir."""
    # DUSUS ve YUKSELIS ayri anlatilir. Ikisi tek listede toplanirsa
    # beklenenin tersine yukselen bir metrik "esigi asiyor" diye, yani
    # bozulma kaniti gibi okunur - D1'de tam olarak bu oluyordu.
    from teshis.degerlendirme.senaryo_ozeti import asan_yone_gore

    dusen, yukselen = asan_yone_gore(gozlem.get("metrikler"))
    n = gozlem.get("kontrol_kosu_sayisi") or 0
    if not gurultu_esigi_gecerli_mi(senaryo):
        return esik_yoklugu_aciklamasi(senaryo)
    if not n:
        return esik_yoklugu_aciklamasi(senaryo)

    ters = (f" <b>{', '.join(yukselen)}</b> ise eşiği aşan bir "
            "<b>yükseliş</b> gösteriyor — beklenen yönün tersi, bozulma "
            "kanıtı değil." if yukselen else "")
    if not dusen:
        return (f"{senaryo}'nun hiçbir genel metriği gürültü eşiğini aşan bir "
                "düşüş göstermiyor: genel metriklere dayanan bir etki iddiası "
                "kurulamaz." + ters)
    kalan = [m for m in METRIKLER if m not in dusen and m not in yukselen]
    return (f"<b>{', '.join(dusen)}</b> gürültü eşiğini aşan bir düşüş "
            "gösteriyor"
            + (f", <b>{', '.join(kalan)}</b> gürültünün içinde kalıyor"
               if kalan else "")
            + "." + ters)


# --- Gorsel kanit -----------------------------------------------------------

# Sira ONEMLI: ilk secenek varsayilan olur.
#
# Varsayilan "saglikli modelden en cok ayrisan"dir, "en fazla kacirilan"
# degil. Sebep olculdu: "en fazla kacirilan" olcutu on senaryonun DOKUZUNDA
# ayni kareyi seciyor - o kare zaten en kalabalik olani ve hangi bozulma
# uygulanirsa uygulansin basa cikiyor. Yani senaryo degistiginde ekranda
# neredeyse ayni goruntu kaliyordu.
#
# Ayrisma olcutu "bu bozulma HANGI kareyi bozdu" sorusunu sorar: saglikli
# modele gore hata ARTISINA bakar ve on senaryoda alti farkli kare secer.
OLCUT = {
    "sağlıklı modelden en çok ayrışan": ("_ayrisma", True),
    "en fazla kaçırılan nesne": ("false_negatives", True),
    "en fazla fazladan kutu": ("false_positives", True),
    "en düşük IoU": ("mean_iou", False),
}
GOSTERILEN_ADAY = 8


def _gorsel_kanit(senaryo: str) -> None:
    galeriler = error_galleries()
    galeri = galeriler.get(senaryo)
    # Gorsel referans SAYISAL referansla AYNI olmali. Sabit
    # `v00_saglikli` kullanmak D1n'i (referansi v00n) ve last_pt
    # kosularini (referansi v00'in last.pt'si) yanlis tabana gore
    # gosteriyordu: ekranda sayilar bir referansa, goruntuler baska
    # bir referansa gore okunuyordu.
    ref_ad, saglikli_galeri = referans_galerisi(senaryo)
    if not galeri:
        st.info(
            "Bu koşu için hata galerisi üretilmemiş, görsel karşılaştırma "
            "yapılamıyor."
        )
        return

    saglikli = {e.get("source"): e for e in (saglikli_galeri.get("entries") or [])}
    kayitlar = [k for k in galeri["entries"] if k.get("source") in saglikli]
    if not kayitlar:
        st.info(
            f"Bu koşunun en sorunlu kareleri {ref_ad} galerisinde yok; "
            "galeriler her koşunun KENDİ en kötü kareleriyle üretildiği için "
            "listeler her zaman örtüşmez."
        )
        return

    a, b = st.columns([3, 2])
    with a:
        olcut = st.radio("Kare seçimi", list(OLCUT), horizontal=True,
                         key=f"olcut_{senaryo}")
    kaynaklar = sorted({str(k.get("source", "")).split("__")[0]
                        for k in kayitlar if k.get("source")})
    with b:
        kaynak = st.selectbox(
            "Kaynak grubu", ["hepsi", *kaynaklar], key=f"kaynak_{senaryo}",
            help=("Kaynak grupları ayrı çekim koşullarını temsil eder; "
                  "bir bozulma bir kaynakta diğerinden çok daha görünür "
                  "olabilir."),
        )
    if kaynak != "hepsi":
        kayitlar = [k for k in kayitlar
                    if str(k.get("source", "")).startswith(kaynak + "__")]
        if not kayitlar:
            st.info(f"{kaynak} grubunda ortak kare yok.")
            return

    alan, ters = OLCUT[olcut]
    if alan == "_ayrisma":
        # Saglikli modele gore en cok BOZULAN kare: iki kaydin hata
        # sayilarindaki artis. Tek basina "en cok hata" yaniltici olurdu -
        # zaten zor olan kareler her modelde kotu.
        for k in kayitlar:
            s = saglikli[k["source"]]
            k["_ayrisma"] = (
                (k.get("false_negatives") or 0) - (s.get("false_negatives") or 0)
                + (k.get("false_positives") or 0) - (s.get("false_positives") or 0)
            )
    sirali = sorted([k for k in kayitlar if alan in k],
                    key=lambda k: k[alan], reverse=ters)
    if not sirali:
        st.info("Bu ölçüte göre sıralanabilir kare bulunamadı.")
        return

    # Tek bir kare gostermek yetmiyordu: olcut ne olursa olsun her zaman
    # siralamanin BIRINCISI ciziliyordu ve baska bir ornege bakmanin yolu
    # yoktu. Aday listesi, secilen karenin neden secildigini de gosterir.
    adaylar = sirali[:GOSTERILEN_ADAY]

    def _aday_etiketi(i_k):
        i, k = i_k
        s = saglikli[k["source"]]
        ek = (f"ayrışma +{k['_ayrisma']}" if alan == "_ayrisma"
              else f"{olcut}: {k.get(alan)}")
        return (f"{i + 1}. {k['source']}  ·  {ek}  ·  kaçırılan "
                f"{s.get('false_negatives')} → {k.get('false_negatives')}")

    secim = st.selectbox(
        "Örnek kare", list(enumerate(adaylar)), format_func=_aday_etiketi,
        key=f"kare_{senaryo}_{olcut}_{kaynak}",
        help="Bu ölçüte göre en üstteki kareler. Başka bir örneğe geçebilirsiniz.",
    )
    kayit = secim[1]
    eslesen = saglikli[kayit["source"]]

    a, b = st.columns(2)
    with a:
        stil.ust_baslik(f"referans modeli — {ref_ad}")
        yol = gorsel_coz((saglikli_galeri.get("folder") or galeri["folder"])
                         / eslesen["image"])
        if yol:
            st.image(str(yol), width="stretch")
        else:
            st.info(f"{ref_ad} referansının bu karesi bulunamadı.")
        stil.yorum(f"kaçırılan {eslesen.get('false_negatives')} · "
                   f"fazladan {eslesen.get('false_positives')} · "
                   f"IoU {eslesen.get('mean_iou', 0):.2f}")
    with b:
        stil.ust_baslik(f"{senaryo} modeli")
        yol = gorsel_coz(galeri["folder"] / kayit["image"])
        if yol:
            st.image(str(yol), width="stretch")
        else:
            st.warning("Bu karenin görseli bulunamadı.")
        stil.yorum(f"kaçırılan {kayit.get('false_negatives')} · "
                   f"fazladan {kayit.get('false_positives')} · "
                   f"IoU {kayit.get('mean_iou', 0):.2f}")

    stil.yorum(
        "Aynı termal kare, iki model. Yeşil = gerçek etiket, kırmızı = modelin "
        "tahmini; yeşilin yanında kırmızı yoksa o nesne kaçırılmıştır. "
        f"Bu karede kaçırma "
        f"<b>{eslesen.get('false_negatives')} → {kayit.get('false_negatives')}</b>, "
        f"fazladan kutu "
        f"<b>{eslesen.get('false_positives')} → {kayit.get('false_positives')}</b>."
    )
    if alan == "_ayrisma":
        stil.yorum(
            "Varsayılan ölçüt <b>ayrışma</b>: sağlıklı modele göre hata "
            "<b>artışı</b>. \"En fazla kaçırılan nesne\" ölçütü senaryodan "
            "senaryoya neredeyse hep aynı kareyi seçer — o kare zaten en "
            "kalabalık olanıdır ve hangi bozulma uygulanırsa uygulansın başa "
            "çıkar. Ayrışma ise \"bu bozulma hangi kareyi bozdu\" sorusunu "
            "sorar."
        )


# --- Deney defteri (ikincil) ------------------------------------------------

def _defter_tablosu(sonuclar: pd.DataFrame) -> pd.DataFrame:
    satirlar = []
    for _, r in sonuclar.iterrows():
        s = r["scenario"]
        g = ne_gozlendi(s)
        if not g:
            continue
        m, k = g["metrikler"], (g.get("kimlik") or {})
        satirlar.append({
            "senaryo": s,
            "ölçek": " | ".join([
                k.get("model", "?"), k.get("degerlendirme_seti", "?"),
                f"{k.get('imgsz_eval', '?')} px", f"{k.get('checkpoint', '?')}.pt",
            ]),
            "referansı": g.get("referans_senaryo") or "—",
            "mAP50": m["mAP50"]["deger"],
            "Δ mAP50": m["mAP50"]["fark"],
            "Δ precision": m["precision"]["fark"],
            "Δ recall": m["recall"]["fark"],
            "eşiği aşan": ", ".join(g["asan_metrikler"]) or "-",
            "kanıt": stil.seviye_adi(kanit_gucu(s)["seviye"]),
        })
    df = pd.DataFrame(satirlar)
    # Referansi olmayan kosularda fark HESAPLANMAZ. None birakilirsa sutun
    # object tipine duser ve ekranda harfi harfine "None" yazar; sayisal
    # tutulunca bos hucre olur - dogru okuma da budur: deger yok, sifir degil.
    for sutun in ("Δ mAP50", "Δ precision", "Δ recall"):
        df[sutun] = pd.to_numeric(df[sutun], errors="coerce")
    return df


def _erken_durdurma(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Bozulmasiz kosularin durdugu ve en iyi checkpoint'i verdigi epoch."""
    import csv as _csv

    from data_loader import run_dir_for

    satirlar = []
    for _, r in sonuclar.iterrows():
        ad = str(r["scenario"])
        if not bozulmasiz_mi(ad) or not str(r.get("weights_path", "")).endswith("best.pt"):
            continue
        dizin = run_dir_for(r)
        yol = (dizin / "results.csv") if dizin else None
        if not yol or not yol.is_file():
            continue
        with yol.open(encoding="utf-8") as f:
            kayit = [{k.strip(): v for k, v in s.items()} for s in _csv.DictReader(f)]
        ciftler = [(int(float(s["epoch"])), float(s["metrics/mAP50(B)"]))
                   for s in kayit if s.get("metrics/mAP50(B)")]
        if not ciftler:
            continue
        en_iyi = max(ciftler, key=lambda c: c[1])
        satirlar.append({
            "koşu": ad, "seed": int(r["seed"]),
            "durduğu epoch": len(kayit), "en iyi epoch": en_iyi[0],
            "en iyi mAP50 (eğitim val)": round(en_iyi[1], 4),
        })
    return pd.DataFrame(sorted(satirlar, key=lambda s: s["seed"]))


def _checkpoint_ciftleri(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Ayni kosunun best.pt / last.pt satirlarini yan yana koyar.

    Her satir KENDI checkpoint ailesinin referansiyla karsilastirilir;
    onceden ikisi de best.pt referansiyla tartiliyordu ve her last.pt satiri
    otomatik olarak "bozulmus" gorunuyordu.
    """
    adlar = set(sonuclar["scenario"])
    satirlar = []
    for ad in sorted(adlar):
        if not ad.endswith(" last_pt"):
            continue
        taban = ad[: -len(" last_pt")]
        if taban not in adlar:
            continue
        for etiket, s in ((f"{taban} best.pt", taban), (f"{taban} last.pt", ad)):
            g = ne_gozlendi(s)
            if not g:
                continue
            m = g["metrikler"]["mAP50"]
            satirlar.append({
                "koşu": etiket, "mAP50": m["deger"],
                "referansı": g.get("referans_senaryo") or "—",
                "Δ kendi referansına": m["fark"],
                "eşiği aşıyor": {True: "evet", False: "hayır"}.get(
                    m["asiyor"], "eşik yok"),
            })
    return pd.DataFrame(satirlar)


def _band_verisi(sonuclar: pd.DataFrame, metrik: str) -> pd.DataFrame:
    satirlar = []
    for _, r in sonuclar.iterrows():
        s = str(r["scenario"])
        g = ne_gozlendi(s)
        if not g or metrik not in g["metrikler"]:
            continue
        d = g["metrikler"][metrik]
        esik = d["gurultu_esigi"]
        if esik is None or d["fark"] is None or bozulmasiz_mi(s):
            continue
        satirlar.append({"senaryo": s, "fark": d["fark"], "band_alt": -esik,
                         "band_ust": esik,
                         "asiyor": "evet" if d["asiyor"] else "hayir"})
    return pd.DataFrame(sorted(satirlar, key=lambda s: s["fark"]))


def _esikler(kontroller: list[str], ref: str, defter: dict) -> dict[str, float]:
    return {m: max(abs(float(defter[c][m]) - float(defter[ref][m]))
                   for c in kontroller) for m in METRIKLER}


def _esik_buyumesi(sonuclar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Ilk kontrolle olculen esik vs butun kontrollerle olculen esik.

    Iki tablo da elle yazilmisti ve kontrol sayisi degistiginde geride
    kalirdi. Simdi ikisi de defterden turetilir: defter append-only oldugu
    icin satir sirasi kronolojiktir, ilk C satiri ilk kontrol kosusudur.
    """
    defter = {str(r["scenario"]): r for _, r in sonuclar.iterrows()}
    ref = "v00_saglikli"
    kontroller = kontrol_kosulari("D1")
    if not kontroller or ref not in defter:
        return pd.DataFrame(), pd.DataFrame(), 0
    sirali = [ad for ad in sonuclar["scenario"] if ad in kontroller]
    ilk, hepsi = _esikler(sirali[:1], ref, defter), _esikler(sirali, ref, defter)

    buyume = pd.DataFrame([{
        "metrik": m,
        f"n=1 eşiği ({sirali[0]})": round(ilk[m], 4),
        f"n={len(sirali)} eşiği": round(hepsi[m], 4),
        "kat": f"{hepsi[m] / ilk[m]:.1f}x" if ilk[m] else "—",
    } for m in METRIKLER])

    zayiflayan = []
    for _, r in sonuclar.iterrows():
        ad = str(r["scenario"])
        g = ne_gozlendi(ad)
        if (not g or g["karsilastirma_turu"] != "ayni_olcek"
                or bozulmasiz_mi(ad) or g["referans_senaryo"] != ref):
            continue
        once = [m for m in METRIKLER
                if abs(float(r[m]) - float(defter[ref][m])) > ilk[m]]
        sonra = g["asan_metrikler"]
        kaybedilen = [m for m in once if m not in sonra]
        if kaybedilen:
            zayiflayan.append({
                "senaryo": ad,
                "ilk yorum (n=1 eşiğiyle)": f"{', '.join(once)} etkilendi",
                "düzeltilmiş yorum": (f"yalnızca {', '.join(sonra)} kaldı"
                                      if sonra else "hiçbir kanıt kalmadı"),
                "kaybettiği": ", ".join(kaybedilen),
            })
    return buyume, pd.DataFrame(zayiflayan), len(sirali)


def _best_last_dususu(sonuclar: pd.DataFrame):
    defter = {str(r["scenario"]): float(r["mAP50"]) for _, r in sonuclar.iterrows()}
    taban, dususler = None, []
    for ad, deger in defter.items():
        if not ad.endswith(" last_pt"):
            continue
        temel = ad[: -len(" last_pt")]
        if temel not in defter:
            continue
        dusus = deger - defter[temel]
        if bozulmasiz_mi(temel):
            taban = dusus
        else:
            dususler.append((temel, dusus))
    return taban, sorted(dususler, key=lambda c: c[1], reverse=True)


def _gurultu_tablosu() -> pd.DataFrame:
    from teshis.degerlendirme.gurultu import alt_grup_bandi

    satirlar = []
    for alan, gruplar in alt_grup_bandi().items():
        for grup, d in gruplar.items():
            satirlar.append({
                "kırılım": alan.replace("_recall", ""),
                "grup": vs.BANT_ADI.get(grup, vs.KOD_ADI.get(grup, grup)),
                "bbox": d["bbox_n"], "band": d["band"],
                "std": d["std"], "koşu": d["n_kosu"],
            })
    return pd.DataFrame(sorted(satirlar, key=lambda s: -s["band"]))


def _deney_defteri(sonuclar: pd.DataFrame) -> None:
    st.markdown(
        "Bütün koşuların toplu görünümü. Her koşu **kendi ölçeğindeki** "
        "sağlıklı referansla karşılaştırılır: aynı başlangıç modeli, aynı "
        "değerlendirme kümesi, aynı çözünürlük, aynı checkpoint."
    )
    df = _defter_tablosu(sonuclar)
    olcekler = ["hepsi"] + sorted({s for s in df["ölçek"] if s})
    a, b = st.columns([3, 2])
    with a:
        olcek = st.selectbox("Ölçek", olcekler, key="defter_olcek")
    with b:
        sadece = st.checkbox("Yalnızca eşiği aşanlar", key="defter_asan")
    gosterilen = df if olcek == "hepsi" else df[df["ölçek"] == olcek]
    if sadece:
        gosterilen = gosterilen[gosterilen["eşiği aşan"] != "-"]
    st.dataframe(gosterilen.drop(columns=["ölçek"]), hide_index=True,
                 width="stretch", height=420)
    st.download_button("Tabloyu CSV olarak indir",
                       gosterilen.to_csv(index=False).encode("utf-8-sig"),
                       file_name="deney_defteri.csv", mime="text/csv")

    st.markdown("### Bütün senaryolar tek grafikte")
    metrik = st.radio("Metrik", ["mAP50", "precision", "recall"],
                      horizontal=True, key="defter_metrik")
    band = _band_verisi(sonuclar, metrik)
    if not band.empty:
        st.altair_chart(grafik.gurultu_bandi_grafigi(band), width="stretch")
        stil.yorum(
            "Dolu mavi kareler bandı aşıyor; içi açık turuncu daireler bandın "
            "içinde. Kontrol koşuları ölçüm aracıdır, ölçüm nesnesi değil — "
            "grafiğe girmezler."
        )

    st.markdown("### Gürültü tabanı ölçülünce ne değişti")
    buyume, zayiflayan, n = _esik_buyumesi(sonuclar)
    st.markdown(
        f"İlk ölçüm tek bir kontrol koşusuna dayanıyordu. {n} kontrole "
        "çıkıldığında eşikler büyüdü ve bazı iddialar geri çekildi:"
    )
    a, b = st.columns(2)
    with a:
        st.dataframe(buyume, hide_index=True, width="stretch")
    with b:
        st.dataframe(zayiflayan, hide_index=True, width="stretch")
    stil.yorum(
        "İki tablo da defterden türetilir; yeni bir kontrol koşusu "
        "eklendiğinde kendiliğinden güncellenir. Elle yazıldıkları dönemde "
        "\"beş iddia\" diyorlardı ve E1 ile E2 atlanmıştı."
    )

    st.markdown("### Alt grup gürültü bandı")
    st.dataframe(_gurultu_tablosu(), hide_index=True, width="stretch")
    stil.yorum(
        "Bu yalnızca küçük örneklem sorunu değil: `termal` grubu 858 bbox "
        "taşır ama bandı `hituav`ın (2.165 bbox) bandının on katından fazladır."
    )

    st.markdown("### Checkpoint ve seed")
    a, b = st.columns(2)
    with a:
        st.dataframe(_checkpoint_ciftleri(sonuclar), hide_index=True,
                     width="stretch")
    with b:
        st.dataframe(_erken_durdurma(sonuclar), hide_index=True, width="stretch")
    taban, dususler = _best_last_dususu(sonuclar)
    if taban is not None:
        esik = _grup_esikleri("D1")["mAP50"] or 0.0
        sert = [f"{a} ({d:+.4f})" for a, d in dususler if d < taban - esik]
        yakin = [f"{a} ({d:+.4f})" for a, d in dususler if d >= taban - esik]
        stil.kutu(
            "<b>Son checkpoint düşüşünün de bir tabanı var.</b> Sağlıklı "
            f"referansın kendisi best→last geçerken <b>{taban:+.4f}</b> "
            "düşüyor. Tabanı mAP50 gürültü eşiği "
            f"({esik:.4f}) kadar aşmayanlar: {', '.join(yakin) or 'yok'}. "
            f"Belirgin ayrışanlar: {', '.join(sert) or 'yok'}."
        )


# --- Sayfa ------------------------------------------------------------------

def _varyant_etiketi(ana: str, varyant: str) -> str:
    """Varyantin ana kosudan hangi alanda ayrildigi - kimlikten turetilir."""
    from teshis.degerlendirme.karsilastirilabilirlik import kimlik

    a, b = kimlik(ana), kimlik(varyant)
    if a is None or b is None:
        return "varyant"
    # Etiket NE OLDUGUNU degil NE ANLAMA GELDIGINI soyler. "checkpoint: last"
    # teknik olarak dogruydu ama izleyiciye hicbir sey anlatmiyordu: ayni
    # egitimin son epoch'uyla raporlanmis hali oldugu gorunmuyordu.
    if a.checkpoint != b.checkpoint:
        return ("aynı eğitim, son epoch ile raporlandı"
                if b.checkpoint == "last"
                else "aynı eğitim, en iyi epoch ile raporlandı")
    if a.model != b.model:
        return f"farklı başlangıç modeli ({b.model})"
    if a.imgsz_eval != b.imgsz_eval:
        return f"farklı çıkarım çözünürlüğü ({b.imgsz_eval} px)"
    if a.degerlendirme_seti != b.degerlendirme_seti:
        return f"farklı değerlendirme kümesi ({b.degerlendirme_seti})"
    return "yalnızca rastgelelik tohumu farklı"


def goster() -> None:
    sonuclar = load_results()
    st.title("Karşılaştırma")

    # SENARYO once, KOSU sonra. Onceki surum tek bir listede 20 kosu
    # gosteriyordu ve "hata senaryosu" diye etiketliyordu - oysa listede
    # D4 ile "D4 last_pt" yan yana, ayni seviyedeymis gibi duruyordu.
    import katalog

    hepsi = katalog.senaryolar()
    senaryolar = [x for x in hepsi if x["ana_kosu"]]
    # Kosusu olmayan senaryolar seciciye giremez ama SESSIZCE de dusmemeli:
    # Genel Bakis 14 senaryo sayiyor, burada 13 gorunuyordu ve aradaki fark
    # aciklanmiyordu.
    kosusuz = [x for x in hepsi if not x["ana_kosu"]]
    a, b = st.columns([3, 3])
    with a:
        secilen = st.selectbox(
            "Hata senaryosu", senaryolar,
            index=next((i for i, x in enumerate(senaryolar)
                        if x["kod"] == "D4"), 0),
            format_func=lambda x: f"{x['kod']} · {x['ad']}",
        )
    kosular = [secilen["ana_kosu"], *secilen["varyantlar"]]
    with b:
        senaryo = st.selectbox(
            "Koşu (ana koşu / varyant)", kosular,
            format_func=lambda k: (
                f"{k}  ·  ana koşu (en iyi epoch)" if k == secilen["ana_kosu"]
                else f"{k}  ·  {_varyant_etiketi(secilen['ana_kosu'], k)}"
            ),
        )
    if len(kosular) > 1:
        stil.yorum(
            f"<b>{secilen['kod']} senaryosunun {len(kosular)} kaydı var.</b> "
            "Ayrı senaryolar değil, <b>aynı hipotezin farklı kayıtları</b>: "
            "ana koşu, o eğitimin en iyi epoch'uyla (best.pt) raporlanmış "
            "halidir — normalde bildirilen sayı budur. Varyantlar aynı "
            "hipotezi başka bir kayıtla gösterir: " + ", ".join(
                f"<b>{k}</b> ({_varyant_etiketi(secilen['ana_kosu'], k)})"
                for k in secilen["varyantlar"]
            ) + ". Son epoch varyantları özellikle önemli, çünkü bir arıza "
            "yalnızca son checkpoint'te görünüyor olabilir — E1 tam olarak "
            "bunu gösteriyor."
        )
    if kosusuz:
        stil.yorum(
            "Katalogdaki " + str(len(hepsi)) + " senaryodan "
            + ", ".join(f"<b>{x['kod']}</b>" for x in kosusuz)
            + " bu listede yok: ölçülebilir bir koşu üretmedi, dolayısıyla "
            "karşılaştırılacak bir metriği de yok. Ayrıntı: Deney Senaryoları."
        )

    o = ozet(senaryo)
    gozlem = o["ne_gozlendi"]
    _ust_bant(senaryo, o, gozlem)
    if o["ne_olcuyor"]:
        stil.yorum(o["ne_olcuyor"].strip())

    st.write("")
    a, b = st.columns(2)
    with a:
        veri = _genel_metrik_verisi(senaryo, gozlem)
        if veri.empty:
            st.info(
                "Bu koşunun kendi ölçeğinde sağlıklı bir referansı yok, "
                "genel metrik karşılaştırması yapılamıyor."
            )
        else:
            st.altair_chart(
                grafik.degerli_gruplu_bar(veri, "metrik", "değer", "seri",
                                          baslik="Genel metrikler",
                                          alan_adi="değer"),
                width="stretch",
            )
            stil.yorum(
                "Y ekseni sıfırdan başlar; kırpılmış bir eksen küçük farkları "
                "olduklarından dramatik gösterirdi."
            )
    with b:
        band = _fark_bandi_verisi(gozlem)
        if band.empty:
            # Metin TEK KAYNAKTAN gelir. Burada ayri bir cumle yazilmisti ve
            # eslenik olcum duzeltmesi ona ulasmadi: E4 secildiginde kutu
            # hala "rastgelelikten ayrilamiyor" diyordu.
            st.info(esik_yoklugu_aciklamasi(senaryo))
        else:
            st.altair_chart(
                grafik.gurultu_bandi_grafigi(band, senaryo="metrik"),
                width="stretch",
            )
            stil.yorum(
                "Gri kuşak, hiçbir bozulma içermeyen koşular arasında "
                "gözlenen yayılım. Kuşağın içindeki bir nokta saf "
                "rastgelelikten ayırt edilemez."
            )

    tur = sg.yildiz_turu(senaryo)
    if tur:
        st.markdown(f"### {sg.BASLIK.get(tur, 'Kırılım')}")
        cizim, notu = sg.ciz(senaryo, tur)
        if cizim is None:
            st.info(notu)
        else:
            st.altair_chart(cizim, width="stretch")
            stil.yorum(notu)

    st.markdown("### Görsel kanıt: aynı kare, iki model")
    _gorsel_kanit(senaryo)

    st.markdown("---")
    guc = o["kanit_gucu"]
    stil.kutu(
        stil.guc_rozeti(guc["seviye"]) + " &nbsp;"
        + _sonuc_cumlesi(senaryo, gozlem, "")
    )

    with st.expander("Deney ayrıntıları (parametreler, ham metrikler, sınırlamalar)"):
        _ayrintilar(senaryo, o, gozlem)

    with st.expander("Deney defteri — bütün koşular"):
        _deney_defteri(sonuclar)


def _ayrintilar(senaryo: str, o: dict, gozlem: dict) -> None:
    from data_loader import evidence_for

    a, b = st.columns(2)
    with a:
        stil.ust_baslik("ne değiştirildi")
        d = o["ne_degisti"]
        satir = f"<b>Tür:</b> {d.get('tur') or '—'}<br>" + _ne_degisti(o)
        if d.get("veri_surumu"):
            satir += f"<br><b>Veri sürümü:</b> <code>{d['veri_surumu']}</code>"
        stil.kutu(satir)
    with b:
        stil.ust_baslik("referansla ortak olan")
        stil.kutu("<br>".join(o["ne_sabit_kaldi"]))

    if o["beklenen_etki"]:
        stil.ust_baslik("deneyden önce ne bekleniyordu")
        stil.kutu(" ".join(str(o["beklenen_etki"]).split()))

    st.markdown("**Karşılaştırma tablosu**")
    satirlar = []
    for ad, d in gozlem["metrikler"].items():
        ref, deger, fark = d["referans"], d["deger"], d["fark"]
        satirlar.append({
            "metrik": ad,
            "sağlıklı referans": ref,
            "senaryo": deger,
            # "mutlak fark" YANLIS bir baslikti: gosterilen degerler
            # negatif olabiliyor, mutlak fark ise negatif olamaz.
            "fark (senaryo − referans)": fark,
            "göreli değişim (%)": (None if not ref or fark is None
                                   else round(100 * fark / ref, 2)),
            "gürültü eşiği": d["gurultu_esigi"],
            "karar": {True: "eşiği aşıyor", False: "gürültü içinde"}.get(
                d["asiyor"], "eşik yok"),
        })
    st.dataframe(pd.DataFrame(satirlar), hide_index=True, width="stretch")

    if not gozlem["asan_metrikler"] and gozlem["kontrol_kosu_sayisi"]:
        st.markdown("**Fark neden görülmemiş olabilir?**")
        for baslik, metin in GORULMEME_NEDENLERI:
            st.markdown(f"- **{baslik}** — {metin}")

    st.markdown("**Sınırlamalar**")
    for s in o["sinirlamalar"]:
        st.markdown(f"- {s}")

    with st.expander("Ham metrik dosyası (JSON)"):
        st.json(evidence_for(senaryo))


# Fark GORULMEDIGINDE olasi nedenler. Her madde bu projede GERCEKTEN
# gozlenmis bir mekanizmadir, spekulasyon degil.
GORULMEME_NEDENLERI = [
    ("Modelin önceden öğrenilmiş özellikleri",
     "Eğitim verisindeki kayıp, önceki temsili silmeye yetmemiş olabilir. "
     "D1'in main_model kurgusunda etkisiz, yolo26n kurgusunda güçlü çıkması "
     "bunun doğrudan kanıtıdır."),
    ("Veri fazlalığı — bozulma soğuruluyor",
     "D3 (nadir sınıflar) güçlü etki üretirken D3b (bol örnekli taşıt/insan) "
     "aynı karışıklığı büyük ölçüde soğurdu."),
    ("Bozulma yalnızca belirli bir alt grupta görünüyor",
     "D4'te <16 px bandının recall'ı yarıdan fazla düşerken toplam mAP50 "
     "farkı küçük kaldı."),
    ("Ölçüm seed gürültüsünün içinde kalıyor",
     "Fark gerçek olabilir ama bozulmasız koşular arasında da görülen "
     "büyüklükte olabilir; D6b tam bu nedenle bulgu olmaktan çıktı."),
    ("Checkpoint seçimi problemi gizliyor",
     "En iyi checkpoint sağlıklı görünürken son checkpoint arızayı gösterir "
     "(E1 ve D5)."),
    ("Nadir sınıfta örnek sayısı yetersiz",
     "UAP (15 bbox) ve UAI (17 bbox) üzerinde 'etki yok' sonucu da 'etki var' "
     "sonucu kadar belirsizdir."),
    ("Deney hipotezi test etmeye yetmiyor",
     "E3'te öğrenme oranı 100 kat yükseltildiğinde kararsızlık değil tam "
     "ıraksama üretildi ve hipotez hiç test edilemedi; E3b bu yüzden 10 katla "
     "tekrarlandı."),
]

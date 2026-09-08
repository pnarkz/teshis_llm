"""Karsilastirma ve Gurultu: her kosu KENDI olceginde, gurultuye karsi tartilmis.

Bu sayfanin manseti bir sayi degil bir DUZELTMEDIR: gurultu tabani olculunce
bes iddia zayifladi ve bir senaryo bulgu olmaktan cikti. Bunu gizlemek yerine
one koymak, olcumun calistiginin kanitidir.

Sayfa iki kez duzeltildi. Ikincisi daha ciddiydi: butun kosular tek bir
referansla (v00) karsilastiriliyordu, oysa bazilari farkli baslangic modeli,
farkli degerlendirme kumesi veya farkli checkpoint tasiyor. O yuzden
`v00_saglikli last_pt` - icinde hicbir bozulma olmayan bir kosu - "guclu
bozulma kaniti" olarak etiketleniyordu. Artik her kosu yalnizca kendi
olcegindeki referansla karsilastirilir (karsilastirilabilirlik.py).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import stil
from data_loader import load_results
from teshis.degerlendirme.karsilastirilabilirlik import (
    METRIKLER,
    bozulmasiz_mi,
    kontrol_kosulari,
)
from teshis.degerlendirme.karsilastirilabilirlik import esikler as _grup_esikleri
from teshis.degerlendirme.senaryo_ozeti import kanit_gucu, ne_gozlendi


def _tablo(sonuclar: pd.DataFrame) -> pd.DataFrame:
    satirlar = []
    for _, r in sonuclar.iterrows():
        senaryo = r["scenario"]
        g = ne_gozlendi(senaryo)
        if not g:
            continue
        m = g["metrikler"]
        k = g.get("kimlik") or {}
        satirlar.append({
            "senaryo": senaryo,
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
            "kanıt": stil.seviye_adi(kanit_gucu(senaryo)["seviye"]),
        })
    df = pd.DataFrame(satirlar)
    # Referansi olmayan kosularda fark HESAPLANMAZ. None birakilirsa sutun
    # object tipine duser ve ekranda harfi harfine "None" yazar; sayisal
    # tutulunca bos hucre olarak gorunur - dogru okuma da budur: deger yok,
    # sifir degil.
    for sutun in ("Δ mAP50", "Δ precision", "Δ recall"):
        df[sutun] = pd.to_numeric(df[sutun], errors="coerce")
    return df


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
        if not bozulmasiz_mi(ad) or not str(r.get("weights_path", "")).endswith("best.pt"):
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
            "koşu": ad,
            "seed": int(r["seed"]),
            "durduğu epoch": len(kayit),
            "en iyi epoch": en_iyi[0],
            "en iyi mAP50 (eğitim val)": round(en_iyi[1], 4),
        })
    return pd.DataFrame(sorted(satirlar, key=lambda s: s["seed"]))


def _checkpoint_ciftleri(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Ayni kosunun best.pt / last.pt satirlarini yan yana koyar.

    Her satir KENDI checkpoint ailesinin referansiyla karsilastirilir:
    best.pt satiri v00'in best.pt'siyle, last.pt satiri v00'in last.pt'siyle.
    Onceden ikisi de best.pt referansiyla tartiliyordu ve her last.pt satiri
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
        for etiket, senaryo in ((f"{taban} best.pt", taban), (f"{taban} last.pt", ad)):
            g = ne_gozlendi(senaryo)
            if not g:
                continue
            m = g["metrikler"]["mAP50"]
            satirlar.append({
                "koşu": etiket,
                "mAP50": m["deger"],
                "referansı": g.get("referans_senaryo") or "—",
                "Δ kendi referansına": m["fark"],
                "eşiği aşıyor": {True: "evet", False: "hayır"}.get(m["asiyor"], "eşik yok"),
            })
    return pd.DataFrame(satirlar)


def _band_verisi(sonuclar: pd.DataFrame, metrik: str) -> pd.DataFrame:
    """Her senaryonun farkini ve o metrigin gürültü bandini bir arada verir."""
    satirlar = []
    for _, r in sonuclar.iterrows():
        senaryo = str(r["scenario"])
        g = ne_gozlendi(senaryo)
        if not g or metrik not in g["metrikler"]:
            continue
        d = g["metrikler"][metrik]
        esik = d["gurultu_esigi"]
        if esik is None or d["fark"] is None or bozulmasiz_mi(senaryo):
            continue
        satirlar.append({
            "senaryo": senaryo,
            "fark": d["fark"],
            "band_alt": -esik,
            "band_ust": esik,
            "asiyor": "evet" if d["asiyor"] else "hayir",
        })
    return pd.DataFrame(sorted(satirlar, key=lambda s: s["fark"]))


# --- Gurultu tabani buyudugunde ne degisti? Ikisi de TURETILIR --------------

def _esikler(kontroller: list[str], ref: str, defter: dict) -> dict[str, float]:
    return {
        m: max(abs(float(defter[c][m]) - float(defter[ref][m])) for c in kontroller)
        for m in METRIKLER
    }


def _esik_buyumesi(sonuclar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Ilk kontrolle olculen esik vs butun kontrollerle olculen esik.

    Iki tablo da elle yazilmisti ve kontrol sayisi degistiginde geride
    kalirdi. Simdi ikisi de defterden turetilir: defter append-only oldugu
    icin satir sirasi kronolojiktir, ilk C satiri ilk kontrol kosusudur.
    """
    defter = {str(r["scenario"]): r for _, r in sonuclar.iterrows()}
    ref = "v00_saglikli"
    kontroller = kontrol_kosulari("D1")           # v00 ile ayni olcekteki C kosulari
    if not kontroller or ref not in defter:
        return pd.DataFrame(), pd.DataFrame(), 0
    sirali = [ad for ad in sonuclar["scenario"] if ad in kontroller]
    ilk = _esikler(sirali[:1], ref, defter)
    hepsi = _esikler(sirali, ref, defter)

    buyume = pd.DataFrame([
        {
            "metrik": m,
            f"n=1 eşiği ({sirali[0]})": round(ilk[m], 4),
            f"n={len(sirali)} eşiği": round(hepsi[m], 4),
            "kat": f"{hepsi[m] / ilk[m]:.1f}x" if ilk[m] else "—",
        }
        for m in METRIKLER
    ])

    # Hangi iddialar zayifladi? Ayni kosular, yalnizca esik degisti.
    zayiflayan = []
    for _, r in sonuclar.iterrows():
        ad = str(r["scenario"])
        g = ne_gozlendi(ad)
        if not g or g["karsilastirma_turu"] != "ayni_olcek" or bozulmasiz_mi(ad):
            continue
        if g["referans_senaryo"] != ref:
            continue
        once = [m for m in METRIKLER
                if abs(float(r[m]) - float(defter[ref][m])) > ilk[m]]
        sonra = g["asan_metrikler"]
        kaybedilen = [m for m in once if m not in sonra]
        if kaybedilen:
            zayiflayan.append({
                "senaryo": ad,
                "kaybettiği metrik": ", ".join(kaybedilen),
                "geriye kalan": ", ".join(sonra) or "hiçbir şey",
            })
    return buyume, pd.DataFrame(zayiflayan), len(sirali)


def _best_last_dususu(sonuclar: pd.DataFrame) -> tuple[float | None, list[tuple[str, float]]]:
    """Her kosunun best.pt -> last.pt mAP50 dususu; taban saglikli referansinki.

    Bu kutuda sayilar elle yaziliydi ve checkpoint tablosu kendi referansina
    gecince onunla celisir hale geldi. Iki yerde yasayan bir kural er gec
    ayrisir - bu projede tekrarlayan hata oruntusu tam olarak budur.
    """
    defter = {str(r["scenario"]): float(r["mAP50"]) for _, r in sonuclar.iterrows()}
    taban = None
    dususler = []
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


def goster() -> None:
    sonuclar = load_results()
    st.title("Karşılaştırma ve Gürültü")

    st.markdown(
        "Her koşu **kendi ölçeğindeki** sağlıklı referansla karşılaştırılır: "
        "aynı başlangıç modeli, aynı değerlendirme kümesi, aynı çözünürlük, "
        "aynı checkpoint. Bu dört alandan biri farklıysa fark, bozulmanın "
        "değil o alanın etkisini taşır."
    )
    stil.kutu(
        "<b>Bu kural sonradan eklendi ve bir hatayı ortaya çıkardı.</b> "
        "Önceden bütün koşular v00'in best.pt'siyle karşılaştırılıyordu; "
        "bu yüzden <code>v00_saglikli last_pt</code> — içinde hiçbir bozulma "
        "olmayan bir koşu — üç metrikte \"eşiği aşıyor\" diye işaretleniyor ve "
        "\"güçlü kanıt\" görünüyordu. Tek farkı checkpoint seçimiydi."
    )

    df = _tablo(sonuclar)

    st.markdown("### Farklar ve gürültü bandı")
    st.markdown(
        "Gri kuşak, hiçbir bozulma içermeyen koşular arasında gözlenen "
        "yayılımdır. Kuşağın içinde kalan bir nokta, saf rastgelelikten "
        "ayırt edilemez. Yalnızca eşiği hesaplanabilen bozulma senaryoları "
        "çizilir; kontrol koşuları ölçüm aracıdır, ölçüm nesnesi değil."
    )
    metrik = st.radio("Metrik", ["mAP50", "precision", "recall"], horizontal=True)
    st.altair_chart(
        stil.gurultu_bandi_grafigi(_band_verisi(sonuclar, metrik)),
        width="stretch",
    )
    stil.yorum(
        "Turuncu noktalar bandın içinde: o metrikte bozulma kanıtı yok. "
        "Mavi noktalar bandı aşıyor."
    )

    st.markdown("### Tablo")
    # Olcek filtresi, karsilastirilabilirlik kuralini ELLE TUTULUR kilar:
    # bir olcek secildiginde tabloda kalan her satir birbiriyle gercekten
    # kiyaslanabilir. Karisik tabloda bu goze carpmiyordu.
    olcekler = ["hepsi"] + sorted({s for s in df["ölçek"] if s})
    a, b = st.columns([3, 2])
    with a:
        olcek = st.selectbox("Ölçek (model | küme | çözünürlük | checkpoint)",
                             olcekler)
    with b:
        sadece_asan = st.checkbox("Yalnızca gürültü eşiğini aşanlar", value=False)
    gosterilen = df if olcek == "hepsi" else df[df["ölçek"] == olcek]
    if sadece_asan:
        gosterilen = gosterilen[gosterilen["eşiği aşan"] != "-"]
    st.dataframe(gosterilen.drop(columns=["ölçek"]), hide_index=True,
                 width="stretch", height=420)
    if olcek != "hepsi":
        stil.yorum(
            f"Bu ölçekte {len(gosterilen)} koşu var ve hepsi birbiriyle "
            "doğrudan karşılaştırılabilir."
        )
    stil.yorum(
        "'kanıt' sütunu yalnızca kendi ölçeğinde hem referansı hem gürültü "
        "eşiği olan koşularda derecelendirilir (güçlü / zayıf / gürültü "
        "içinde). Diğerleri neden derecelendirilemediğini söyler."
    )

    st.markdown("---")
    st.markdown("## Gürültü tabanı ölçülünce ne değişti")
    buyume, zayiflayan, n = _esik_buyumesi(sonuclar)
    st.markdown(
        f"İlk ölçüm tek bir kontrol koşusuna dayanıyordu ve gürültüyü ciddi "
        f"biçimde **küçük** gösteriyordu. {n} kontrol koşusuna çıkıldığında "
        "eşikler büyüdü:"
    )
    st.dataframe(buyume, hide_index=True, width="stretch")

    st.markdown(f"### Zayıflayan {len(zayiflayan)} iddia")
    st.dataframe(zayiflayan, hide_index=True, width="stretch")
    stil.yorum(
        "Bu iki tablo elle yazılmaz; defterden türetilir. Yeni bir kontrol "
        "koşusu eklendiğinde eşikler ve zayıflayan iddialar kendiliğinden "
        "güncellenir. Genel örüntü: recall'a dayanan iddialar en kırılgan "
        "olanlar."
    )

    st.markdown("### Erken durdurma noktası da seed'e bağlı")
    st.dataframe(_erken_durdurma(sonuclar), hide_index=True, width="stretch")
    stil.yorum(
        "Aynı veri, aynı protokol: eğitim süresi 11 ile 30 epoch arasında "
        "değişiyor. Gürültü yalnızca son metrikte değil sürecin kendisinde de var."
    )

    st.markdown("---")
    st.markdown("## Checkpoint seçimi bir kör nokta")
    st.dataframe(_checkpoint_ciftleri(sonuclar), hide_index=True, width="stretch")
    stil.yorum(
        "Her satır kendi checkpoint ailesinin referansıyla karşılaştırılır. "
        "last.pt ailesinde henüz kontrol koşusu yok, bu yüzden o satırlarda "
        "eşik hesaplanamıyor — fark ölçülebiliyor ama gürültüden ayrılamıyor."
    )
    taban, dususler = _best_last_dususu(sonuclar)
    if taban is not None:
        # "Tabandan sert" olcutu: dusus, tabani mAP50 gurultu esigi kadar
        # asiyorsa. Elle secilmis bir sayi degil, olculmus esik.
        esik = _grup_esikleri("D1")["mAP50"] or 0.0
        sert = [f"{a} ({d:+.4f})" for a, d in dususler if d < taban - esik]
        yakin = [f"{a} ({d:+.4f})" for a, d in dususler if d >= taban - esik]
        stil.kutu(
            "<b>Dikkat — son checkpoint düşüşünün de bir tabanı var.</b> "
            "Sağlıklı referansın kendisi best.pt'den last.pt'ye geçerken "
            f"<b>{taban:+.4f}</b> düşüyor. Bu yüzden her last.pt düşüşü "
            "bozulma işareti değildir. Tabanı mAP50 gürültü eşiği "
            f"({esik:.4f}) kadar aşmayanlar: {', '.join(yakin) or 'yok'}. "
            f"Belirgin ayrışanlar: {', '.join(sert) or 'yok'}."
        )

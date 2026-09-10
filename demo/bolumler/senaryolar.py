"""Deney Senaryolari: arastirma senaryolarinin kesif paneli.

Bilgi mimarisi
--------------
Onceki surum 26 kosuyu esit agirlikta kart olarak gosteriyordu ve bu, veri
modelindeki her satiri arayuze siziyordu: kullanici D4 ile `C2 seed21`'i
ayni onemde goruyordu. Oysa bunlar ayni seviyede kavramlar degil - senaryo
bir HIPOTEZ, kosu o hipotezi gerceklestiren bir KAYIT.

Sayfa artik ikiye ayrilir:

- **Senaryo katalogu** (ust): yalnizca arastirma sorusu tasiyan 14 senaryo.
  Kartlarda kod degil INSAN DILINDEKI AD baskin; her kartta tek cumlelik
  bozulma, gurultuye gore tartilmis ana etki ve kanit rozeti var.
- **Kosu defteri** (alt, acilir): referans, kontrol, seed/checkpoint/
  cozunurluk varyantlari dahil butun 26 satir.

Ayrintida grafik ANA ANLATICI. Her senaryonun kendi mekanizmasini en iyi
anlatan grafik one cikar (senaryo_grafikleri.py); genel metrik tablosu
"ham degerler" acilirina tasindi.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import grafik
import katalog
import senaryo_grafikleri as sg
import stil
import veri_seti as vs
from data_loader import (
    error_galleries,
    evidence_for,
    gorsel_coz,
    load_results,
    referans_galerisi,
)
from teshis.degerlendirme.karsilastirilabilirlik import (
    bozulmasiz_mi,
    esik_yoklugu_aciklamasi,
    gurultu_esigi_gecerli_mi,
    kimlik,
)
from teshis.degerlendirme.senaryo_ozeti import kanit_gucu, ne_gozlendi, ozet

SECIM = "secili_senaryo"


# --- Senaryo katalogu -------------------------------------------------------

def _etki_metni(s: dict) -> str:
    e = s["ana_etki"]
    if not s["olculdu"]:
        # Olculemeyen senaryo ile "olculdu ama etki cikmadi" AYNI SEY DEGIL.
        # E3 iraksadi: hicbir model uretilmedi, dolayisiyla "etki yok"
        # demek yanlis olurdu.
        return "ölçülemedi — eğitim ıraksadı"
    if not e:
        return "gürültüyü aşan etki yok"
    ad = vs.BANT_ADI.get(e["alan"].split()[0], e["alan"])
    if e["alan"].split()[0] in vs.BANT_ADI:
        ad = f"{vs.BANT_ADI[e['alan'].split()[0]]} recall"
    olcut = ("eşlenik ölçüm" if e.get("eslenik")
             else f"gürültünün {e['oran']:.0f} katı")
    return f"{ad} <b>{e['fark']:+.4f}</b> · {olcut}"


def _kart(s: dict, secili: bool) -> None:
    kenar = stil.ADAY if secili else stil.CIZGI
    zemin = stil.YUZEY_2 if secili else stil.YUZEY
    st.markdown(
        f'<div style="border:1px solid {kenar};border-left:3px solid '
        f'{stil.ADAY if secili else stil.CIZGI};border-radius:8px;'
        f'background:{zemin};padding:.75rem .9rem;min-height:158px">'
        f'<div style="display:flex;justify-content:space-between;gap:.4rem">'
        f'<div><span style="color:{stil.METIN_SOLUK};font-size:.78rem">'
        f'{s["kod"]}</span><br>'
        f'<b style="font-size:1.0rem;color:#fff">{s["ad"]}</b></div>'
        f'<div>{stil.guc_rozeti(s["kanit"])}</div></div>'
        f'<div class="yorum" style="margin-top:.4rem">{s["ne_olcuyor"]}</div>'
        f'<div style="margin-top:.45rem;font-size:.82rem;'
        f'color:{stil.METIN_SOLUK}">Ana etki: {_etki_metni(s)}</div></div>',
        unsafe_allow_html=True,
    )


def _katalog(secili: str) -> None:
    st.markdown(
        "Yalnızca **araştırma senaryoları**. Referans, kontrol koşuları ve "
        "seed/checkpoint/çözünürlük varyantları aşağıdaki koşu defterinde."
    )
    hepsi = katalog.senaryolar()

    a, b = st.columns([3, 2])
    with a:
        aileler = list(dict.fromkeys(s["aile"] for s in hepsi))
        aile = st.radio(
            "Aile", ["Tümü"] + [katalog.AILE_ADI[a] for a in aileler],
            horizontal=True, label_visibility="collapsed",
        )
    with b:
        kanit = st.radio(
            "Kanıt", ["Tümü", "Güçlü bulgu", "Zayıf / gürültü içinde"],
            horizontal=True, label_visibility="collapsed",
        )

    gosterilen = hepsi
    if aile != "Tümü":
        kod = next(k for k, v in katalog.AILE_ADI.items() if v == aile)
        gosterilen = [s for s in gosterilen if s["aile"] == kod]
    if kanit == "Güçlü bulgu":
        gosterilen = [s for s in gosterilen if s["kanit"] == "guclu"]
    elif kanit == "Zayıf / gürültü içinde":
        gosterilen = [s for s in gosterilen
                      if s["kanit"] in ("zayif", "gurultu icinde", "olcum yok")]

    # "Gelismis filtreler" diye bir acilir kutu vardi ama icinde yalnizca
    # "bunlar kosu defterinde" yazan bir yonlendirme metni bulunuyordu -
    # hicbir filtre sunmuyordu. Bos vaat vermek yerine kaldirildi; koshu
    # defterinin kendi filtreleri zaten asagida.

    if not gosterilen:
        st.warning("Bu filtrelerle senaryo bulunamadı.")
        return

    for satir in range(0, len(gosterilen), 3):
        for sutun, s in zip(st.columns(3), gosterilen[satir:satir + 3]):
            with sutun:
                _kart(s, s["kod"] == secili)
                # Kart HTML'i tiklanabilir degildir; gercek bir dugme
                # kullanilir. Gorsel olarak kart, davranis olarak dugme olan
                # bir yapi kullaniciyi yaniltirdi.
                etiket = ("● incelenen" if s["kod"] == secili
                          else "İncele →")
                if st.button(etiket, key=f"sec_{s['kod']}",
                             width="stretch",
                             disabled=s["kod"] == secili):
                    st.session_state[SECIM] = s["kod"]
                    st.rerun()
                st.write("")


# --- Senaryo ayrintisi ------------------------------------------------------

def _sonuc_kartlari(s: dict, gozlem: dict) -> None:
    e = s["ana_etki"]
    en_belirgin = _etki_metni(s).replace("<b>", "").replace("</b>", "")
    genel = "—"
    if gozlem:
        genel_adaylar = [(a, d) for a, d in gozlem["metrikler"].items()
                         if d["fark"] is not None]
        if genel_adaylar:
            ad, d = max(genel_adaylar, key=lambda c: abs(c[1]["fark"]))
            genel = f"{ad} {d['fark']:+.4f}"
    asan = gozlem.get("asan_metrikler") if gozlem else []
    durum = (f"{len(asan)} metrik eşiği aştı" if asan
             else "hiçbir metrik eşiği aşmadı")
    a, b, c = st.columns(3)
    with a:
        stil.kpi("En belirgin etki", en_belirgin.split(" · ")[0] or "—",
                 en_belirgin.split(" · ")[1] if " · " in en_belirgin else "")
    with b:
        stil.kpi("Genel metriklerde", genel, "en büyük genel fark")
    with c:
        stil.kpi("Kanıt durumu", stil.seviye_adi(s["kanit"]), durum)


def _kritik_sinirlama(kosu: str, gozlem: dict) -> str | None:
    """Grafigin YANINDA kalmasi gereken tek kritik uyari.

    Ayrintili sinirlamalar "Sonuclar ve Sinirlamalar" sayfasinda; ama
    kritik olan grafikten ayrilmamali - ayrilirsa grafik oldugundan guclu
    gorunur.
    """
    from teshis.degerlendirme.bootstrap import VAL_DIAGNOSTIC_BBOX_N

    if not gozlem:
        return None
    if (gozlem["karsilastirma_turu"] == "yok"
            or not gurultu_esigi_gecerli_mi(kosu)
            or not gozlem["kontrol_kosu_sayisi"]):
        # Metin TEK KAYNAKTAN; burada ayri yazilmisti ve eslenik olcumde
        # "kontrol kosusu yok" diyordu - oysa aranan bir kontrol yok.
        return esik_yoklugu_aciklamasi(kosu)
    kod = kosu.split()[0]
    if kod in ("D3",):
        az = ", ".join(f"{a} n={n}" for a, n in VAL_DIAGNOSTIC_BBOX_N.items()
                       if n < 30)
        return (f"Nadir sınıflar ({az}): etkinin yönü görülüyor, büyüklüğü "
                "kesin tahmin edilemez.")
    n = gozlem["kontrol_kosu_sayisi"]
    if n:
        return (f"Gürültü eşiği {n} kontrol koşusundan hesaplandı; az "
                "gözlemle eşik gerçek yayılımı olduğundan küçük gösterir.")
    return None


def _genel_gorunum(s: dict, kosu: str, o: dict, gozlem: dict) -> None:
    st.markdown("#### Ne değiştirildi?")
    p = (o["ne_degisti"].get("parametreler") or {})
    stil.kutu(
        (" · ".join(f"{k}: <b>{v}</b>" for k, v in p.items())
         if p else "Bu koşu için parametre kaydı yok.")
        + (f'<div class="yorum" style="margin-top:.4rem">Referans: '
           f'<code>{gozlem.get("referans_senaryo") or "yok"}</code> · '
           "değişen tek unsur senaryonun kendisi; model, değerlendirme "
           "kümesi, çözünürlük ve checkpoint sabit.</div>" if gozlem else "")
    )

    if not gozlem:
        st.warning(
            "Bu senaryonun defterde ölçümü yok. "
            + (o.get("ne_olcuyor") or "")
        )
        return

    a, b = st.columns([1, 1])
    with a:
        veri = sg.metrik_imzasi_verisi(kosu)
        if veri is not None:
            st.altair_chart(
                grafik.fark_profili(veri, "metrik", "fark", "gürültü eşiği",
                                    baslik="Genel metrikler: referansa fark"),
                width="stretch",
            )
            stil.yorum(
                "Solda düşüş, sağda artış. Dolu çubuklar gürültü eşiğini "
                "aşıyor, soluk olanlar rastgelelikten ayırt edilemiyor."
            )
    with b:
        tur = sg.yildiz_turu(kosu)
        if tur:
            cizim, notu = sg.ciz(kosu, tur)
            if cizim is None:
                st.info(notu)
            else:
                st.altair_chart(cizim, width="stretch")
                stil.yorum(notu)

    kritik = _kritik_sinirlama(kosu, gozlem)
    asan = gozlem["asan_metrikler"]
    e = s["ana_etki"]
    sonuc = (
        f"<b>{kosu}</b> "
        + (f"{', '.join(asan)} metriklerinde ölçülebilir etki oluşturdu. "
           if asan else "hiçbir genel metrikte gürültü eşiğini aşmadı. ")
        + (f"Temel bulgu genel metriklerde değil, "
           f"<b>{e['alan']}</b> değerindeki {e['fark']:+.4f} değişimdir."
           if e and e.get("kirilim") else "")
    )
    stil.kutu(stil.guc_rozeti(s["kanit"]) + " &nbsp;" + sonuc
              + (f'<div class="yorum" style="margin-top:.5rem">'
                 f"<b>Kritik sınırlama:</b> {kritik}</div>" if kritik else ""))


def _gorsel_kanit(kosu: str) -> None:
    galeriler = error_galleries()
    galeri = galeriler.get(kosu)
    # Gorsel referans SAYISAL referansla AYNI olmali. Sabit
    # `v00_saglikli` kullanmak D1n'i (referansi v00n) ve last_pt
    # kosularini (referansi v00'in last.pt'si) yanlis tabana gore
    # gosteriyordu: ekranda sayilar bir referansa, goruntuler baska
    # bir referansa gore okunuyordu.
    ref_ad, saglikli_galeri = referans_galerisi(kosu)
    saglikli = {e.get("source"): e for e in (saglikli_galeri.get("entries") or [])}

    if galeri:
        eslesen = [k for k in galeri["entries"] if k.get("source") in saglikli]
        if eslesen:
            st.markdown("#### Aynı kare, iki model")
            olcutler = {
                "en fazla kaçırılan nesne": "false_negatives",
                "en fazla fazladan kutu": "false_positives",
                "en düşük IoU": "mean_iou",
            }
            secim = st.radio("Kare seçimi", list(olcutler), horizontal=True,
                             key=f"kare_{kosu}")
            alan = olcutler[secim]
            kayit = sorted([k for k in eslesen if alan in k],
                           key=lambda k: k[alan],
                           reverse=alan != "mean_iou")[0]
            es = saglikli[kayit["source"]]
            a, b = st.columns(2)
            for sutun, baslik, kyt, klasor in (
                (a, f"referans modeli — {ref_ad}", es,
                 saglikli_galeri.get("folder")),
                (b, f"{kosu} modeli", kayit, galeri["folder"]),
            ):
                with sutun:
                    stil.ust_baslik(baslik)
                    yol = gorsel_coz(klasor / kyt["image"]) if klasor else None
                    if yol:
                        st.image(str(yol), width="stretch")
                    else:
                        st.info("Bu karenin görseli bulunamadı.")
                    stil.yorum(
                        f"kaçırılan {kyt.get('false_negatives')} · fazladan "
                        f"{kyt.get('false_positives')} · "
                        f"IoU {kyt.get('mean_iou', 0):.2f}"
                    )
            stil.yorum(
                "Yeşil = gerçek etiket, kırmızı = modelin tahmini; yeşilin "
                "yanında kırmızı yoksa o nesne kaçırılmıştır. Bu karede "
                f"kaçırma <b>{es.get('false_negatives')} → "
                f"{kayit.get('false_negatives')}</b>."
            )
        else:
            st.info(
                f"Bu koşunun en sorunlu kareleri {ref_ad} galerisinde "
                "yok; galeriler her koşunun KENDİ en kötü kareleriyle "
                "üretildiği için listeler her zaman örtüşmez. Eşleşmiş çift "
                "bulunmadığı için yan yana karşılaştırma yapılmıyor."
            )
    else:
        st.info("Bu koşu için hata galerisi üretilmemiş.")

    veri = sg.karisiklik_verisi(kosu)
    if veri is not None:
        st.markdown("#### Karışıklık farkı")
        st.altair_chart(grafik.karisiklik_farki(veri), width="stretch")
        stil.yorum(
            "Sayısal ölçümden çizilir, Ultralytics'in PNG'sinden değil — o "
            "görsel bir kez kendi raporladığı recall ile çelişti (D3b)."
        )


def _teknik(kosu: str, o: dict, gozlem: dict) -> None:
    a, b = st.columns(2)
    with a:
        stil.ust_baslik("deney protokolü")
        stil.kutu("<br>".join(o["ne_sabit_kaldi"]))
    with b:
        stil.ust_baslik("deneyden önce ne bekleniyordu")
        stil.kutu(" ".join(str(o.get("beklenen_etki") or "—").split()))

    if gozlem:
        st.markdown("**Ham değerler**")
        satirlar = []
        for ad, d in gozlem["metrikler"].items():
            ref, fark = d["referans"], d["fark"]
            satirlar.append({
                "metrik": ad,
                "sağlıklı referans": ref,
                "senaryo": d["deger"],
                # "mutlak fark" YANLIS bir basliktir: gosterilen degerler
                # negatif olabiliyor. Mutlak fark negatif olamaz.
                "fark (senaryo − referans)": fark,
                "göreli değişim (%)": (None if not ref or fark is None
                                       else round(100 * fark / ref, 2)),
                "gürültü eşiği": d["gurultu_esigi"],
                "karar": {True: "eşiği aşıyor", False: "gürültü içinde"}.get(
                    d["asiyor"], "eşik yok"),
            })
        st.dataframe(pd.DataFrame(satirlar), hide_index=True, width="stretch")

    egri, isaretler = sg.egitim_egrisi_verisi(kosu)
    if egri is not None:
        st.markdown("**Eğitim eğrisi**")
        st.altair_chart(
            grafik.egri_isaretli(egri, "epoch", "kayıp", "seri", isaretler,
                                 alan_adi="sınıflandırma kaybı"),
            width="stretch",
        )

    st.markdown("**Sınırlamalar**")
    for x in o["sinirlamalar"]:
        st.markdown(f"- {x}")
    stil.yorum(
        "Projenin tamamına ait sınırlamalar \"Sonuçlar ve Sınırlamalar\" "
        "bölümünde."
    )

    with st.expander("Ham metrik dosyası (JSON)"):
        st.json(evidence_for(kosu))


def _ayrinti(kod: str) -> None:
    s = katalog.senaryo(kod)
    if s is None:
        return
    kosu = s["ana_kosu"]

    st.markdown("---")
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:.6rem">'
        f'<span style="color:{stil.METIN_SOLUK};font-size:1rem">{s["kod"]}</span>'
        f'<span style="font-size:1.45rem;font-weight:600;color:#fff">'
        f'{s["ad"]}</span>{stil.guc_rozeti(s["kanit"])}</div>',
        unsafe_allow_html=True,
    )
    if not kosu:
        stil.kutu(
            f"{s['ne_olcuyor']}<br><br><b>Bu senaryo ölçülemedi.</b> Sonuç "
            "bir eksiklik değil <b>negatif bulgudur</b>: hipotez test "
            "edilebilir bir model üretmedi ve bu, deney tasarımının kendisi "
            "hakkında bilgi taşır."
        )
        return

    o = ozet(kosu)
    gozlem = o["ne_gozlendi"]
    stil.yorum(s["ne_olcuyor"])
    st.write("")
    _sonuc_kartlari(s, gozlem)
    st.write("")

    genel, gorsel, teknik = st.tabs(
        ["Genel görünüm", "Görsel kanıt", "Teknik ayrıntılar"]
    )
    with genel:
        _genel_gorunum(s, kosu, o, gozlem)
    with gorsel:
        _gorsel_kanit(kosu)
    with teknik:
        _teknik(kosu, o, gozlem)

    _iliskili_kanit(s, kosu)


def _iliskili_kanit(s: dict, kosu: str) -> None:
    """Senaryonun varyantlari ve onlarin TASIDIGI BULGU.

    Varyantlar ayri arastirma senaryosu degildir ama bazilari projenin
    temel bulgusunu tasir: E1'in asil kaniti best.pt/last.pt ayrismasidir,
    D1'inki iki baslangic modeli arasindaki farktir. Bunlari yalnizca kosu
    defterinde birakmak, bulgunun ait oldugu senaryodan kopmasi demekti.
    """
    from teshis.degerlendirme.senaryo_ozeti import ne_gozlendi as _ng

    if not s["varyantlar"]:
        return
    st.markdown("#### İlişkili kanıt")
    satirlar = []
    for v in s["varyantlar"]:
        g = _ng(v) or {}
        m = (g.get("metrikler") or {}).get("mAP50") or {}
        ana_m = ((_ng(kosu) or {}).get("metrikler") or {}).get("mAP50") or {}
        fark = (None if m.get("deger") is None or ana_m.get("deger") is None
                else round(m["deger"] - ana_m["deger"], 4))
        satirlar.append({
            "varyant": v,
            "ne değişiyor": _varyant_farki(kosu, v),
            "mAP50": m.get("deger"),
            f"Δ {kosu}": fark,
            "referansı": g.get("referans_senaryo") or "—",
            "kanıt": stil.seviye_adi(kanit_gucu(v)["seviye"]),
        })
    st.dataframe(pd.DataFrame(satirlar), hide_index=True, width="stretch")
    stil.yorum(
        "Bunlar ayrı araştırma senaryosu değil, aynı hipotezin farklı "
        "checkpoint / başlangıç modeli / çözünürlük / seed kayıtlarıdır. "
        "Karşılaştırma sayfasından her biri tek tek açılabilir."
    )


def _varyant_farki(ana: str, varyant: str) -> str:
    """Varyantin ana kosudan HANGI alanda ayrildigi - kimlikten turetilir."""
    a, b = kimlik(ana), kimlik(varyant)
    if a is None or b is None:
        return "—"
    farklar = [
        (ad, getattr(a, alan), getattr(b, alan))
        for alan, ad in (("checkpoint", "checkpoint"),
                         ("model", "başlangıç modeli"),
                         ("imgsz_eval", "çıkarım çözünürlüğü"),
                         ("degerlendirme_seti", "değerlendirme kümesi"))
        if getattr(a, alan) != getattr(b, alan)
    ]
    if farklar:
        return " · ".join(f"{ad}: {x} → {y}" for ad, x, y in farklar)
    return "seed"


# --- Kosu defteri -----------------------------------------------------------

def _kosu_defteri(sonuclar: pd.DataFrame) -> None:
    st.markdown(
        "Defterdeki **bütün** koşular: senaryoların ana koşuları, sağlıklı "
        "referanslar, kontrol koşuları ve seed/checkpoint/çözünürlük "
        "varyantları."
    )
    satirlar = []
    for _, r in sonuclar.iterrows():
        ad = str(r["scenario"])
        if kimlik(ad) is None:
            continue
        g = ne_gozlendi(ad)
        m = (g or {}).get("metrikler", {})
        if bozulmasiz_mi(ad):
            rol = "referans" if ad.startswith("v00") else "kontrol"
        elif katalog.kosu_defteri_disinda_mi(ad):
            rol = "senaryo ana koşusu"
        else:
            rol = "varyant"
        satirlar.append({
            "koşu": ad,
            "rol": rol,
            "model": r.get("model"),
            "checkpoint": ("last.pt" if str(r.get("weights_path", "")).endswith(
                "last.pt") else "best.pt"),
            "imgsz": r.get("imgsz_eval"),
            "seed": r.get("seed"),
            "referansı": (g or {}).get("referans_senaryo") or "—",
            "mAP50": m.get("mAP50", {}).get("deger"),
            "Δ mAP50": m.get("mAP50", {}).get("fark"),
            "kanıt": stil.seviye_adi(kanit_gucu(ad)["seviye"]),
        })
    df = pd.DataFrame(satirlar)
    df["Δ mAP50"] = pd.to_numeric(df["Δ mAP50"], errors="coerce")

    a, b, c, d = st.columns(4)
    with a:
        rol = st.multiselect("Rol", sorted(set(df["rol"])), key="defter_rol")
    with b:
        cp = st.multiselect("Checkpoint", sorted(set(df["checkpoint"])),
                            key="defter_cp")
    with c:
        model = st.multiselect("Başlangıç modeli", sorted(set(df["model"])),
                               key="defter_model")
    with d:
        # Seed filtresi metinde vaat ediliyordu ama defterde YOKTU.
        seed = st.multiselect("Seed", sorted(set(df["seed"])), key="defter_seed")
    g = df
    if rol:
        g = g[g["rol"].isin(rol)]
    if cp:
        g = g[g["checkpoint"].isin(cp)]
    if model:
        g = g[g["model"].isin(model)]
    if seed:
        g = g[g["seed"].isin(seed)]
    st.dataframe(g, hide_index=True, width="stretch", height=440)
    st.download_button("Koşu defterini CSV olarak indir",
                       g.to_csv(index=False).encode("utf-8-sig"),
                       file_name="kosu_defteri.csv", mime="text/csv")
    stil.yorum(
        "Her koşu kendi ölçeğindeki referansla karşılaştırılır. Kontrol "
        "koşuları ve referanslar derecelendirilmez: onlar ölçüm aracıdır, "
        "ölçüm nesnesi değil."
    )


def _senaryo_kosu_haritasi() -> None:
    """SENARYO ile KOSU arasindaki iliskiyi acikca yazar.

    Genel Bakis "14 senaryo" ve "26 kosu" diyor; ikisi arasindaki bag
    hicbir yerde gorunmuyordu ve izleyici sayilarin birbirini tutmadigini
    dusunuyordu. Bir senaryo bir HIPOTEZ, bir kosu o hipotezin bir
    KAYDIDIR; bazi hipotezlerin birden fazla kaydi var.
    """
    h = katalog.senaryo_kosu_haritasi()
    with st.expander(
        f"{h['senaryo_sayisi']} senaryo, {h['toplam']} koşu — hangisi "
        "hangisine bağlı?", expanded=False,
    ):
        st.markdown(
            "Bir **senaryo** bir hipotezdir; bir **koşu** o hipotezin bir "
            "kaydıdır. Bazı hipotezlerin birden fazla kaydı var — aynı "
            "eğitimin son epoch'u, farklı bir başlangıç modeli ya da başka "
            "bir rastgelelik tohumu."
        )
        st.dataframe(pd.DataFrame([
            {"senaryo": f"{r['kod']} — {r['ad']}",
             "koşu sayısı": r["sayi"],
             "koşular": ", ".join(r["kosular"]) or "ölçülebilir koşu üretmedi"}
            for r in h["senaryolar"]
        ]), hide_index=True, width="stretch")

        stil.ust_baslik("hiçbir senaryoya bağlı olmayan koşular")
        st.dataframe(pd.DataFrame([
            {"koşu": k, "rolü": katalog.kosu_adi(k).split("·", 1)[-1].strip()}
            for k in h["altyapi"]
        ]), hide_index=True, width="stretch")
        stil.yorum(
            f"<b>{h['senaryo_kosusu']} senaryo koşusu + "
            f"{h['altyapi_kosusu']} altyapı koşusu = {h['toplam']} koşu.</b> "
            "Altyapı koşuları bir hipotez değildir: sağlıklı referanslar "
            "ölçümün tabanını, kontrol koşuları gürültü eşiğini verir. "
            "İkisi de <b>ölçüm aracıdır, ölçüm nesnesi değil</b> — bu yüzden "
            "hiçbir yerde bulgu olarak derecelendirilmezler."
        )


def goster() -> None:
    st.title("Deney Senaryoları")
    _senaryo_kosu_haritasi()
    hepsi = katalog.senaryolar()
    if SECIM not in st.session_state:
        st.session_state[SECIM] = "D4" if any(
            s["kod"] == "D4" for s in hepsi) else hepsi[0]["kod"]

    _katalog(st.session_state[SECIM])
    _ayrinti(st.session_state[SECIM])

    st.markdown("---")
    with st.expander("Koşu defteri — bütün deney kayıtları"):
        _kosu_defteri(load_results())

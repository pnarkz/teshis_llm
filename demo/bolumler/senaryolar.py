"""Deney Senaryolari: her kosunun tam kunyesi ve kaniti.

Sayfa iki katmanlidir:

1. **Filtreli genel bakis** - butun kosular tek ekranda, aile/kanit/checkpoint
   filtreleriyle. Liste `results.csv`'den turetilir; hicbir senaryo koda
   gomulu degildir.
2. **Secilen kosunun tam analizi** - 15 zorunlu soru, karsilastirma sablonu,
   grafikler ve ajan iliskisi.

Icerigin bes bileseninden dordu turetilir (teshis/degerlendirme/senaryo_ozeti);
elle yazilan tek alan senaryonun ne olctugudur (senaryolar/anlatim.yaml).

"Referansla ortak olan" basligi bilerek boyle: sayfa bir donem "ne sabit
kaldi" deyip degerlendirme setini SABIT val_diagnostic yaziyordu, oysa D6a
kasitli olarak sizintili kumede olculur.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import grafik
import stil
import veri_seti as vs
from data_loader import (
    ajan_kaydi,
    ajan_kosu_haritasi,
    evidence_for,
    examples_for,
    images_for,
    load_results,
)
from teshis.degerlendirme.karsilastirilabilirlik import bozulmasiz_mi, kimlik
from teshis.degerlendirme.senaryo_ozeti import ozet

AILE_ADI = {
    "D": "D — veri arızası",
    "E": "E — eğitim/çıkarım ayarı",
    "C": "C — kontrol (bozulma yok)",
    "v": "referans",
}


def _aile(senaryo: str) -> str:
    kod = senaryo.split()[0]
    if kod.startswith("v00") or kod == "Baseline":
        return "v"
    return kod[0] if kod[0] in AILE_ADI else "D"


def _checkpoint(satir) -> str:
    return "last.pt" if str(satir.get("weights_path", "")).endswith("last.pt") else "best.pt"


def _kosu_ozetleri(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Filtre ve kart izgarasinin beslendigi tablo - defterden turetilir."""
    satirlar = []
    for _, r in sonuclar.iterrows():
        ad = str(r["scenario"])
        if kimlik(ad) is None:          # defterde olmayan sentetik satirlar
            continue
        o = ozet(ad)
        g = o["ne_gozlendi"]
        m = g["metrikler"] if g else {}
        # "En onemli fark": mutlak buyuklugu en yuksek olan genel metrik.
        onemli, onemli_fark = "—", None
        for ad_m, d in m.items():
            if d["fark"] is None:
                continue
            if onemli_fark is None or abs(d["fark"]) > abs(onemli_fark):
                onemli, onemli_fark = ad_m, d["fark"]
        satirlar.append({
            "senaryo": ad,
            "aile": _aile(ad),
            "tür": (o["ne_degisti"].get("tur") or
                    ("kontrol" if bozulmasiz_mi(ad) else "—")),
            "kanıt": o["kanit_gucu"]["seviye"],
            "eşiği aşan": ", ".join(g["asan_metrikler"]) if g else "",
            "checkpoint": _checkpoint(r),
            "model": r.get("model", "?"),
            "en büyük fark": onemli,
            "fark": onemli_fark,
            "referansı": (g or {}).get("referans_senaryo") or "—",
        })
    return pd.DataFrame(satirlar)


def _filtreler(tablo: pd.DataFrame) -> pd.DataFrame:
    a, b, c = st.columns(3)
    with a:
        aileler = st.multiselect(
            "Senaryo ailesi",
            [AILE_ADI[k] for k in AILE_ADI if k in set(tablo["aile"])],
            default=[],
        )
    with b:
        seviyeler = st.multiselect(
            "Kanıt gücü",
            sorted({stil.seviye_adi(s) for s in tablo["kanıt"]}),
            default=[],
        )
    with c:
        checkpointler = st.multiselect(
            "Checkpoint", sorted(set(tablo["checkpoint"])), default=[],
        )
    d, e = st.columns(2)
    with d:
        modeller = st.multiselect(
            "Başlangıç modeli", sorted(set(tablo["model"])), default=[],
        )
    with e:
        yalnizca_asan = st.checkbox(
            "Yalnızca gürültü eşiğini aşanlar", value=False,
            help="En az bir genel metrikte eşiği aşan koşular.",
        )

    df = tablo
    if aileler:
        kodlar = {k for k, v in AILE_ADI.items() if v in aileler}
        df = df[df["aile"].isin(kodlar)]
    if seviyeler:
        df = df[df["kanıt"].map(stil.seviye_adi).isin(seviyeler)]
    if checkpointler:
        df = df[df["checkpoint"].isin(checkpointler)]
    if modeller:
        df = df[df["model"].isin(modeller)]
    if yalnizca_asan:
        df = df[df["eşiği aşan"] != ""]
    return df


def _kart(satir: pd.Series) -> None:
    fark = satir["fark"]
    fark_metni = "—" if fark is None or pd.isna(fark) else f"{fark:+.4f}"
    stil.kutu(
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:flex-start;gap:.5rem">'
        f'<div><b style="font-size:1.02rem">{satir["senaryo"]}</b><br>'
        f'<span class="yorum">{AILE_ADI.get(satir["aile"], "")} · '
        f'{satir["checkpoint"]}</span></div>'
        f'<div>{stil.guc_rozeti(satir["kanıt"])}</div></div>'
        f'<div style="margin-top:.5rem;font-size:.82rem;color:{stil.METIN_SOLUK}">'
        f'en büyük fark: <span style="color:{stil.METIN}">'
        f'{satir["en büyük fark"]} {fark_metni}</span><br>'
        f'eşiği aşan: <span style="color:{stil.METIN}">'
        f'{satir["eşiği aşan"] or "—"}</span></div>'
    )


# --- Secilen senaryonun tam analizi -----------------------------------------

def _uygulama_orani(o: dict) -> str:
    """"Ne kadar uygulandi?" - konfig parametrelerinden okunur.

    "Veri azaltildi" gibi belirsiz ifade yerine gercek oran yazilir; oran
    konfigde yoksa uydurulmaz, parametreler oldugu gibi listelenir.
    """
    p = o["ne_degisti"].get("parametreler") or {}
    if not p:
        return "Bu koşu için parametre kaydı yok."
    return "<br>".join(f"<b>{k}:</b> <code>{v}</code>" for k, v in p.items())


def _karsilastirma_sablonu(gozlem: dict) -> pd.DataFrame:
    """Metrik | referans | senaryo | mutlak fark | goreli | esik | karar."""
    satirlar = []
    for ad, d in gozlem["metrikler"].items():
        ref, deger, fark = d["referans"], d["deger"], d["fark"]
        goreli = (None if not ref or fark is None else round(100 * fark / ref, 2))
        karar = {True: "eşiği aşıyor", False: "gürültü içinde"}.get(
            d["asiyor"], "eşik yok"
        )
        satirlar.append({
            "metrik": ad,
            "sağlıklı referans": ref,
            "senaryo": deger,
            "mutlak fark": fark,
            "göreli değişim (%)": goreli,
            "gürültü eşiği": d["gurultu_esigi"],
            "karar": karar,
        })
    return pd.DataFrame(satirlar)


def _kirilim_karsilastirmasi(senaryo: str, alan: str, ad_haritasi=None):
    """Senaryo ve referansin ayni kirilim alanindaki recall'lari yan yana."""
    gozlem = ozet(senaryo)["ne_gozlendi"]
    ref_ad = (gozlem or {}).get("referans_senaryo")
    if not ref_ad:
        return None
    k, ref = vs.kirilim(senaryo), vs.kirilim(ref_ad)
    if not k or not ref:
        return None
    satirlar = []
    for grup, d in (k.get(alan) or {}).items():
        r = (ref.get(alan) or {}).get(grup) or {}
        etiket = (ad_haritasi or {}).get(grup, grup)
        satirlar.append({"grup": etiket, "seri": "sağlıklı referans",
                         "recall": r.get("recall"), "kutu": r.get("gercek_kutu")})
        satirlar.append({"grup": etiket, "seri": senaryo,
                         "recall": d.get("recall"), "kutu": d.get("gercek_kutu")})
    return pd.DataFrame([s for s in satirlar if s["recall"] is not None])


# Fark GORULMEDIGINDE olasi nedenler. Liste mevcut bulgulara dayanir; her
# madde projede GERCEKTEN gozlenmis bir mekanizmadir, spekulasyon degil.
GORULMEME_NEDENLERI = [
    ("Modelin önceden öğrenilmiş özellikleri",
     "Fine-tune edilen model bozulmadan önce de o sınıfı tanıyordu; eğitim "
     "verisindeki kayıp, önceki temsili silmeye yetmemiş olabilir. D1'in "
     "main_model kurgusunda etkisiz, yolo26n kurgusunda güçlü çıkması bunun "
     "doğrudan kanıtıdır."),
    ("Veri fazlalığı — bozulma soğuruluyor",
     "Bol örnekli bir sınıfta aynı bozulma soğurulabilir. D3 (nadir sınıflar) "
     "güçlü etki üretirken D3b (bol örnekli taşıt/insan) aynı karışıklığı "
     "büyük ölçüde soğurdu."),
    ("Bozulma yalnızca belirli bir alt grupta görünüyor",
     "Toplam metrik değişmezken tek bir kırılım çökebilir. D4'te <16 px "
     "bandının recall'ı yarıdan fazla düşerken toplam mAP50 farkı küçük "
     "kaldı."),
    ("Ölçüm seed gürültüsünün içinde kalıyor",
     "Fark gerçek olabilir ama hiçbir bozulma içermeyen koşular arasında da "
     "görülen büyüklükte olabilir. Bu durumda fark kanıt sayılmaz — D6b tam "
     "olarak bu nedenle bulgu olmaktan çıktı."),
    ("Checkpoint seçimi problemi gizliyor",
     "En iyi checkpoint ile raporlanan bir model sağlıklı görünebilirken son "
     "checkpoint arızayı gösterir. E1 ve D5 bunun iki ayrı örneğidir."),
    ("Nadir sınıfta örnek sayısı yetersiz",
     "UAP (15 bbox) ve UAI (17 bbox) üzerinde ölçülen oranlar tek tek "
     "nesnelere aşırı duyarlıdır; bu sınıflarda 'etki yok' sonucu da 'etki "
     "var' sonucu kadar belirsizdir."),
    ("Deney hipotezi test etmeye yetmiyor",
     "Bozulma oranı, epoch sayısı veya örneklem, beklenen etkiyi ölçülebilir "
     "kılacak kadar güçlü olmayabilir. E3'te öğrenme oranı 100 kat "
     "yükseltildiğinde kararsızlık değil tam ıraksama üretildi ve hipotez "
     "hiç test edilemedi; E3b bu yüzden 10 katla tekrarlandı."),
]


def _ajan_bolumu(senaryo: str) -> None:
    kayit = ajan_kaydi()
    harita = ajan_kosu_haritasi()
    kosu_id = next((k for k, ad in harita.items() if ad == senaryo), None)
    if kosu_id is None:
        st.info(
            "**Bu koşu ajan değerlendirmesine dahil edilmedi.** Ajana yalnızca "
            "aynı ölçekte (aynı model, küme, çözünürlük, checkpoint) ölçülmüş "
            "ve protokol sapması taşımayan koşular verilir; filtre yapısaldır, "
            "ad listesine dayanmaz."
        )
        return

    cevap = (kayit.get("cevaplar") or {}).get(kosu_id)
    puan = (kayit.get("puanlar") or {}).get(kosu_id)
    cagrilar = ((kayit.get("arac_kaydi") or {}).get(kosu_id) or {}).get(
        "arac_cagrilari", []
    )
    if not cevap:
        st.info(f"Bu koşu ajana `{kosu_id}` olarak sunuluyor ama tamamlanmış "
                "bir kayıtlı cevabı yok.")
        return

    a, b, c = st.columns(3)
    with a:
        stil.kpi("Anonim kimlik", f"<code>{kosu_id}</code>", "ajanın gördüğü ad")
    with b:
        stil.kpi("Teşhis puanı",
                 f"{puan.get('diagnosis_score')}" if puan else "—",
                 "cevap anahtarına göre")
    with c:
        stil.kpi("Araç çağrısı", len(cagrilar), "ajanın kendi seçimi")

    st.write("")
    d, e = st.columns(2)
    with d:
        stil.ust_baslik("ajanın teşhisi")
        stil.kutu(f"<b>{cevap.get('diagnosis', '—')}</b>")
    with e:
        stil.ust_baslik("beklenen teşhis (cevap anahtarı)")
        stil.kutu(f"<b>{puan.get('expected', '—') if puan else '—'}</b>")

    if puan:
        st.dataframe(
            pd.DataFrame([{
                "teşhis puanı": puan.get("diagnosis_score"),
                "tespit-farkındalıklı": puan.get("diagnosis_score_tespit"),
                "kanıt puanı": puan.get("evidence_score"),
                "sınırlama puanı": puan.get("limitation_score"),
                "toplam": puan.get("total"),
                "hata türü": puan.get("hata_turu") or "—",
            }]),
            hide_index=True, width="stretch",
        )
        if puan.get("tespit_notu"):
            stil.yorum(
                "Bu koşuda bozulma kanıtta anlamlı iz bırakmıyor; "
                f"gerekçe: {puan['tespit_notu']}"
            )


def goster() -> None:
    sonuclar = load_results()
    st.title("Deney Senaryoları")
    st.markdown(
        "Bütün D, E ve kontrol koşuları. Liste `results.csv`'den türetilir — "
        "hiçbir senaryo koda gömülü değildir."
    )

    tablo = _kosu_ozetleri(sonuclar)
    df = _filtreler(tablo)

    st.markdown(f"### {len(df)} koşu")
    if df.empty:
        st.warning("Bu filtrelerle koşu bulunamadı.")
        return

    for satir in range(0, len(df), 3):
        for sutun, (_, kayit) in zip(st.columns(3), df.iloc[satir:satir + 3].iterrows()):
            with sutun:
                _kart(kayit)
                st.write("")

    st.markdown("---")
    adlar = df["senaryo"].tolist()
    varsayilan = adlar.index("D4") if "D4" in adlar else 0
    senaryo = st.selectbox("Ayrıntılı analiz", adlar, index=varsayilan)
    _ayrinti(senaryo, sonuclar)


def _ayrinti(senaryo: str, sonuclar: pd.DataFrame) -> None:
    o = ozet(senaryo)
    gozlem = o["ne_gozlendi"]

    st.markdown(f"## {senaryo}")
    if o["ne_olcuyor"]:
        stil.kutu(f"<b>Ne test ediyor?</b> {o['ne_olcuyor'].strip()}")

    kurgu, sonuc, kirilim_sek, gorsel, ajan_sek = st.tabs(
        ["Deney kurgusu", "Sonuç ve karar", "Kırılımlar", "Görseller", "Ajan"]
    )

    with kurgu:
        a, b = st.columns(2)
        with a:
            stil.ust_baslik("ne değiştirildi ve hangi oranda")
            d = o["ne_degisti"]
            stil.kutu(
                f"<b>Tür:</b> {d.get('tur') or '—'}<br>{_uygulama_orani(o)}"
                + (f"<br><b>Veri sürümü:</b> <code>{d['veri_surumu']}</code>"
                   if d.get("veri_surumu") else "")
            )
        with b:
            stil.ust_baslik("referansla ortak olan")
            stil.kutu("<br>".join(o["ne_sabit_kaldi"]))

        if o["beklenen_etki"]:
            stil.ust_baslik("deneyden önce ne bekleniyordu")
            stil.kutu(" ".join(str(o["beklenen_etki"]).split()))

        if gozlem:
            stil.ust_baslik("karşılaştırma ölçeği")
            k = gozlem.get("kimlik") or {}
            stil.kutu(
                f"<b>Aday:</b> {senaryo} &nbsp;→&nbsp; "
                f"<b>Referans:</b> {gozlem.get('referans_senaryo') or '— yok —'}<br>"
                f"<code>{k.get('model', '?')}</code> · "
                f"<code>{k.get('degerlendirme_seti', '?')}</code> · "
                f"<code>{k.get('imgsz_eval', '?')} px</code> · "
                f"<code>{k.get('checkpoint', '?')}.pt</code><br>"
                f'<span class="yorum">{gozlem.get("karsilastirma_aciklamasi", "")}</span>'
            )

    with sonuc:
        if not gozlem:
            st.warning("Bu koşunun defterde ölçümü yok.")
        else:
            guc = o["kanit_gucu"]
            st.markdown(stil.guc_rozeti(guc["seviye"]) + " " + guc["aciklama"],
                        unsafe_allow_html=True)
            st.write("")
            sablon = _karsilastirma_sablonu(gozlem)
            st.dataframe(sablon, hide_index=True, width="stretch")

            a, b = st.columns(2)
            with a:
                uzun = sablon.melt(
                    id_vars="metrik",
                    value_vars=["sağlıklı referans", "senaryo"],
                    var_name="seri", value_name="değer",
                ).dropna()
                if not uzun.empty:
                    st.altair_chart(
                        grafik.gruplu_bar(uzun, "metrik", "değer", "seri",
                                          baslik="Referans ve senaryo",
                                          alan_adi="değer"),
                        width="stretch",
                    )
            with b:
                fark_df = sablon.dropna(subset=["mutlak fark"])
                if not fark_df.empty:
                    st.altair_chart(
                        grafik.fark_bar(fark_df, "metrik", "mutlak fark",
                                        esik_alani="gürültü eşiği",
                                        baslik="Fark ve gürültü eşiği"),
                        width="stretch",
                    )
                    stil.yorum(
                        "Vurgulu çubuklar gürültü eşiğini aşıyor; soluk olanlar "
                        "hiçbir bozulma içermeyen koşular arasında da görülen "
                        "büyüklükte."
                    )

            if not gozlem["asan_metrikler"]:
                st.markdown("### Fark neden görülmemiş olabilir?")
                st.markdown(
                    "Bu bir başarısızlık değil, bir bulgudur — ama nedeni "
                    "belirsizdir. Aşağıdaki mekanizmaların hepsi bu projede "
                    "**gerçekten gözlendi**; hangisinin geçerli olduğunu "
                    "ayırt etmek için ek ölçüm gerekir."
                )
                for baslik, metin in GORULMEME_NEDENLERI:
                    with st.expander(baslik):
                        st.markdown(metin)

            st.markdown("### Ne söylenebilir, ne söylenemez")
            a, b = st.columns(2)
            with a:
                stil.ust_baslik("söylenebilir")
                asan = gozlem["asan_metrikler"]
                stil.kutu(
                    (f"{senaryo} koşusunda <b>{', '.join(asan)}</b> metrikleri "
                     f"kendi ölçeğinin gürültü eşiğini aşıyor; bu fark saf "
                     "rastgelelikle açıklanamaz."
                     if asan else
                     "Bu koşuda hiçbir genel metrik gürültü eşiğini aşmıyor; "
                     "genel metriklere dayanan bir etki iddiası kurulamaz.")
                )
            with b:
                stil.ust_baslik("söylenemez")
                stil.kutu(
                    "Bu tek koşudur; etkinin <b>büyüklüğü</b> için güven "
                    "aralığı verilemez. Sonuç bu model ailesine ve bu veri "
                    "sürümüne özgüdür; başka bir başlangıç modeliyle aynı "
                    "çıkacağı gösterilmemiştir."
                    + ("<br>Ayrıca eşik yalnızca "
                       f"{gozlem['kontrol_kosu_sayisi']} kontrol koşusundan "
                       "hesaplandı; gerçek yayılım daha geniş olabilir."
                       if gozlem["kontrol_kosu_sayisi"] else "")
                )

        st.markdown("### Sınırlamalar")
        for s in o["sinirlamalar"]:
            st.markdown(f"- {s}")

    with kirilim_sek:
        secim = st.radio(
            "Kırılım", ["nesne boyutu", "veri kaynağı", "sınıf"],
            horizontal=True, key=f"kirilim_{senaryo}",
        )
        alan, harita = {
            "nesne boyutu": ("boyut_bandi_recall", vs.BANT_ADI),
            "veri kaynağı": ("kaynak_recall", None),
            "sınıf": ("sinif_recall", vs.KOD_ADI),
        }[secim]
        veri = _kirilim_karsilastirmasi(senaryo, alan, harita)
        if veri is None or veri.empty:
            st.info("Bu koşu için kırılım ölçümü bulunamadı.")
        else:
            st.altair_chart(
                grafik.gruplu_bar(veri, "grup", "recall", "seri", yatay=True,
                                  alan_adi="recall"),
                width="stretch",
            )
            genis = veri.pivot(index="grup", columns="seri",
                               values="recall").reset_index()
            st.dataframe(genis, hide_index=True, width="stretch")
            stil.yorum(
                "Hangi alt grubun daha fazla etkilendiği burada görünür. "
                "Toplam metrik değişmezken tek bir grubun çökmesi mümkündür; "
                "bu, projenin en sık tekrarlayan bulgusudur."
            )

    with gorsel:
        egri = vs.egitim_egrisi(senaryo)
        if egri is not None and not egri.empty:
            sutunlar = [s for s in ("train/cls_loss", "val/cls_loss")
                        if s in egri.columns]
            if sutunlar:
                st.markdown("#### Eğitim eğrisi")
                uzun = egri.melt(id_vars="epoch", value_vars=sutunlar,
                                 var_name="seri", value_name="kayıp")
                st.altair_chart(
                    grafik.cizgi(uzun, "epoch", "kayıp", seri="seri",
                                 alan_adi="sınıflandırma kaybı"),
                    width="stretch",
                )
                stil.yorum(
                    "Train ve val kaybı arasındaki farkın açılması aşırı uyum "
                    "imzasıdır; metrikler sessizken eğri konuşabilir."
                )
        gorseller_ = images_for(senaryo)
        if gorseller_:
            st.markdown("#### Confusion matrix ve eğriler")
            secili = [g for g in gorseller_ if "confusion" in g.name.lower()][:1]
            for g in (secili or gorseller_[:1]):
                st.image(str(g), width="stretch")
            with st.expander("Diğer grafikler"):
                for g in gorseller_:
                    st.image(str(g), caption=g.name, width="stretch")
        ornekler = examples_for(senaryo)
        if ornekler:
            st.markdown("#### Örnek tahminler")
            st.image([str(p) for p in ornekler[:3]], width="stretch")
        if not gorseller_ and not ornekler and (egri is None or egri.empty):
            st.info("Bu koşu için görsel çıktı bulunamadı.")
        with st.expander("Ham metrik dosyası"):
            st.json(evidence_for(senaryo))

    with ajan_sek:
        _ajan_bolumu(senaryo)

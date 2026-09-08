"""Sonuclar ve Sinirlamalar: projenin toplu hukmu.

Bu sayfa iki listeden olusur ve ikincisi birincisi kadar onemlidir. Bir
savunmada en guclu kart, neyi soyleyemedigini bilmektir.

Hipotez tablosu ELLE YAZILMAZ: her senaryonun beklentisi konfig dosyasindan
(`beklenen_kanit`), gozlenen sonucu olcumden, kanit gucu de gurultu
bandindan gelir. "Desteklendi mi?" hukmu bu ucunun birlesimidir.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import stil
from data_loader import ajan_kaydi, load_results
from teshis.degerlendirme.karsilastirilabilirlik import (
    bozulmasiz_mi,
    kimlik,
    kontrol_kosulari,
)
from teshis.degerlendirme.senaryo_ozeti import kanit_gucu, ne_gozlendi, ozet

DERECELENDIRILEN = stil.DERECELENDIRILEN


def _hipotez_tablosu(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Her senaryo icin: ne bekleniyordu, ne cikti, hangi hukum.

    Hukum yalnizca kanit gucunden turetilir - "desteklendi" demek icin
    gurultu esigini asmis olmak gerekir. Beklentinin YONU ayrica kontrol
    edilmez: bu, olcumden degil metinden okunacak bir sey olurdu.
    """
    HUKUM = {
        "guclu": ("Desteklendi", "guclu"),
        "zayif": ("Kısmen desteklendi", "uyari"),
        "gurultu icinde": ("Desteklenmedi (gürültü içinde)", "kritik"),
    }
    satirlar = []
    for _, r in sonuclar.iterrows():
        ad = str(r["scenario"])
        if bozulmasiz_mi(ad) or kimlik(ad) is None:
            continue
        seviye = kanit_gucu(ad)["seviye"]
        if seviye not in HUKUM:
            continue
        gozlem = ne_gozlendi(ad)
        o = ozet(ad)
        satirlar.append({
            "senaryo": ad,
            # YAML katlanmis skaler (">") sonunda satir sonu birakir ve
            # tabloda kacis dizisi olarak gorunuyordu; bosluklar tek tek
            # normallestirilir.
            "beklenen etki": " ".join(str(o.get("beklenen_etki") or "—").split()),
            "eşiği aşan metrikler": ", ".join(gozlem["asan_metrikler"]) or "—",
            "hüküm": HUKUM[seviye][0],
            "_tur": HUKUM[seviye][1],
        })
    return pd.DataFrame(satirlar)


def _derecelendirilemeyenler(sonuclar: pd.DataFrame) -> pd.DataFrame:
    satirlar = []
    for _, r in sonuclar.iterrows():
        ad = str(r["scenario"])
        if kimlik(ad) is None:
            continue
        guc = kanit_gucu(ad)
        if guc["seviye"] in DERECELENDIRILEN:
            continue
        satirlar.append({
            "koşu": ad,
            "durum": stil.seviye_adi(guc["seviye"]),
            "neden": stil.seviye_aciklamasi(guc["seviye"]),
        })
    return pd.DataFrame(satirlar)


def _sinirlamalar(sonuclar: pd.DataFrame) -> list[str]:
    """Sayilari defterden turetilen sinirlama listesi."""
    adlar = [a for a in (str(r["scenario"]) for _, r in sonuclar.iterrows())
             if kimlik(a) is not None]
    ana_olcek = len(kontrol_kosulari("D1")) + 1
    last_kontrol = [a for a in adlar if bozulmasiz_mi(a) and a.endswith(" last_pt")]
    last_senaryo = [a for a in adlar
                    if a.endswith(" last_pt") and not bozulmasiz_mi(a)]
    esiksiz = sorted(a for a in adlar
                     if not bozulmasiz_mi(a) and not kontrol_kosulari(a))
    ajan = ajan_kaydi()
    deneme = len(ajan.get("cevaplar") or {})

    from teshis.degerlendirme.bootstrap import VAL_DIAGNOSTIC_BBOX_N

    az = [f"{a} (n={n})" for a, n in VAL_DIAGNOSTIC_BBOX_N.items() if n < 30]

    return [
        "**Tek model ailesi, koşu başına tek deneme.** Ölçülen her skor bir "
        "nokta tahminidir; koşu tekrarı olmadığı için güven aralığı "
        "hesaplanamaz.",
        f"**Gürültü tabanı {ana_olcek} bozulmasız koşudan geliyor.** Az "
        "gözlemle band gerçek yayılımı olduğundan küçük gösterir; eşikler "
        "muhtemelen hâlâ dar.",
        "**Referans tek bir koşudur (v00)** ve sağlıklı koşuların en "
        "zayıfıdır. Daha sağlam bir taban onların ortalaması olurdu.",
        f"**Nadir sınıflarda örnek yetersiz:** {', '.join(az)}. Bu "
        "sınıflardaki oranlar genellenemez.",
        f"**last.pt ölçeğinde yalnızca {len(last_kontrol)} bozulmasız koşu "
        f"var**, yani orada gürültü eşiği hiç hesaplanamıyor; o ölçekteki "
        f"{len(last_senaryo)} senaryonun farkı ölçülebiliyor ama gürültüden "
        "ayrılamıyor.",
        "**Kendi ölçeğinde eşiği olmayan koşular:** "
        f"{', '.join(esiksiz) or 'yok'}. Bunlar derecelendirilmez.",
        f"**Ajan denemesi {deneme} koşuluk tek turdur.** Koşu başına tekrar "
        "yok; ajanın 'sorun uydurmama' oranı için verilebilecek aralık çok "
        "geniş.",
        "**Final test seti hiç kullanılmadı** ve bu bilinçli bir karardır. "
        "Yani buradaki hiçbir sayı 'nihai test performansı' değildir.",
        "**Çalışma zamanı servisi (Aşama 2) tamamlanmadı.** Proje bir ölçüm "
        "ve teşhis altyapısıdır; canlı bir izleme servisi değildir.",
    ]


# Bilimsel sonuclar: her biri BIR olcume dayanir ve o olcum burada
# gosterilir. Metni sabit tutup sayiyi turetmek, ikisinin ayrismasini
# onler - bu projede tekrarlayan hata oruntusu tam olarak buydu.
def _bilimsel_sonuclar(sonuclar: pd.DataFrame) -> list[dict]:
    deger = {str(r["scenario"]): r for _, r in sonuclar.iterrows()}

    def m(ad, alan="mAP50"):
        return float(deger[ad][alan]) if ad in deger else None

    kartlar = []

    g = ne_gozlendi("D4")
    if g:
        kartlar.append({
            "baslik": "Farklı arızalar farklı metrik imzası bırakıyor",
            "metin": (
                "D4 (küçük nesne sinyal kaybı) toplam recall'ı "
                f"{g['metrikler']['recall']['fark']:+.4f} oynatıyor — bu, "
                f"recall'ın gürültü eşiğinin "
                f"({g['metrikler']['recall']['gurultu_esigi']:.4f}) ALTINDA "
                "kalıyor, yani rastgelelikten ayırt edilemiyor. precision ise "
                f"{g['metrikler']['precision']['fark']:+.4f} ile eşiği aşıyor. "
                "Hangi metriğin bozulduğu arızanın türünü söylüyor; hangisinin "
                "bozulmadığı da söylüyor."
            ),
        })

    kirilim_d4 = None
    try:
        import veri_seti as vs

        k = vs.kirilim("D4")
        ref = vs.kirilim("v00_saglikli")
        if k and ref:
            a = k["boyut_bandi_recall"]["cok_kucuk_16_alti"]["recall"]
            b = ref["boyut_bandi_recall"]["cok_kucuk_16_alti"]["recall"]
            kirilim_d4 = (a, b)
    except Exception:  # noqa: BLE001 - kirilim yoksa kart atlanir
        pass
    if kirilim_d4:
        kartlar.append({
            "baslik": "Toplam mAP yerel bir çöküşü tamamen gizleyebiliyor",
            "metin": (
                "D4'te <16 px bandının recall'ı "
                f"{kirilim_d4[1]:.4f} → {kirilim_d4[0]:.4f} düşüyor "
                f"({kirilim_d4[0] - kirilim_d4[1]:+.4f}), ama toplam mAP50 "
                "farkı bunun yanında küçük kalıyor. Yalnızca toplam metriğe "
                "bakan bir denetim bu arızayı görmez."
            ),
        })

    if m("E1") is not None and m("E1 last_pt") is not None:
        kartlar.append({
            "baslik": "Checkpoint seçimi aşırı uyumu gizleyebiliyor",
            "metin": (
                "E1'de 200 epoch süren ders kitabı niteliğinde bir aşırı uyum "
                "üretildi. En iyi checkpoint ile raporlandığında model "
                f"sağlıklı görünüyor (mAP50 {m('E1'):.4f}); son checkpoint'te "
                f"{m('E1 last_pt'):.4f}'e düşüyor. Sağlıklı referansın kendi "
                f"best→last düşüşü {m('v00_saglikli last_pt') - m('v00_saglikli'):+.4f} "
                "olduğu için, düşüşün kendisi değil tabandan ne kadar "
                "ayrıldığı anlamlı."
            ),
        })

    if m("E4 imgsz512") is not None:
        g = ne_gozlendi("E4 imgsz512")
        kartlar.append({
            "baslik": "Yanlış çıkarım çözünürlüğü büyük ve tek yönlü etki yapıyor",
            "metin": (
                "Aynı ağırlıklar 768 px yerine 512 px'te çalıştırıldığında "
                f"mAP50 {g['metrikler']['mAP50']['fark']:+.4f}, recall "
                f"{g['metrikler']['recall']['fark']:+.4f} değişiyor. Eğitim "
                "rastgeleliği hiç devrede değil: model dosyası birebir aynı, "
                "değişen tek şey çıkarım ayarı."
            ),
        })

    esik = None
    g = ne_gozlendi("D1")
    if g:
        esik = g["metrikler"]["recall"]["gurultu_esigi"]
    if esik:
        kartlar.append({
            "baslik": "Gürültü ölçülmeden \"etki\" iddiası kurulamaz",
            "metin": (
                "Aynı veri, aynı protokol, yalnızca farklı rastgelelik "
                f"tohumu: recall'da {esik:.4f} kadar fark çıkabiliyor. Bu, "
                "birçok senaryonun ölçülen 'etkisinden' büyük. Tek kontrol "
                "koşusuyla yapılan ilk ölçüm gürültüyü kat kat küçük "
                "gösteriyordu."
            ),
        })

    ajan = ajan_kaydi()
    puanlar = list((ajan.get("puanlar") or {}).values())
    if puanlar:
        n = len(puanlar)
        teshis = sum(p["diagnosis_score"] for p in puanlar) / n
        kanit = sum(p["evidence_score"] for p in puanlar) / n
        kartlar.append({
            "baslik": "Ajan kanıt üretiyor ama doğru nedeni bulmakta zayıf",
            "metin": (
                f"{n} koşuluk denemede doğru neden teşhisi %{teshis * 100:.0f}, "
                f"kanıt bileşeni %{kanit * 100:.0f}. Ajan her koşuda "
                "savunulabilir sayısal kanıt ve sınırlama üretiyor; ayırt "
                "edici olan tek bileşen teşhisin kendisi. Rubrik ortalaması "
                "bu yüzden tek başına 'ajan başarısı' olarak okunamaz."
            ),
        })
    return kartlar


def goster() -> None:
    sonuclar = load_results()
    st.title("Sonuçlar ve Sınırlamalar")
    st.markdown(
        "Projenin toplu hükmü. Hipotez tablosu ölçümlerden türetilir; "
        "sınırlamalar da defterden — ikisi de elle güncellenmez."
    )

    st.markdown("## Bilimsel sonuçlar")
    kartlar = _bilimsel_sonuclar(sonuclar)
    for satir in range(0, len(kartlar), 2):
        for sutun, kart in zip(st.columns(2), kartlar[satir:satir + 2]):
            with sutun:
                stil.kutu(f"<b>{kart['baslik']}</b><br>"
                          f'<span class="yorum">{kart["metin"]}</span>')
                st.write("")

    st.markdown("---")
    st.markdown("## Hipotezler ve hükümler")
    tablo = _hipotez_tablosu(sonuclar)
    if not tablo.empty:
        sayim = tablo["hüküm"].value_counts()
        stil.kpi_satiri([
            ("Desteklendi", int(sayim.get("Desteklendi", 0)),
             "birden fazla metrik eşiği aşıyor"),
            ("Kısmen desteklendi", int(sayim.get("Kısmen desteklendi", 0)),
             "yalnızca tek metrik"),
            ("Desteklenmedi",
             int(sayim.get("Desteklenmedi (gürültü içinde)", 0)),
             "hiçbir metrik eşiği aşmıyor"),
            ("Derecelendirilemeyen", len(_derecelendirilemeyenler(sonuclar)),
             "referansı veya eşiği yok"),
        ])
        st.dataframe(tablo.drop(columns=["_tur"]), hide_index=True,
                     width="stretch", height=420)
        stil.yorum(
            "\"Desteklendi\" hükmü yalnızca ölçüme dayanır: birden fazla genel "
            "metrik, kendi ölçeğinin gürültü eşiğini aşıyorsa. Beklentinin "
            "metnine bakılarak verilmez."
        )

    with st.expander("Derecelendirilmeyen koşular ve nedenleri"):
        st.dataframe(_derecelendirilemeyenler(sonuclar), hide_index=True,
                     width="stretch")

    st.markdown("---")
    st.markdown("## Neyi HENÜZ söyleyemiyoruz")
    st.markdown(
        "Projenin asıl sorusu \"bir LLM bozulmayı teşhis edebilir mi?\" idi. "
        "**Bu soruyu kesin cevaplayacak örneklem henüz yok.**"
    )
    for madde in _sinirlamalar(sonuclar):
        st.markdown(f"- {madde}")

    stil.kutu(
        "<b>Bu bölümün amacı bulguları zayıflatmak değil.</b> Hangilerinin ne "
        "kadar dayanıklı olduğunu açıkça söylemek. Gürültü tabanı ölçüldükten "
        "sonra bir dizi iddia geri çekildi ve bir senaryo bulgu olmaktan "
        "çıktı; bu, ölçümün çalıştığının kanıtıdır."
    )

    st.markdown("---")
    st.markdown("## Sonraki adımlar")
    for madde in (
        "Sağlıklı bir `last.pt` kontrol koşusu eğitmek — o ölçekteki dört "
        "senaryo şu an eşiksiz duruyor.",
        "`final_best.pt` ve `yolo26n` ölçekleri için sağlıklı referans "
        "koşuları — D2b final_best şu an karşılaştırılamaz durumda.",
        "Her senaryo için çoklu seed: nokta tahmini yerine dağılım.",
        "Ajan denemesinin tekrarı — koşu başına tek deneme, doğruluk "
        "oranına aralık vermeye yetmiyor.",
        "Çalışma zamanı servisi (Aşama 2): ölçüm altyapısının canlı bir "
        "izleme hattına bağlanması.",
    ):
        st.markdown(f"- {madde}")

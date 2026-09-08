"""Deney Tasarimi ve Sinirlar.

Bu bolum projenin en savunulabilir kismidir: neyi kontrol ettigimiz ve neyi
HALA soyleyemedigimiz. Sinirlari gizlemek yerine one koymak, bir savunmada
en guclu karttir.

"Neyi henuz soyleyemiyoruz" listesi elle yazilmaz; sayilari defterden
turetilir. Elle yazildiginda geride kalmisti: metin "dort saglikli kosu"
diyordu, defterde ise dort kontrol + referans vardi.
"""

from __future__ import annotations

import streamlit as st

import stil
from data_loader import ajana_gizlenenler, load_results


def _gurultu_tablosu():
    from teshis.degerlendirme.gurultu import alt_grup_bandi

    band = alt_grup_bandi()
    satirlar = []
    for alan, gruplar in band.items():
        for grup, d in gruplar.items():
            satirlar.append({
                "kırılım": alan.replace("_recall", ""),
                "grup": grup,
                "bbox": d["bbox_n"],
                "band": d["band"],
                "std": d["std"],
                "koşu": d["n_kosu"],
            })
    return sorted(satirlar, key=lambda s: -s["band"])


def _sinirlar(sonuclar) -> list[str]:
    """Neyi HENUZ soyleyemiyoruz - sayilari defterden turetilerek."""
    from teshis.degerlendirme.karsilastirilabilirlik import (
        bozulmasiz_mi,
        kimlik,
        kontrol_kosulari,
    )

    # Defterde OLMAYAN satirlar (demo'nun ekledigi sentetik "Baseline" gibi)
    # sayilara girmemeli; kimligi olmayan bir kosunun olcegi de yoktur.
    adlar = [a for a in (str(r["scenario"]) for _, r in sonuclar.iterrows())
             if kimlik(a) is not None]
    ana_olcek = len(kontrol_kosulari("D1")) + 1          # kontroller + referans
    last_kontrol = [a for a in adlar if bozulmasiz_mi(a) and a.endswith(" last_pt")]
    last_senaryo = [a for a in adlar
                    if a.endswith(" last_pt") and not bozulmasiz_mi(a)]
    esiksiz = sorted(a for a in adlar
                     if not bozulmasiz_mi(a) and not kontrol_kosulari(a))
    return [
        "Tek model, koşu başına tek deneme, tekrar yok. Ölçülen skor bir "
        "nokta tahminidir; güven aralığı hesaplanamaz.",
        "Ajanın \"sorun uydurmama\" oranı için verilebilecek aralık çok geniş "
        "(dört saf kontrolün birinde uydurdu).",
        f"Ana ölçeğin gürültü bandı {ana_olcek} bozulmasız koşudan hesaplandı; "
        "az gözlemle band gerçek yayılımı olduğundan küçük gösterir.",
        "Referans tek bir koşudur (v00) ve sağlıklı koşuların en zayıfıdır; "
        "daha sağlam bir taban onların ortalaması olurdu.",
        f"last.pt ölçeğinde yalnızca {len(last_kontrol)} bozulmasız koşu var, "
        f"yani orada gürültü eşiği hiç hesaplanamıyor: o ölçekteki "
        f"{len(last_senaryo)} senaryonun farkı ölçülebiliyor ama gürültüden "
        "ayrılamıyor.",
        "Kendi ölçeğinde eşiği olmayan koşular: "
        f"{', '.join(esiksiz) or 'yok'}. Bunlar için derecelendirme yapılmaz.",
    ]


def goster() -> None:
    sonuclar = load_results()
    st.title("Deney Tasarımı ve Sınırlar")

    st.markdown("## Kontrollü deney kurgusu")
    st.markdown(
        "Gerçek hayatta \"model neden kötü çalışıyor?\" sorusu cevaplanamaz, "
        "çünkü aynı anda birden fazla şey yanlış olabilir. Burada tersi "
        "yapılır: sağlıklı bir referans eğitilir, sonra **her seferinde tek "
        "bir şey** kasıtlı olarak bozulur."
    )

    a, b = st.columns(2)
    with a:
        stil.ust_baslik("değişen")
        stil.kutu(
            "<b>D serisi:</b> veri (etiket, dağılım, temsil)<br>"
            "<b>E serisi:</b> eğitim veya çıkarım ayarı<br>"
            "<b>C serisi:</b> yalnızca rastgelelik tohumu"
        )
    with b:
        stil.ust_baslik("sabit tutulan")
        stil.kutu(
            "Kilitli tanı seti (val_diagnostic) — hiç değişmez<br>"
            "Eğitim protokolü — tek dosyada beyan edilir<br>"
            "Başlangıç modeli, seed, çözünürlük, checkpoint<br>"
            "Test seti — final aşamasına kadar yasak"
        )

    st.markdown("### Karşılaştırılabilirlik kuralı")
    st.markdown(
        "Bir farkın bozulmaya ait olabilmesi için aday ve referansın dört "
        "kimlik alanında da aynı olması gerekir: **başlangıç modeli, "
        "değerlendirme kümesi, çıkarım çözünürlüğü, checkpoint**. Biri "
        "farklıysa fark, bozulmanın değil o alanın etkisini taşır."
    )
    stil.kutu(
        "<b>Bu kural sonradan eklendi.</b> Önce bütün koşular tek bir "
        "referansla karşılaştırılıyordu; bu yüzden içinde hiçbir bozulma "
        "olmayan bir koşu (v00'in son checkpoint'i) \"güçlü bozulma kanıtı\" "
        "olarak etiketlenmişti. Aynı filtre ajan araçlarında zaten vardı — "
        "kural iki yerde yaşayınca biri geride kaldı."
    )

    st.markdown("### Protokol sapmaları beyan edilir")
    st.markdown(
        "E serisi protokolü **kasıtlı olarak** bozar. Sapmalar koda dağılmış "
        "bayraklarla değil, tek bir protokol dosyasında beyan edilir; her "
        "koşu kendi sapmasını manifestinde taşır. Böylece hangi koşunun "
        "protokolden nerede ayrıldığı tek yerden okunur."
    )
    stil.kutu(
        "Somut örnek: E3'ün tezi \"öğrenme oranı 100 kat yüksek\". Ultralytics "
        "<code>optimizer=auto</code> iken lr0'ı <b>yok sayar</b>; bu fark "
        "edilmeseydi E3 sessizce sağlıklı bir koşuya dönüşür ve \"kararsızlık "
        "gözlenmedi\" diye raporlanırdı. E3 sapması artık optimizer'ı da "
        "açıkça yazıyor."
    )

    st.markdown("---")
    st.markdown("## Ajanın körleştirilmesi")
    st.markdown(
        "Ajan hangi koşunun hangi senaryo olduğunu bilmez. Filtreler yapısaldır "
        "ve testlidir; ad listesine dayanmaz."
    )
    gizli = ajana_gizlenenler()
    st.dataframe(
        [{"alan": k, "durum": v} for k, v in gizli.items()],
        hide_index=True, width="stretch",
    )
    stil.yorum(
        "Cevap anahtarının gönderilmemesi kritik: puanlama ancak ajan cevabı "
        "tamamlandıktan sonra ayrı bir yerel işlemle yapılır."
    )

    st.markdown("---")
    st.markdown("## Gürültü tabanı")
    st.markdown(
        "Hiçbir şey bozulmadan, yalnızca rastgelelik tohumu değiştirilerek "
        "eğitilen koşular arasındaki yayılım. Bir farkın bu bandın altında "
        "kalması, o farkın **saf rastgelelikten ayırt edilemediği** anlamına "
        "gelir — büyüklüğü ne olursa olsun."
    )
    st.dataframe(_gurultu_tablosu(), hide_index=True, width="stretch")
    stil.yorum(
        "Dikkat: bu yalnızca küçük örneklem sorunu değil. termal grubu 858 "
        "bbox taşır ama bandı hituav'ın (2.165 bbox) bandının on katından "
        "fazladır; bazı gruplar gerçekten oynaktır."
    )

    st.markdown("---")
    st.markdown("## Neyi HENÜZ söyleyemiyoruz")
    st.markdown(
        "Projenin asıl sorusu \"bir LLM bozulmayı teşhis edebilir mi?\" idi. "
        "**Bu soruyu cevaplayacak örneklem henüz yok.**"
    )
    for madde in _sinirlar(sonuclar):
        st.markdown(f"- {madde}")

    stil.kutu(
        "<b>Bu bölümün amacı:</b> bulguları zayıflatmak değil, hangilerinin "
        "ne kadar dayanıklı olduğunu açıkça söylemek. Gürültü tabanı "
        "ölçüldükten sonra bir dizi iddia geri çekildi; bu, ölçümün "
        "çalıştığının kanıtıdır."
    )

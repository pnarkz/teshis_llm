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
    """Her senaryo icin: ne bekleniyordu, ne olctuk, hangi hukum.

    HUKUM NASIL VERILIR
    -------------------
    Beklenti metni serbest yazidir ("insan recall ve AP belirgin duser") ve
    makine tarafindan ayristirilamaz. Dolayisiyla hukum, beklentinin
    METNINI degil, olculen etkinin VARLIGINI ve YONUNU degerlendirir.

    GERCEK HATA: ilk surum yalnizca "esigi asan metrik var mi" diye
    bakiyordu. Sonuc D1'de sacmaydi - beklenti "insan recall duser" iken
    tablo, genel `mAP50_95` degerinin ARTMASI uzerinden "Kismen
    desteklendi" yaziyordu. Bir metrigin yukselmesi, dususu ongoren bir
    hipotezi desteklemez.

    Bozulma senaryolarinda beklenen yon her zaman DUSUS'tur; bu tek varsayim
    makine tarafindan uygulanabilir. Esigi asan etki artis yonundeyse hukum
    "beklenmedik yon" olur ve bu bir bulgudur, bir basari degil.

    Tablo `katalog.senaryolar()` uzerinden kurulur: 14 arastirma senaryosu,
    E3'un olculemedigi ve E4/D6a'nin eslenik olcum oldugu dahil. Onceden
    `results.csv` satirlarindan kuruluyordu ve E3b'nin iki seed'i ayri satir
    olarak gorunurken E3, E4 ve D6a tabloda hic yoktu.
    """
    import katalog

    satirlar = []
    for s in katalog.senaryolar():
        kosu = s["ana_kosu"]
        o = ozet(kosu) if kosu else {}
        gozlem = ne_gozlendi(kosu) if kosu else {}
        e = s["ana_etki"]

        if not kosu:
            hukum, tur = "Ölçülemedi", "notr"
            gozlenen = "eğitim ıraksadı; değerlendirilebilir model üretilmedi"
        elif gozlem.get("karsilastirma_turu") == "eslenik":
            hukum, tur = "Eşlenik ölçüm", "ikincil"
            gozlenen = (f"{e['alan']} {e['fark']:+.4f} — aynı ağırlıklar, "
                        "eğitim gürültüsü devrede değil" if e else "—")
        elif not gozlem.get("kontrol_kosu_sayisi"):
            hukum, tur = "Eşik yok", "uyari"
            gozlenen = "bu ölçekte kontrol koşusu yok"
        elif not e:
            hukum, tur = "Gürültüyü aşan etki yok", "kritik"
            gozlenen = "hiçbir etki gürültü eşiğini aşmıyor"
        elif e["fark"] > 0:
            # Bozulma bekleniyordu, olculen etki ARTIS yonunde.
            hukum, tur = "Beklenmedik yönde etki (yükseliş)", "uyari"
            gozlenen = f"{e['alan']} {e['fark']:+.4f} (yükseliş)"
        else:
            asan = gozlem.get("asan_metrikler") or []
            guclu = len(asan) >= 2 or (e.get("kirilim") and e["oran"] >= 5)
            # ETIKET HESAPLANANI ANLATIR, hipotez hukmu VERMEZ. "Desteklendi"
            # demek, beklentinin metnini denetledigimizi ima ederdi - oysa
            # denetlemiyoruz. D2a'nin beklentisi "mAP50-95 mAP50'den daha
            # fazla duser"; kucuk nesne recall dususu bu beklentiyi tek
            # basina dogrulamaz ama eski etiket "Desteklendi" yaziyordu.
            hukum = ("Eşiği aşan düşüş (birden fazla metrik)" if guclu
                     else "Eşiği aşan düşüş (tek metrik)")
            tur = "guclu" if guclu else "uyari"
            gozlenen = f"{e['alan']} {e['fark']:+.4f}"

        satirlar.append({
            "kod": s["kod"],
            "senaryo": s["ad"],
            # YAML katlanmis skaler (">") sonunda satir sonu birakir ve
            # tabloda kacis dizisi olarak gorunuyordu.
            "beklenen etki": " ".join(str(o.get("beklenen_etki") or "—").split()),
            "gözlenen": gozlenen,
            "eşiği aşan metrikler": ", ".join(gozlem.get("asan_metrikler") or []) or "—",
            "hüküm": hukum,
            "_tur": tur,
        })
    return pd.DataFrame(satirlar)


def _guc_dagilimi(sonuclar: pd.DataFrame) -> pd.DataFrame:
    """Yalnizca DERECELENDIRILEBILEN kosularin kanit gucu dagilimi.

    Kontrol kosulari, referanslar, eslenik olcumler ve esigi olmayan kosular
    bu dagilima girmez - girseydi "guclu bulgu" sayisi, hicbir bozulma
    icermeyen kosularla sisirilirdi.

    Genel Bakis'tan tasindi: giris ekraninda kanit gucu dagilimi, izleyici
    daha "kanit gucu" kavramini duymadan gosteriliyordu.
    """
    sayim: dict[str, int] = {}
    for _, r in sonuclar.iterrows():
        seviye = kanit_gucu(str(r["scenario"]))["seviye"]
        if seviye in DERECELENDIRILEN:
            sayim[seviye] = sayim.get(seviye, 0) + 1
    return pd.DataFrame(
        {"koşu": [sayim.get(s, 0) for s in DERECELENDIRILEN]},
        index=[stil.seviye_adi(s) for s in DERECELENDIRILEN],
    )


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


def _ajan_orneklem_notu(eski_deneme: int) -> str:
    """Ajan orneklemi hakkindaki sinirlama - DENEYDEN turetilir.

    Bu satir elle yaziliydi ("koşu başına tekrar yok") ve 2026-09-09
    tekrarli deneyi calistirildiginda YANLIS hale geldi; metin geride
    kaldigi icin ekranda hala tekrar olmadigi yaziyordu. Artik deneyin
    kendisinden okunuyor.
    """
    from data_loader import ajan_deneyi

    deney = ajan_deneyi()
    if not deney:
        return (f"**Ajan denemesi {eski_deneme} koşuluk tek turdur.** Koşu "
                "başına tekrar yok; ajanın 'sorun uydurmama' oranı için "
                "verilebilecek aralık çok geniştir.")
    p = deney["puan"]
    tekrar = p["gozlem"] // max(p["kosu"], 1)
    kontrol = (p.get("rol_bazli") or {}).get("kontrol") or {}
    return (
        f"**Ajan deneyi {p['kosu']} koşu × {tekrar} tekrar = {p['gozlem']} "
        f"gözlem.** Tekrarlar model kararlılığını ölçer, senaryo evrenindeki "
        "belirsizliği değil: 9 bozulma senaryosu ve tek bir model üzerinden "
        "hesaplanan oran bu senaryoların dışına genellenemez."
        + (f" Kontrol koşularında {kontrol['gozlem']} gözlemin tamamı doğru."
           if kontrol.get("dogru_teshis") == 1.0 else "")
    )


def _ajan_sonraki_adim() -> str:
    from data_loader import ajan_deneyi

    if ajan_deneyi():
        return ("Ajan deneyinin başka bir model ailesiyle tekrarı — şu anki "
                "sonuç tek bir modele ait.")
    return ("Ajan denemesinin tekrarı — koşu başına tek deneme, doğruluk "
            "oranına aralık vermeye yetmiyor.")


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
        "**Tek model ailesi, senaryo başına tek eğitim koşusu.** Ölçülen her "
        "metrik bir nokta tahminidir; aynı senaryo yeniden eğitilmediği için "
        "senaryo metriğine güven aralığı verilemez. (Ajan deneyinin tekrarı "
        "var; o ayrı bir ölçüdür — model kararlılığını ölçer, senaryo "
        "evrenindeki belirsizliği değil.)",
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
        _ajan_orneklem_notu(deneme),
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
    st.markdown("## Beklentiler ve ölçülen etki")
    tablo = _hipotez_tablosu(sonuclar)
    if not tablo.empty:
        sayim = tablo["hüküm"].value_counts()
        stil.kpi_satiri([
            ("Eşiği aşan düşüş",
             int(sayim.get("Eşiği aşan düşüş (birden fazla metrik)", 0)),
             "birden fazla metrikte"),
            ("Tek metrikte düşüş",
             int(sayim.get("Eşiği aşan düşüş (tek metrik)", 0)),
             "yalnızca bir metrikte"),
            ("Beklenmedik yön",
             int(sayim.get("Beklenmedik yönde etki (yükseliş)", 0)),
             "etki var ama yükseliş yönünde"),
            ("Gürültüyü aşan etki yok",
             int(sayim.get("Gürültüyü aşan etki yok", 0)),
             "hiçbir etki eşiği aşmıyor"),
            ("Değerlendirilemeyen",
             int(sayim.get("Ölçülemedi", 0)) + int(sayim.get("Eşlenik ölçüm", 0))
             + int(sayim.get("Eşik yok", 0)),
             "ölçülemedi / eşlenik / eşiksiz"),
        ])
        st.dataframe(tablo.drop(columns=["_tur"]), hide_index=True,
                     width="stretch", height=460)
        stil.kutu(
            "<b>Bu tablo bir hipotez testi değildir ve öyle olduğunu iddia "
            "etmez.</b> Beklenti sütunu serbest metindir (\"mAP50-95 "
            "mAP50'den daha fazla düşer\") ve makine tarafından "
            "ayrıştırılamaz; hangi <i>sınıfın</i> hangi <i>metriğinin</i> ne "
            "kadar düşmesi beklendiği otomatik denetlenmez. Bu yüzden hüküm "
            "sütunu yalnızca <b>ölçülen şeyi</b> adlandırır: etki gürültü "
            "eşiğini aşıyor mu, kaç metrikte ve hangi yönde. Beklentiyle "
            "eşleşip eşleşmediğine okuyucu karar verir — iki sütun yan yana."
            '<div class="yorum" style="margin-top:.5rem">Önceki iki sürüm de '
            "yanlıştı. İlki yalnızca \"eşiği aşan metrik var mı\" diye "
            "bakıyordu ve D1'de beklenti \"insan recall düşer\" iken genel "
            "mAP50-95'in <b>artması</b> üzerinden \"kısmen desteklendi\" "
            "yazıyordu. İkincisi yönü düzeltti ama hâlâ \"Desteklendi\" "
            "diyordu — herhangi bir metrikteki düşüş, senaryonun kendi "
            "beklentisini doğrulamaz.</div>"
        )

    dagilim = _guc_dagilimi(sonuclar)
    derece_disi = _derecelendirilemeyenler(sonuclar)
    a, b = st.columns([2, 3])
    with a:
        stil.ust_baslik("kanıt gücü dağılımı")
        # st.bar_chart DIKEY cizip etiketleri 90 derece dondururken
        # "gürültü içinde" gibi uzun etiketler okunmuyordu. Yatay cubuk
        # etiketi duz yazar; sayi da cubugun ucunda durur.
        import grafik

        st.altair_chart(
            grafik.yatay_bar(
                dagilim.reset_index(names="seviye"), "seviye", "koşu",
                alan_adi="koşu sayısı", etiket=True, sirala="-x"),
            width="stretch",
        )
    with b:
        stil.ust_baslik("derecelendirilmeyen koşular")
        st.dataframe(derece_disi, hide_index=True, width="stretch")
        stil.yorum(
            f"Yalnızca kendi ölçeğinde referansı VE gürültü eşiği olan "
            f"{int(dagilim['koşu'].sum())} koşu derecelendirilir; "
            f"kalan {len(derece_disi)} koşu nedeniyle birlikte yanda."
        )

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
        _ajan_sonraki_adim(),
        "Çalışma zamanı servisi (Aşama 2): ölçüm altyapısının canlı bir "
        "izleme hattına bağlanması.",
    ):
        st.markdown(f"- {madde}")

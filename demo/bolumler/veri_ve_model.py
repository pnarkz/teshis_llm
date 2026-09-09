"""Veri ve Saglikli Model: her senaryonun uzerine kuruldugu taban.

Bu bolum onceki konsolda hic yoktu ve en somut eksikti: sunumda "veri neye
benziyor?" sorusu gelince klasor acmak gerekiyordu. Burada veri seti kendi
raporundan anlatilir, sonra sagligi referans model kunyesiyle birlikte
gosterilir.

Hicbir sayi elle yazilmaz (bkz. veri_seti.py). Bir alan kayitta yoksa
"kayıtta yok" yazar - uydurulmaz.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import grafik
import gorseller
import stil
import veri_seti as vs


def _veri_seti_sekmesi() -> None:
    rapor = vs.veri_raporu()
    tani = vs.tani_seti()
    if not rapor:
        st.warning(
            "`reports/veri_raporu.json` bulunamadı. Veri istatistikleri bu "
            "dosyadan türetilir; üretmek için: "
            "`python -m teshis.veri.istatistik --config config.yaml`"
        )
        return

    toplam = rapor.get("toplam") or {}
    stil.kpi_satiri([
        ("Toplam görüntü", f"{toplam.get('goruntu', 0):,}".replace(",", "."),
         "train + val + test"),
        ("Toplam bbox", f"{toplam.get('bbox', 0):,}".replace(",", "."),
         "etiketlenmiş nesne"),
        ("Kilitli tanı seti",
         f"{tani['goruntu_sayisi']:,}".replace(",", ".")
         if tani.get("goruntu_sayisi") else "kayıtta yok",
         f"{tani.get('bbox_sayisi', 0):,} bbox".replace(",", ".")
         if tani.get("bbox_sayisi") else "künye okunamadı"),
        ("Veri kaynağı", len(rapor.get("kaynak_toplam") or {}),
         "ayrı görüntü kaynağı"),
    ])
    stil.yorum(
        "Bütün metrikler kilitli tanı seti üzerinde ölçülür. Bu set val "
        "bölümünden türetilmiştir; kaynak tekilliği ve split ayrıklığı "
        "gözetilerek seçilmiş, sonra kilitlenmiştir."
    )
    if tani.get("_yeniden_kuruldu"):
        st.info(
            "Kilitli tanı setinin künyesi (`val_diagnostic/manifest.json`) bu "
            "makinede yok — o dizin Git dışıdır. Sayılar depoyla gelen ölçüm "
            f"dosyalarından yeniden kuruldu: görüntü sayısı "
            f"`reports/kirilim/{tani.get('_kaynak')}`, sınıf başına bbox "
            "`teshis/degerlendirme/bootstrap.py`. Kaynak grubu başına bbox "
            "sayısı hiçbir izlenen dosyada tam durmuyor; uydurulmadı, boş "
            "bırakıldı."
        )

    st.markdown("### Bölüm dağılımı")
    bolumler = pd.DataFrame(vs.split_dagilimi())
    # Kilitli tani seti icin "bos etiket" sayilmadi; None birakilirsa sutun
    # object tipine duser ve ekranda "None" yazar. Sayisal tutulunca bos
    # hucre olur - dogru okuma da budur: deger yok, sifir degil.
    for sutun in ("görüntü", "bbox", "boş etiket", "görüntü başına nesne"):
        bolumler[sutun] = pd.to_numeric(bolumler[sutun], errors="coerce")
    # Cubuk degeri UCUNDA yaziyor; yanina ayni sayilari tekrar eden bir
    # tablo koymaya gerek yok. Tablonun tasidigi FAZLA sutunlar (goruntu,
    # bos etiket, goruntu basina nesne) acilir bolumde duruyor.
    # Etiket ONCEDEN bicimlenir: Vega'nin ",.0f" bicimi "131,700" yazar,
    # sayfanin geri kalani ise "154.141" kullaniyor. Ayni ekranda iki
    # binlik ayraci olmaz.
    st.altair_chart(
        grafik.yatay_bar(
            bolumler.assign(_etiket=[f"{int(v):,}".replace(",", ".")
                                     for v in bolumler["bbox"]]),
            "bölüm", "bbox", alan_adi="bbox sayısı",
            etiket="_etiket", sirala=None),
        width="stretch",
    )
    with st.expander("Tam sayılar (görüntü, boş etiket, görüntü başına nesne)"):
        st.dataframe(bolumler, hide_index=True, width="stretch")
    stil.yorum(
        "Kilitli tanı seti val'in bir alt kümesidir; ayrı bir bölüm değildir. "
        "Test bölümü final aşamasına kadar hiç açılmadı."
    )

    st.markdown("### Sınıf dağılımı")
    kapsam = st.radio(
        "Kapsam", ["Kilitli tanı seti", "Tüm veri", "train", "val", "test"],
        horizontal=True, key="sinif_kapsam",
    )
    anahtar = {"Kilitli tanı seti": "tani", "Tüm veri": "toplam"}.get(kapsam, kapsam)
    siniflar = pd.DataFrame(vs.sinif_dagilimi(anahtar))
    # Etikette hem sayi hem pay: "2.718 · %67,7". Tablo ayni ikisini
    # tekrar ediyordu.
    etiketli = siniflar.assign(_etiket=[
        f"{int(r['bbox']):,}".replace(",", ".")
        + (f"  ·  %{r['pay']:.1f}" if r.get("pay") is not None else "")
        for _, r in siniflar.iterrows()
    ])
    st.altair_chart(
        grafik.yatay_bar(etiketli, "sınıf", "bbox", alan_adi="bbox sayısı",
                         etiket="_etiket"),
        width="stretch",
    )

    nadir = [s for s in vs.sinif_dagilimi("tani") if s["bbox"] < 30]
    if nadir:
        stil.kutu(
            "<b>Bu dengesizlik bir sonuç değil, bir sınırlamadır.</b> Kilitli "
            "tanı setinde "
            + ", ".join(f"{s['sınıf']} ({s['bbox']} bbox)" for s in nadir)
            + " sınıfları çok az örnekle temsil ediliyor. Bu sınıflarda "
            "ölçülen oranlar tek tek nesnelere aşırı duyarlıdır: bir nesnenin "
            "bulunup bulunmaması yüzdeyi büyük ölçüde oynatır. Bu yüzden "
            "nadir sınıf sonuçları hiçbir yerde tek başına kanıt sayılmaz."
        )

    st.markdown("### Veri kaynağı dağılımı")
    a, b = st.columns(2)
    with a:
        stil.ust_baslik("tüm veri")
        st.altair_chart(
            grafik.yatay_bar(pd.DataFrame(vs.kaynak_dagilimi("toplam")),
                             "kaynak", "bbox", alan_adi="bbox"),
            width="stretch",
        )
    with b:
        stil.ust_baslik("kilitli tanı seti")
        st.altair_chart(
            grafik.yatay_bar(pd.DataFrame(vs.kaynak_dagilimi("tani")),
                             "kaynak", "bbox", alan_adi="bbox",
                             renk=stil.IKINCIL),
            width="stretch",
        )
    stil.yorum(
        "Kaynak grupları ayrı çekim koşullarını temsil eder; bu yüzden model "
        "performansı kaynağa göre ciddi biçimde değişebiliyor. D5 senaryosu "
        "tam olarak bunu ölçer, gürültü bandı da kaynak kırılımında en geniş "
        "yayılımı gösterir."
    )

    st.markdown("### Veri sağlığı taraması")
    uyarilar = vs.saglik_uyarilari()
    if uyarilar:
        st.dataframe(
            pd.DataFrame([{"bulgu": u["bulgu"], "adet": u["adet"]} for u in uyarilar]),
            hide_index=True, width="stretch",
        )
        stil.yorum(
            "Sıfır olan denetimler listelenmez. Kalanlar kabul edilmiş "
            "durumlardır: çok küçük kutular gerçek bir termal tespit "
            "zorluğudur (D4 senaryosu bunu ölçer), sınır dışına taşan kutular "
            "ise kırpma sırasında oluşur ve sayıları ihmal edilebilir."
        )
    else:
        st.success("Veri sağlık taramasında sıfırdan farklı bulgu yok.")


def _ornek_galerisi() -> None:
    durum = gorseller.durum()
    if durum["kaynak"] == "yok":
        st.warning(
            "Etiketli örnek bulunamadı. Kilitli tanı seti (`val_diagnostic/`) "
            "bu makinede yok ve taşınabilir örnek seti de üretilmemiş. "
            "Üretmek için: `python demo/gorseller.py`"
        )
        return
    if durum["kaynak"] == "yedek":
        st.info(durum["açıklama"] + " Metrikler her zaman tam kilitli set "
                "üzerinde ölçülür; bu galeri yalnızca görsel örneklemedir.")

    kayitlar = list(gorseller.katalog())
    a, b, c, d = st.columns([2, 2, 2, 1])
    with a:
        siniflar = sorted({s for k in kayitlar for s in k["siniflar"]})
        sinif = st.selectbox("Sınıf", ["hepsi"] + siniflar)
    with b:
        kaynaklar = sorted({k["kaynak"] for k in kayitlar})
        kaynak = st.selectbox("Kaynak", ["hepsi"] + kaynaklar)
    with c:
        bant = st.selectbox(
            "Nesne boyutu",
            ["hepsi"] + [gorseller.BANT_ADI[b] for b in gorseller.BANT_ADI],
        )
    with d:
        adet = st.number_input("Adet", 1, 12, 4)

    if sinif != "hepsi":
        kayitlar = [k for k in kayitlar if sinif in k["siniflar"]]
    if kaynak != "hepsi":
        kayitlar = [k for k in kayitlar if k["kaynak"] == kaynak]

    # Boyut bandi goruntuyu acmayi gerektirir; yalnizca aday kayitlar icin
    # hesaplanir, butun set icin degil.
    kayitlar = gorseller.boyut_bantlarini_doldur(kayitlar[: int(adet) * 6])
    if bant != "hepsi":
        kod = next(k for k, v in gorseller.BANT_ADI.items() if v == bant)
        kayitlar = [k for k in kayitlar if kod in k["bantlar"]]

    if not kayitlar:
        st.warning("Bu filtrelerle örnek bulunamadı; filtreleri gevşetin.")
        return

    st.markdown(
        " ".join(
            stil.rozet(ad, tur) for ad, tur in
            [("taşıt", "aday"), ("insan", "ikincil"),
             ("UAP", "uyari"), ("UAI", "kritik")]
        ),
        unsafe_allow_html=True,
    )
    for satir in range(0, min(len(kayitlar), int(adet)), 2):
        for sutun, kayit in zip(st.columns(2), kayitlar[satir:satir + 2]):
            with sutun:
                gorsel = gorseller.kutulu_gorsel(kayit)
                if gorsel is None:
                    st.image(str(kayit["yol"]), width="stretch")
                else:
                    st.image(gorsel, width="stretch")
                sayim = {}
                for kutu in kayit["kutular"]:
                    ad = gorseller.SINIF_ADI.get(kutu["sinif"], "?")
                    sayim[ad] = sayim.get(ad, 0) + 1
                st.markdown(
                    f'<div class="yorum"><code>{kayit["kaynak"]}</code> · '
                    f"{kayit['nesne']} nesne · "
                    + ", ".join(f"{a}: {n}" for a, n in sorted(sayim.items()))
                    + "</div>",
                    unsafe_allow_html=True,
                )


def _model_sekmesi() -> None:
    SENARYO = "v00_saglikli"
    kunye = vs.egitim_ayarlari(SENARYO)
    metrik = vs.metrikler(SENARYO)

    stil.kutu(
        "<b>Bu model bütün karşılaştırmaların tabanıdır.</b> Senaryolarla "
        "<i>birebir aynı</i> protokolde, hiç bozulmamış veriyle eğitildi. Bir "
        "senaryonun metriği bununla karşılaştırılır — ama yalnızca dört kimlik "
        "alanı (başlangıç modeli, değerlendirme kümesi, çözünürlük, "
        "checkpoint) aynıysa."
    )

    if metrik:
        stil.kpi_satiri([
            ("mAP50", f"{metrik['mAP50']:.4f}", "IoU 0.50"),
            ("mAP50-95", f"{metrik['mAP50_95']:.4f}", "IoU 0.50:0.95"),
            ("precision", f"{metrik['precision']:.4f}", ""),
            ("recall", f"{metrik['recall']:.4f}", ""),
        ])

    st.markdown("### Fine-tune ne kazandırdı?")
    st.markdown(
        "Dağıtımdaki model (`main_model.pt`) hiç fine-tune edilmeden aynı "
        "kilitli set üzerinde ölçüldü. Senaryolarla aynı protokolde "
        "EĞİTİLMEDİĞİ için bozulma karşılaştırmalarında taban olarak "
        "kullanılmaz — ama fine-tune'un kendi etkisini görünür kılar."
    )
    from data_loader import load_results

    defter = {str(r["scenario"]): r for _, r in load_results().iterrows()}
    if "Baseline" in defter and metrik:
        taban = defter["Baseline"]
        satirlar = []
        for alan, ad in (("mAP50", "mAP50"), ("mAP50_95", "mAP50-95"),
                         ("precision", "precision"), ("recall", "recall")):
            b, v = float(taban[alan]), float(defter["v00_saglikli"][alan])
            satirlar.append({"metrik": ad, "fine-tune öncesi": round(b, 4),
                             "sağlıklı referans (v00)": round(v, 4),
                             "fark": round(v - b, 4)})
        fine = pd.DataFrame(satirlar)
        # Gruplu cubuk yerine SIFIR MERKEZLI fark. Dort metrigin ikisi de
        # 0.9 civarindaydi; yan yana iki cubugu gozle kiyaslamak farki
        # okunmaz kiliyordu. Okunmasi gereken sey zaten fark: en buyugu
        # 0.03. Sifir merkezli grafikte yon de dogrudan gorunuyor -
        # precision YUKSELIYOR, digerleri dusuyor.
        st.altair_chart(
            grafik.fark_profili(fine, "metrik", "fark"),
            width="stretch",
        )
        with st.expander("Ham değerler (fine-tune öncesi / sonrası)"):
            st.dataframe(fine, hide_index=True, width="stretch")
        stil.yorum(
            "Bu fark bir senaryo etkisi DEĞİLDİR; iki farklı eğitim durumunun "
            "karşılaştırmasıdır. Bozulma senaryolarının tabanı her zaman v00'dur "
            "— fine-tune edilmemiş modele göre ölçüm yapmak, bozulma etkisiyle "
            "fine-tune etkisini birbirine karıştırırdı."
        )

    st.markdown("### Eğitim künyesi")
    st.markdown(
        "Değerler koşunun kendi `args.yaml` ve `run_manifest.json` "
        "dosyalarından okunur — protokolde ne yazdığından değil, eğitimin ne "
        "ile koştuğundan."
    )
    kunye_df = pd.DataFrame(
        [{"alan": k, "değer": ("kayıtta yok" if v is None else str(v))}
         for k, v in kunye.items()]
    )
    # Yirmi satir iki tablo halinde duruyordu ve hepsi ayni agirliktaydi.
    # Bir karsilastirmanin gecerli olup olmadigi DORT kimlik alanina bakar
    # (baslangic modeli, degerlendirme kumesi, cozunurluk, checkpoint) -
    # onlar gorunur kalir. Geri kalani inceleyen icin, izleyici icin degil.
    KIMLIK = ("başlangıç ağırlığı", "değerlendirme kümesi",
              "çıkarım çözünürlüğü", "checkpoint", "seed", "veri sürümü")
    kartlar = [(a, kunye.get(a)) for a in KIMLIK if kunye.get(a) is not None]
    if kartlar:
        stil.kpi_satiri([(a, str(d), "") for a, d in kartlar[:3]])
        if len(kartlar) > 3:
            stil.kpi_satiri([(a, str(d), "") for a, d in kartlar[3:6]])
        stil.yorum(
            "İlk dört alan <b>karşılaştırılabilirlik kimliğidir</b>: bir "
            "senaryonun metriği ancak bu dördü de aynı olan bir referansla "
            "karşılaştırılabilir. Seed ve veri sürümü tekrarlanabilirlik için."
        )
    with st.expander("Eğitim künyesinin tamamı (optimizer, lr, batch, süre)"):
        a, b = st.columns(2)
        yari = (len(kunye_df) + 1) // 2
        with a:
            st.dataframe(kunye_df.iloc[:yari], hide_index=True, width="stretch")
        with b:
            st.dataframe(kunye_df.iloc[yari:], hide_index=True, width="stretch")

    not_ = vs.optimizer_notu(SENARYO)
    if not_:
        stil.kutu("<b>Dikkat.</b> " + not_)

    st.markdown("### Sınıf bazlı performans")
    siniflar = pd.DataFrame(vs.sinif_metrikleri(SENARYO))
    if not siniflar.empty:
        uzun = siniflar.melt(
            id_vars="sınıf", value_vars=["AP50", "AP50-95", "recall"],
            var_name="metrik", value_name="değer",
        )
        st.altair_chart(
            grafik.gruplu_bar(uzun, "sınıf", "değer", "metrik",
                              yatay=True, alan_adi="değer"),
            width="stretch",
        )
        with st.expander("Sayısal değerler ve örnek sayıları"):
            st.dataframe(siniflar, hide_index=True, width="stretch")
        stil.yorum(
            "UAP ve UAI'nin yüksek görünen skorları yanıltıcıdır: sırasıyla "
            "15 ve 17 bbox ile ölçülüyorlar. 'az örnek' sütunu bunu işaretler."
        )

    kirilim = vs.kirilim(SENARYO)
    if kirilim:
        st.markdown("### Kırılımlı performans")
        # Iki tablo yerine iki grafik: kirilimin anlatmak istedigi sey
        # "hangi grup geride kaliyor" ve bunu bir cubuk anında soyluyor.
        # Etikette recall'in yaninda ORNEK SAYISI da var - az ornekli bir
        # grubun yuksek recall'i tek basina okunmasin diye.
        def _etiketle(veri, kategori):
            return veri.assign(_etiket=[
                f"{r['recall']:.4f}   (n={int(r['gerçek kutu']):,})".replace(",", ".")
                if r.get("gerçek kutu") else f"{r['recall']:.4f}"
                for _, r in veri.iterrows()
            ])

        a, b = st.columns(2)
        with a:
            stil.ust_baslik("nesne boyutuna göre recall")
            boyut = pd.DataFrame([
                {"bant": vs.BANT_ADI.get(k, k), "recall": v.get("recall"),
                 "gerçek kutu": v.get("gercek_kutu")}
                for k, v in (kirilim.get("boyut_bandi_recall") or {}).items()
            ])
            st.altair_chart(
                grafik.yatay_bar(_etiketle(boyut, "bant"), "bant", "recall",
                                 alan_adi="recall", etiket="_etiket",
                                 sirala=None),
                width="stretch",
            )
        with b:
            stil.ust_baslik("veri kaynağına göre recall")
            kaynak = pd.DataFrame([
                {"kaynak": k, "recall": v.get("recall"),
                 "gerçek kutu": v.get("gercek_kutu")}
                for k, v in (kirilim.get("kaynak_recall") or {}).items()
            ])
            st.altair_chart(
                grafik.yatay_bar(_etiketle(kaynak, "kaynak"), "kaynak",
                                 "recall", alan_adi="recall",
                                 etiket="_etiket", renk=stil.IKINCIL),
                width="stretch",
            )
        stil.yorum(
            "Sağlıklı modelde bile küçük nesnelerde ve bazı kaynaklarda "
            "belirgin düşüş var. Bu, bozulma değil verinin kendi zorluğudur — "
            "senaryolar bu tabanın ÜZERİNE eklenen etkiyi ölçer."
        )

    egri = vs.egitim_egrisi(SENARYO)
    if egri is not None and not egri.empty:
        st.markdown("### Eğitim eğrisi")
        sutunlar = [s for s in ("train/cls_loss", "val/cls_loss")
                    if s in egri.columns]
        if sutunlar:
            uzun = egri.melt(id_vars="epoch", value_vars=sutunlar,
                             var_name="seri", value_name="kayıp")
            st.altair_chart(
                grafik.cizgi(uzun, "epoch", "kayıp", seri="seri",
                             alan_adi="sınıflandırma kaybı"),
                width="stretch",
            )
        stil.yorum(
            f"Koşu {kunye.get('durduğu epoch')} epoch'ta erken durdu "
            f"(sabır: {kunye.get('erken durdurma sabrı')}). Train ve val "
            "kayıpları birlikte iniyor; aşırı uyum imzası yok. E1 senaryosu "
            "aynı grafikte açılan bir makas gösterir."
        )

    from data_loader import images_for

    gorsel_listesi = images_for(SENARYO)
    if gorsel_listesi:
        with st.expander("Ultralytics çıktıları (confusion matrix, eğriler)"):
            for g in gorsel_listesi:
                st.image(str(g), caption=g.name, width="stretch")


def goster() -> None:
    st.title("Veri ve Sağlıklı Model")
    st.markdown(
        "Her senaryo bu iki şeyin üzerine kurulur: **kilitli tanı seti** ve "
        "**hiç bozulmamış referans model**. İkisi de burada, kendi kaynak "
        "dosyalarından okunarak anlatılır."
    )

    veri, ornek, model = st.tabs(
        ["Veri seti", "Etiketli örnekler", "Sağlıklı referans model"]
    )
    with veri:
        _veri_seti_sekmesi()
    with ornek:
        _ornek_galerisi()
    with model:
        _model_sekmesi()

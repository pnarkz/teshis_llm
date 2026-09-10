"""LLM Teshis Ajani: kor teshis akisi.

Iki mod var ve ikisi de AYNI ekrani uretir:

- **Kayitli kosu** (varsayilan): tamamlanmis denemeden okunur. API harcamaz,
  her zaman calisir. Sunumun guvenli yolu budur.
- **Canli kosu**: ajan o anda calistirilir. On kontrol once sessizce yapilir;
  bir kosul eksikse hangisi oldugu yazar. Ucretsiz katman 20 istek/gun ve
  5 istek/dk ile sinirlidir.

KORLUK yapisaldir. Ajana yalnizca anonim `kosu_NN` kimligi ve olcum araclari
verilir; senaryo adi, bozulma aciklamasi, konfig ve cevap anahtari
gonderilmez. Filtre ad listesine degil, her kosunun kendi manifestine bakar
(teshis/ajan/araclar.py::ajana_uygun_mu).

Arayuz de bu korlugu YASATIR: gercek senaryo baslangicta gizlidir, sunucu da
izleyici de ajanla ayni bilgiyle baslar. "Gercegi goster" ile acilir.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

import ajan_katmani
import stil
from data_loader import (
    ajan_araclarini_calistir,
    ajan_deneyi,
    ajan_deneyi_kosu_bazli,
    ajan_kaydi,
    ajan_kaydi_var_mi,
    ajan_kosu_haritasi,
    ajana_gizlenenler,
)

# Kosu rozetleri: hangi kaydin nereden geldigi secicide gorunur.
KAYIT_ROZETI = {
    "ana": ("ana deneme", "guclu"),
    "tekrar": ("kontrol tekrarı", "ikincil"),
    "yok": ("kayıt yok", "uyari"),
}

ARAC_ACIKLAMA = {
    "baseline_metriklerini_getir": "sağlıklı referansın genel metrikleri",
    "kosu_metriklerini_getir": "bu koşunun genel metrikleri",
    "baseline_farkini_getir": "referansa göre farklar",
    "bbox_sayilarini_getir": "sınıf başına örnek sayısı",
    "boyut_bazli_recall_getir": "nesne boyutu kırılımı",
    "kaynak_bazli_recall_getir": "kaynak grubu kırılımı",
    "sinif_karisikligini_getir": "sınıf karışıklığı matrisi",
    "kosu_listesini_getir": "değerlendirilebilir koşuların listesi",
}

# Ajanin akisi - araclarin hangi sirayla cagrildigindan BAGIMSIZ olarak
# surecin kendisi. Canli modda hangi adimda olundugunu gostermek icin.
AKIS = [
    "Anonim koşu seçildi",
    "Sağlıklı referans metrikleri alındı",
    "Koşu metrikleri alındı",
    "Referans farkı hesaplandı",
    "Ajan gerekli gördüğü kırılımları sorguladı",
    "Kanıtlar değerlendirildi",
    "Teşhis üretildi",
    "Yapısal cevap doğrulandı",
]


def _korluk_paneli(kosu_id: str, senaryo: str, acik: bool) -> None:
    """Kor teshis paneli.

    Gercek senaryo BASLANGICTA GIZLIDIR - sunucu da ajanla ayni bilgiyle
    baslar. Bu, kor tasarimi anlatmakla kalmayip izleyiciye YASATIR: once
    kanita bakilir, sonra cevap acilir.
    """
    gercek = (
        f"<b>Gerçek senaryo:</b> {senaryo}" if acik
        else "<b>Gerçek senaryo:</b> <i>gizli — aşağıdan açabilirsiniz</i>"
    )
    stil.kutu(
        f"{gercek}<br>"
        f"<b>Ajana gönderilen kimlik:</b> <code>{kosu_id}</code><br>"
        f"<b>Senaryo adı ajandan gizlendi:</b> EVET"
    )
    with st.expander("Ajana ne gidiyor, ne gitmiyor"):
        st.dataframe(
            [{"alan": k, "durum": v} for k, v in ajana_gizlenenler().items()],
            hide_index=True, width="stretch",
        )
        stil.yorum(
            "Bu filtre bir ad listesi değildir: her koşunun kendi manifesti "
            "okunur ve protokol sapması taşıyan veya farklı ölçekte ölçülmüş "
            "koşular yapısal olarak dışarıda kalır."
        )


def _arac_zaman_cizelgesi(cagrilar: list[dict]) -> pd.DataFrame:
    """Ajanın kanıt toplama sırası.

    Sıra bilgi taşır: ajan önce genel metriklere bakıp sonra hangi kırılımı
    sorguladığı, akıl yürütmesinin izidir.
    """
    satirlar = []
    for i, c in enumerate(cagrilar):
        cevap = c.get("cevap")
        ozet = "—"
        if isinstance(cevap, dict):
            if cevap.get("hata"):
                ozet = f"hata: {cevap['hata']}"
            else:
                anahtarlar = [k for k in cevap if k not in ("hata",)][:4]
                ozet = ", ".join(str(k) for k in anahtarlar) or "boş"
        satirlar.append({
            "sıra": i + 1,
            "tur": c.get("tur"),
            "araç": c.get("arac"),
            "ne sorduğu": ARAC_ACIKLAMA.get(c.get("arac"), "—"),
            "sonuç": ozet,
            "hata": c.get("hata") or "—",
        })
    return pd.DataFrame(satirlar)


def _kanit_goster(kosu_id: str, kayit: dict) -> None:
    """Ajanin gordugu kanit: kayittan mi, yeniden mi uretildi?

    Bu ayrimi yapmak zorunlu. Kirilim araclarina sonradan gurultu bandi
    alanlari eklendi; dolayisiyla ESKI bir kaydin cevabini bugunku arac
    ciktisiyla yan yana koyup "ajanin gordugu kanit tam olarak budur" demek
    YANLIS olur. Ajan o alanlari hic gormemis olabilir.
    """
    kosu_kaydi = (kayit.get("arac_kaydi") or {}).get(kosu_id) or {}
    cagrilar = kosu_kaydi.get("arac_cagrilari") or []
    snapshot = [c for c in cagrilar if "cevap" in c]

    if snapshot:
        st.markdown(
            "Aşağıdaki çıktılar **denemenin kendi kaydından** geliyor: ajanın "
            "o an gördüğü değerlerin birebir kopyası."
        )
        surum = kosu_kaydi.get("arac_surumu")
        if surum:
            stil.yorum(f"Araç sürümü parmak izi: <code>{surum}</code>")
        with st.expander("Araç çıktıları (kayıttan)"):
            for c in snapshot:
                st.markdown(f"**{c.get('arac')}**")
                st.json(c.get("cevap"), expanded=False)
        return

    st.warning(
        "Bu koşunun kaydında araç **cevapları** saklanmamış — yalnızca hangi "
        "aracın çağrıldığı var. Aşağıdaki çıktı bugünün araçlarıyla yeniden "
        "üretildi. Araçlar deterministiktir ve API harcamaz, ancak sonradan "
        "gürültü bandı alanları eklendiği için ajanın o an gördüğü sürüm "
        "bundan farklı olabilir: bu, **yaklaşık** bir yeniden üretimdir."
    )
    with st.expander("Araç çıktıları (bugün yeniden üretildi)"):
        for ad, deger in ajan_araclarini_calistir(kosu_id).items():
            st.markdown(f"**{ad}**")
            st.json(deger, expanded=False)


def _cevap_kartlari(cevap: dict) -> None:
    """Teshis / kanit / guven / sinirlama - dort kart."""
    a, b = st.columns([3, 2])
    with a:
        stil.ust_baslik("teşhis")
        stil.kutu(f'<b style="font-size:1.05rem">{cevap.get("diagnosis", "—")}</b>')
    with b:
        stil.ust_baslik("güven")
        stil.kutu(
            stil.guven_rozeti(cevap.get("confidence", "—"))
            + '<div class="yorum">Bu değer ajanın kendi beyanıdır; kalibre '
              "edilmiş bir olasılık değildir. Doğrulukla ilişkisi ölçülmedi.</div>"
        )

    st.write("")
    c, d = st.columns(2)
    with c:
        stil.ust_baslik("sayısal kanıtlar")
        kanitlar = cevap.get("evidence") or []
        stil.kutu("<br>".join(f"• {k}" for k in kanitlar) or "—")
    with d:
        stil.ust_baslik("sınırlamalar ve önerilen sonraki ölçüm")
        sinirlar = cevap.get("limitations") or []
        icerik = "<br>".join(f"• {k}" for k in sinirlar) or "—"
        if cevap.get("next_measurement"):
            icerik += (f'<div class="yorum" style="margin-top:.5rem">'
                       f"<b>Sonraki ölçüm:</b> {cevap['next_measurement']}</div>")
        stil.kutu(icerik)


def _puan_goster(puan: dict) -> None:
    st.markdown("### Cevap anahtarıyla karşılaştırma")
    st.markdown(
        "Cevap anahtarı ajana **gönderilmedi**; puanlama cevap üretildikten "
        "sonra yerelde yapıldı."
    )
    a, b, c = st.columns(3)
    with a:
        stil.kpi("Gerçek senaryo (beklenen)", puan.get("expected", "—"))
    with b:
        stil.kpi("Ajanın teşhisi", puan.get("model_diagnosis", "—"))
    with c:
        stil.kpi("Teşhis puanı", puan.get("diagnosis_score", "—"),
                 f"tespit-farkındalıklı: {puan.get('diagnosis_score_tespit', '—')}")
    st.write("")
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
            "Bu koşuda bozulma kanıtta anlamlı iz bırakmıyor; gerekçe: "
            f"{puan['tespit_notu']}"
        )


def _canli_calistir(kosu_id: str) -> dict | None:
    """Ajani o anda calistirir. Hata turlerini AYIRT EDEREK raporlar.

    Basarisiz bir cagri uygulamayi cokertmez; kullaniciya kayitli moda
    gecme secenegi kalir ve bunun canli sonuc OLMADIGI acikca yazar.
    """
    from teshis.ajan import ajan as ajan_modulu

    baslangic = time.time()
    with st.status(f"{kosu_id} için canlı teşhis üretiliyor...",
                   expanded=True) as durum:
        for adim in AKIS[:4]:
            st.write(f"• {adim}")
        try:
            cevap, kayit = ajan_modulu.teshis_uret(
                kosu_id, model=ajan_katmani.model_adi()
            )
        except Exception as hata:  # noqa: BLE001
            bilgi = ajan_katmani.hata_turu(hata)
            durum.update(label=bilgi["baslik"], state="error")
            getattr(st, bilgi["seviye"])(bilgi["mesaj"])
            return None
        st.write(f"• {AKIS[4]} ({len(kayit)} araç çağrısı)")
        st.write(f"• {AKIS[5]}")
        st.write(f"• {AKIS[6]}")
        hatalar = ajan_katmani.sonuc_dogrula(cevap)
        st.write(f"• {AKIS[7]}" + (" — uyarı var" if hatalar else " — geçti"))
        sure = time.time() - baslangic
        durum.update(
            label=f"Tamamlandı ({sure:.1f} sn, {len(kayit)} araç çağrısı)",
            state="complete",
        )
    if hatalar:
        st.warning("Şema doğrulaması uyarı üretti: " + "; ".join(hatalar))
    return {"cevap": cevap, "kayit": kayit, "sure": sure}


def _tekrarli_deney_paneli() -> None:
    """Tekrarli, tam ustverili deneyin sonuclari.

    Asagidaki "kayitli koşu" akisindan AYRI bir deneydir ve onunla
    birlestirilmez: eski deneme farkli bir kod haliyle, ustverisiz ve arac
    cevabi saklanmadan uretildi. Iki deneyi tek bir basari oraninda toplamak
    ikisini de yaniltir.
    """
    deney = ajan_deneyi()
    if not deney:
        return

    puan = deney["puan"]
    tam = deney["tam_mi"]
    baslik = ("Tekrarlı deney — tamamlandı" if tam
              else "Tekrarlı deney — DEVAM EDİYOR")
    with st.expander(f"{baslik}  ·  {len(puan['runs'])}"
                     f"/{deney['beklenen_gozlem']} gözlem", expanded=True):
        if not tam:
            st.warning(
                "Bu deney henüz tamamlanmadı. Aşağıdaki oranlar eksik "
                "gözlemle hesaplanmıştır ve nihai değildir."
            )
        st.markdown(
            "Bu, **aşağıdaki kayıtlı denemeden ayrı** bir deneydir; ikisi "
            "birleştirilmez. Her gözlem model, araç sürümü, Git commit'i, "
            "ham cevap ve **her araç çağrısının cevabının anlık kaydını** "
            "taşır — eski denemede araç cevapları hiç saklanmamıştı."
        )

        rol_adi = {"saglikli_referans": "Sağlıklı referans",
                   "kontrol": "Kontrol (yalnızca seed farklı)",
                   "bozulma_senaryosu": "Bozulma senaryosu"}
        st.dataframe(pd.DataFrame([
            {"Rol": rol_adi.get(rol, rol), "Koşu": d["kosu"],
             "Gözlem": d["gozlem"], "Katı puan": d["dogru_teshis"],
             "Tespit-farkındalıklı": d["tespit_farkindalikli"]}
            for rol, d in (puan.get("rol_bazli") or {}).items()
        ]), hide_index=True, width="stretch")
        st.caption(
            "Oranlar rol bazlı verilir ve TOPLANMAZ: kontrol koşuları "
            "\"sorun uyduruyor mu\", bozulma senaryoları \"nedeni "
            "bulabiliyor mu\" sorusunu ölçer. Tek bir ortalama ikisini de "
            "yanıltır. Katı puan uygulanan bozulmanın adını arar; "
            "tespit-farkındalıklı puan, bozulmanın kilitli tanı setinde "
            "anlamlı iz bırakmadığı koşularda (D1, D3b, D6b) "
            "\"anlamlı değişim yok\" cevabını da doğru sayar."
        )

        kosular = ajan_deneyi_kosu_bazli()
        degisken = [k for k in kosular if not k["hukum_tutarli"]]
        tekrar = kosular[0]["tekrar"] if kosular else 0
        if kosular and not degisken:
            st.success(
                f"Tekrar tutarlılığı: {len(kosular)} koşunun tamamı "
                f"{tekrar} tekrarında da **aynı** hükmü verdi. Sözel ifade "
                "değişiyor, hüküm değişmiyor."
            )
        elif degisken:
            st.info(
                f"{len(degisken)}/{len(kosular)} koşuda tekrarlar farklı "
                "hüküm verdi: "
                + ", ".join(f"{k['kosu_id']} ({k['senaryo']}) {k['kati']}"
                            for k in degisken)
            )

        with st.expander("Koşu bazlı puanlar (tekrarlar ortalanmadan)"):
            st.dataframe(pd.DataFrame([
                {"Koşu": k["kosu_id"], "Senaryo": k["senaryo"],
                 "Rol": rol_adi.get(k["rol"], k["rol"]),
                 "Beklenen": k["beklenen"],
                 "Katı (tekrarlar)": str(k["kati"]),
                 "Tespit-farkındalıklı": str(k["tespit"])}
                for k in kosular
            ]), hide_index=True, width="stretch")

        ust = deney["ustveri"]
        st.caption(
            f"Deney kimliği: `{deney['deney_id']}` · model: "
            f"`{ust.get('model')}` · üretim: "
            f"`{ust.get('uretim_parametreleri', {}).get('automatic_function_calling')}`"
        )
        if puan.get("_baglam_notu"):
            st.caption(puan["_baglam_notu"])
        if puan.get("_uyari"):
            st.warning(puan["_uyari"])


def goster() -> None:
    st.title("LLM Teşhis Ajanı")
    st.markdown(
        "Ajana yalnızca anonim metrikler ve kırılım araçları verilir; hangi "
        "koşunun hangi senaryo olduğunu bilmez. Teşhisini kendi seçtiği "
        "kanıtla üretir."
    )

    # Once TEKRARLI deneyin sonucu, sonra tek tek kosu incelemesi. Panel
    # kendi basligini tasir ve deney yoksa (taze klon, henuz kosulmamis)
    # hic acilmaz.
    _tekrarli_deney_paneli()

    kayit = ajan_kaydi()
    harita = ajan_kosu_haritasi()
    # Ajana verilen HER kosu listelenir; kaydi olmayan varsa rozetiyle
    # gorunur. Onceden yalnizca ana denemenin 11 kosusu listeleniyordu ve
    # kosu_12/kosu_13'un ayri dosyalardaki cevaplari hic gorunmuyordu.
    kosular = sorted(set(harita) | set(kayit["cevaplar"])
                     | set(kayit.get("tekrarlar") or {}))
    varsayilan = kosular.index("kosu_08") if "kosu_08" in kosular else 0

    a, b = st.columns([2, 3])
    with a:
        # Secicide GERCEK AD GOSTERILMEZ: sunum sirasinda izleyici de sunucu
        # da ajanla ayni bilgiyle baslar.
        # Secicide YALNIZCA kimlik gorunur. Oneri etiketleri buraya
        # yazilmisti ("belirgin bir bozulma...") ve izleyiciye gercegi
        # senaryo adi acilmadan sizdiriyordu - korluk yalnizca ajan icin
        # degil, salondaki herkes icin gecerli olmali.
        kosu_id = st.selectbox(
            "Koşu", kosular, index=varsayilan,
            format_func=lambda k: (
                # "sunum icin onerilen" etiketi kaldirildi: hangi kosuyu
                # gostereceginin sunumu yapan kisinin bilmesi yeter,
                # ekranda yazmasi izleyiciye bir sey anlatmiyor - ustelik
                # "bu kosu ozel secilmis" izlenimi veriyordu.
                f"{k}  ·  {KAYIT_ROZETI[ajan_kaydi_var_mi(k)][0]}"
            ),
        )
    with b:
        mod = st.radio(
            "Kaynak", ["Kayıtlı koşu", "Canlı çalıştır"], horizontal=True,
            help=("Kayıtlı koşu API harcamaz ve her zaman çalışır. Canlı mod "
                  "ücretsiz katman sınırlarına tabidir (20 istek/gün, "
                  "5 istek/dk)."),
        )
    # "Sunum notu" bolumu kaldirildi. Ekranda durmasi gereksizdi: notun tek
    # okuyucusu sunumu yapan kisi ve o zaten neyi gosterecegini biliyor.
    # Ustelik acildiginda gercek senaryoyu sizdiriyordu - sayfanin geri
    # kalani korlugu tam da bunun icin koruyor.
    # ajan_katmani.ONERILEN_CANLI duruyor; canli kosu icin hangi kosularin
    # uygun oldugunu belirlemekte kullaniliyor.

    senaryo = harita.get(kosu_id, "?")
    anahtar = f"acik_{kosu_id}"
    if anahtar not in st.session_state:
        st.session_state[anahtar] = False
    acik = st.session_state[anahtar]

    st.markdown("---")
    _korluk_paneli(kosu_id, senaryo, acik)

    st.markdown("### Ajana verilen kanıt")
    _kanit_goster(kosu_id, kayit)

    st.markdown("---")
    if mod == "Canlı çalıştır":
        _canli_bolum(kosu_id)
        return

    cevap = kayit["cevaplar"].get(kosu_id)
    cagrilar = (kayit["arac_kaydi"].get(kosu_id) or {}).get("arac_cagrilari", [])
    tekrarlar = (kayit.get("tekrarlar") or {}).get(kosu_id) or []

    if not cevap and tekrarlar:
        # Ana denemede yok ama kontrol tekrarinda var: kosu_12 ve kosu_13
        # ana denemeden SONRA eklendi. Kayit ayri bir dosyada duruyordu ve
        # demo "kayitli cevap yok" diyordu - kayip degil, bagli degildi.
        st.info(
            f"**{kosu_id} ana denemede yok.** Bu koşu kontrol koşusu olarak "
            "sonradan eklendi ve ayrı bir **kontrol tekrarı** olarak "
            "çalıştırıldı. Aşağıdaki cevap o kayıttan geliyor; ana denemenin "
            "puan ortalamalarına dahil DEĞİLDİR."
        )
        cevap = tekrarlar[0]["cevap"]
        cagrilar = tekrarlar[0]["arac_cagrilari"]
        tekrar_puani = tekrarlar[0].get("puan")
    else:
        tekrar_puani = None
    if not cevap:
        st.warning(f"{kosu_id} için kayıtlı cevap yok. Canlı modu deneyebilirsiniz.")
        return
    if cagrilar:
        st.markdown("### Araç çağrıları")
        st.dataframe(_arac_zaman_cizelgesi(cagrilar), hide_index=True,
                     width="stretch")
        stil.yorum(
            f"Ajan bu koşuda {len(cagrilar)} araç çağırdı. Hangi kanıtı "
            "isteyeceğine kendisi karar verdi; araçlar önceden çalıştırılıp "
            "cevaba eklenmedi."
        )

    _cevap_kartlari(cevap)

    if kayit["cevaplar"].get(kosu_id) and tekrarlar:
        _tekrar_karsilastirmasi(kayit["cevaplar"][kosu_id], tekrarlar)

    st.markdown("---")
    if not acik:
        stil.kutu(
            "Ajanın teşhisi yukarıda. <b>Gerçek senaryo hâlâ gizli.</b> "
            "Kanıta bakıp kendi kararınızı verdikten sonra açın."
        )
        if st.button("Gerçek senaryoyu ve puanı göster", type="primary"):
            st.session_state[anahtar] = True
            st.rerun()
    else:
        _gercegi_goster(senaryo, kayit, kosu_id, anahtar)

    with st.expander("Ham cevap (JSON)"):
        st.json(cevap)

    st.markdown("---")
    _kontrol_tekrarlari_ozeti(kayit)
    _denemenin_butunu(kayit)


def _tekrar_karsilastirmasi(ana: dict, tekrarlar: list[dict]) -> None:
    """Ayni kosu iki kez soruldugunda ajan ayni seyi mi diyor?

    Tekrarlanabilirlik sinyali: ana denemenin cevabi ile kontrol tekrarinin
    cevabi yan yana. Ortalamalara KARISTIRILMAZ - iki ayri deneydir.
    """
    with st.expander("Bu koşu ikinci kez de soruldu — ajan aynı şeyi dedi mi?"):
        satirlar = [{
            "deneme": "ana deneme",
            "teşhis": ana.get("diagnosis"),
            "güven": ana.get("confidence"),
            "kanıt sayısı": len(ana.get("evidence") or []),
        }]
        for t in tekrarlar:
            c = t["cevap"]
            satirlar.append({
                "deneme": f"kontrol tekrarı ({t['dosya']})",
                "teşhis": c.get("diagnosis"),
                "güven": c.get("confidence"),
                "kanıt sayısı": len(c.get("evidence") or []),
            })
        st.dataframe(pd.DataFrame(satirlar), hide_index=True, width="stretch")
        ayni = len({s["teşhis"] for s in satirlar}) == 1
        stil.yorum(
            "İki denemede de aynı teşhis üretildi." if ayni else
            "Teşhis metinleri birebir aynı değil. Bu tek başına tutarsızlık "
            "demek değildir — puanlama serbest metni değil, teşhisin "
            "ANLAMINI eşleştirir; ama tekrar sayısı bir güven aralığı "
            "vermeye yetmiyor."
        )


def _canli_bolum(kosu_id: str) -> None:
    kontroller = ajan_katmani.on_kontrol(kosu_id)
    hazir = ajan_katmani.hazir_mi(kontroller)

    st.markdown("### Canlı çalıştırma ön kontrolü")
    if hazir:
        st.success("Canlı ajan hazır.")
    else:
        eksik = [k["ad"] for k in kontroller if not k["tamam"]]
        st.warning("Canlı ajan çalıştırılamaz — eksik: " + ", ".join(eksik))
    st.dataframe(
        pd.DataFrame([
            {"kontrol": k["ad"],
             "durum": "tamam" if k["tamam"] else "eksik",
             "açıklama": k["not"]}
            for k in kontroller
        ]),
        hide_index=True, width="stretch",
    )
    stil.yorum(
        "API anahtarının kendisi hiçbir yerde gösterilmez — yalnızca tanımlı "
        "olup olmadığı."
    )

    if not hazir:
        st.info(
            "Kayıtlı koşu moduna geçerek aynı ekranı API harcamadan "
            "görebilirsiniz."
        )
        return

    if st.button("Ajanı çalıştır", type="primary"):
        sonuc = _canli_calistir(kosu_id)
        if sonuc:
            st.markdown("### Araç çağrıları")
            st.dataframe(_arac_zaman_cizelgesi(sonuc["kayit"]),
                         hide_index=True, width="stretch")
            _cevap_kartlari(sonuc["cevap"])
            with st.expander("Ham cevap (JSON)"):
                st.json(sonuc["cevap"])
        else:
            st.info(
                "Canlı çağrı başarısız oldu. **Kayıtlı koşu** moduna geçerek "
                "aynı koşunun daha önce kaydedilmiş cevabını görebilirsiniz — "
                "ancak o bir canlı sonuç değildir."
            )
    else:
        st.info(
            "Canlı mod seçildi. 'Ajanı çalıştır' düğmesine basıldığında "
            "sağlayıcıya gerçek bir istek gönderilir."
        )


def _gercegi_goster(senaryo: str, kayit: dict, kosu_id: str,
                    anahtar: str) -> None:
    from teshis.degerlendirme.senaryo_ozeti import ozet as senaryo_ozeti

    try:
        o = senaryo_ozeti(senaryo)
    except Exception:  # noqa: BLE001 - ozet uretilemezse akis durmasin
        o = {}

    st.markdown("### Gerçek senaryo")
    stil.kutu(
        f"<b>{senaryo}</b>"
        + (f"<br>{o['ne_olcuyor'].strip()}" if o.get("ne_olcuyor") else "")
    )
    puan = kayit["puanlar"].get(kosu_id)
    if puan:
        _puan_goster(puan)
    if st.button("Yeniden gizle"):
        st.session_state[anahtar] = False
        st.rerun()


def _kontrol_tekrarlari_ozeti(kayit: dict) -> None:
    """Kontrol tekrarlari - ana denemeden AYRI bir sonuc.

    Bu, projenin en zayif iddiasinin dogrudan kanitidir: "ajan bozulma
    yokken sorun uydurmuyor". Ana denemede yalnizca iki saf kontrol vardi;
    tekrarlarla gozlem sayisi artti ve hepsi band'li araclarla uretildi.
    """
    tekrarlar = kayit.get("tekrarlar") or {}
    if not tekrarlar:
        return
    satirlar = []
    for kosu_id, kayitlar in sorted(tekrarlar.items()):
        for t in kayitlar:
            p = t.get("puan") or {}
            satirlar.append({
                "koşu": kosu_id,
                "beklenen": p.get("expected", "—"),
                "ajanın teşhisi": t["cevap"].get("diagnosis"),
                "teşhis puanı": p.get("diagnosis_score"),
                "kaynak dosya": t["dosya"],
            })
    dogru = sum(1 for s in satirlar if s["teşhis puanı"] == 1.0)
    with st.expander(
        f"Kontrol tekrarları — {dogru}/{len(satirlar)} doğru (ana denemeden ayrı)"
    ):
        st.dataframe(pd.DataFrame(satirlar), hide_index=True, width="stretch")
        stil.kutu(
            "<b>Bunlar ana denemenin parçası değildir.</b> Farklı tarihte ve "
            "farklı araç sürümüyle (gürültü bandı alanları eklendikten "
            "sonra) üretildiler; ortalamalar birleştirilirse iki ayrı deneyi "
            "tek orana katan yanıltıcı bir sayı çıkar."
            f'<div class="yorum" style="margin-top:.5rem">Hepsi bozulmasız '
            f"koşu ve ajan {dogru}/{len(satirlar)}'inde \"bozulma yok\" dedi. "
            "Bu, projenin en zayıf iddiasına (\"ajan sorun uydurmuyor\") "
            "doğrudan kanıt ekler — ama örneklem hâlâ küçüktür.</div>"
        )


def _ajan_skorlari(kayit: dict) -> dict[str, float]:
    """Rubrik ortalamasini bilesenlerine ayirir.

    Tek sayi vermek yaniltici: kanit ve sinirlama bilesenleri her kosuda tam
    puan aliyor, dolayisiyla ortalama teshis dogrulugundan cok daha yuksek
    cikiyor.

    Genel Bakis'ta bunun birebir kopyasi duruyordu; artik tek yerde.
    """
    puanlar = list(kayit.get("puanlar", {}).values())
    if not puanlar:
        return {}
    n = len(puanlar)
    return {
        "teshis": sum(p["diagnosis_score"] for p in puanlar) / n,
        "teshis_tespit": sum(p["diagnosis_score_tespit"] for p in puanlar) / n,
        "kanit": sum(p["evidence_score"] for p in puanlar) / n,
        "sinir": sum(p["limitation_score"] for p in puanlar) / n,
        "rubrik": kayit.get("ozet", {}).get("mean_score"),
    }


def _denemenin_butunu(kayit: dict) -> None:
    st.markdown("### Denemenin bütünü")
    skor = _ajan_skorlari(kayit)
    if not skor:
        return
    teshis, tespit = skor["teshis"], skor["teshis_tespit"]
    kanit, sinir, rubrik = skor["kanit"], skor["sinir"], skor["rubrik"]

    a, b, c = st.columns(3)
    with a:
        stil.kpi("Doğru neden teşhisi", f"%{teshis * 100:.0f}",
                 "asıl performans ölçüsü")
    with b:
        stil.kpi("Tespit-farkındalıklı", f"%{tespit * 100:.0f}",
                 "bozulmanın izi yoksa ceza yok")
    with c:
        stil.kpi("Rubrik ortalaması",
                 f"%{rubrik * 100:.0f}" if rubrik else "—",
                 "üç bileşenin ortalaması")
    stil.kutu(
        "<b>Bu üç sayı aynı şeyi ölçmez.</b> Rubrik ortalaması üç bileşenin "
        f"ortalamasıdır ve ikisi doymuştur: kanıt %{kanit * 100:.0f}, "
        f"sınırlama %{sinir * 100:.0f} — her koşuda tam puan. Ayırt eden tek "
        "bileşen teşhistir. Doğru nedeni bulma oranı "
        f"%{teshis * 100:.0f}'dir ve ajanın gerçek performansını temsil eden "
        "sayı budur."
    )

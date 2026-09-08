"""Hata Analizi: skorun altindaki tek tek kareler.

Metrik bir ozetdir; hata neye benziyor sorusunun cevabi degildir. Bu bolum
her kosunun en sorunlu karelerini siralayip yaninda **sagliklı modelin ayni
kareyi nasil gordugunu** koyar - fark boylece skor degil goruntu uzerinden
okunur.

Renk sozlesmesi hakkinda durustluk notu
---------------------------------------
Galeri goruntuleri `teshis/degerlendirme/hata_galerisi.py` tarafindan
ONCEDEN cizilir ve iki renk kullanir: **yesil = gercek etiket (GT)**,
**kirmizi = modelin tahmini**. Yani goruntude "yanlis pozitif turuncu,
kacirilan nesne sari" gibi bir ayrim YOKTUR; yesil bir kutunun yaninda
kirmizi yoksa o nesne kacirilmis, kirmizi bir kutunun yaninda yesil yoksa
uydurulmus demektir. Olmayan bir renk sozlesmesini varmis gibi anlatmak,
projenin butun metodolojisiyle celisirdi; onun yerine gercek sozlesme
yaziliyor ve sayisal ayrim tabloda veriliyor.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import gorseller
import stil
from data_loader import error_galleries, gorsel_coz, gorsel_kaynagi, images_for

SIRALAMA = {
    "toplam hata skoru": "score",
    "yanlış negatif (kaçırılan)": "false_negatives",
    "yanlış pozitif (uydurulan)": "false_positives",
    "düşük IoU": "mean_iou",
}


def _gt_bilgisi(kaynak_dosya: str) -> dict | None:
    """Karenin GERCEK etiketleri - kilitli tani setinden okunur.

    Galeri kaydi yalnizca hata sayilarini tasir; "kac nesne vardi" ve "hangi
    sinifta" bilgisi etiket dosyasindan gelir. Boylece aciklama uydurulmaz.
    """
    for kayit in gorseller.katalog():
        if kayit["dosya"] == kaynak_dosya:
            gorseller.boyut_bantlarini_doldur([kayit])
            sayim: dict[str, int] = {}
            for k in kayit["kutular"]:
                ad = gorseller.SINIF_ADI.get(k["sinif"], "?")
                sayim[ad] = sayim.get(ad, 0) + 1
            return {"nesne": kayit["nesne"], "siniflar": sayim,
                    "kaynak": kayit["kaynak"], "bantlar": kayit["bantlar"]}
    return None


def _aciklama(kayit: dict, gt: dict | None, saglikli: dict | None) -> str:
    """Secilen kare icin otomatik, tamamen turetilmis okuma notu."""
    fn = kayit.get("false_negatives") or 0
    fp = kayit.get("false_positives") or 0
    iou = kayit.get("mean_iou")
    parcalar = []

    if gt:
        bulunan = max(gt["nesne"] - fn, 0)
        parcalar.append(
            f"Bu karede <b>{gt['nesne']}</b> gerçek nesne var "
            f"({', '.join(f'{a}: {n}' for a, n in sorted(gt['siniflar'].items()))}). "
            f"Model <b>{bulunan}</b> tanesini buldu, <b>{fn}</b> tanesini "
            f"kaçırdı ve <b>{fp}</b> fazladan kutu üretti."
        )
        parcalar.append(
            f"Kaynak: <code>{gt['kaynak']}</code> · boyut bandı: "
            + ", ".join(gorseller.BANT_ADI.get(b, b) for b in gt["bantlar"])
        )
    else:
        parcalar.append(
            f"Model <b>{fn}</b> nesne kaçırdı ve <b>{fp}</b> fazladan kutu "
            "üretti. (Gerçek etiket sayısı okunamadı: kilitli tanı seti bu "
            "makinede yok.)"
        )
    if iou:
        parcalar.append(f"Eşleşen kutuların ortalama IoU'su: <b>{iou:.3f}</b>.")

    if saglikli:
        s_fn = saglikli.get("false_negatives") or 0
        s_fp = saglikli.get("false_positives") or 0
        yon = []
        if fn != s_fn:
            yon.append(f"kaçırma {s_fn} → {fn}")
        if fp != s_fp:
            yon.append(f"fazladan kutu {s_fp} → {fp}")
        parcalar.append(
            "<b>Sağlıklı modele göre:</b> " + (", ".join(yon) if yon
                                               else "aynı hata sayıları")
            + "."
        )
    return "<br>".join(parcalar)


def goster() -> None:
    st.title("Hata Analizi")
    st.markdown(
        "Metrik bir özettir; hatanın neye benzediğini söylemez. Burada her "
        "koşunun en sorunlu kareleri sıralanır ve mümkün olduğunda **sağlıklı "
        "modelin aynı kareyi nasıl gördüğü** yanına konur."
    )

    gorsel_durumu = gorsel_kaynagi()
    if gorsel_durumu == "sunum_seti":
        st.info(
            "Tam görsel arşivi (`reports/`, 233 MB) bu makinede yok; depoyla "
            "birlikte gelen **küçültülmüş sunum seti** kullanılıyor. Her "
            "galeriden dört sıralama ölçütünün ilk 8'i mevcut. Ölçümler "
            "hiçbir zaman bu görsellerden üretilmez."
        )
    elif gorsel_durumu == "yok":
        st.warning(
            "Hiçbir hata galerisi görseli bulunamadı. Taşınabilir seti "
            "üretmek için: `python scripts/sunum_gorselleri_hazirla.py`"
        )

    galeriler = error_galleries()
    if not galeriler:
        st.warning(
            "Henüz hata galerisi üretilmemiş. Üretmek için: "
            "`python scripts/hata_galerisi_toplu.py`"
        )
        return

    adlar = sorted(galeriler)
    a, b, c = st.columns([2, 2, 1])
    with a:
        senaryo = st.selectbox(
            "Koşu", adlar, index=adlar.index("D4") if "D4" in adlar else 0
        )
    with b:
        siralama = st.selectbox("Sıralama", list(SIRALAMA))
    with c:
        adet = st.slider("Gösterilecek örnek", 1, 8, 3)

    galeri = galeriler[senaryo]
    kayitlar = list(galeri["entries"])

    saglikli_galeri = galeriler.get("v00_saglikli") or {}
    saglikli = {e.get("source"): e for e in (saglikli_galeri.get("entries") or [])}

    d, e, f = st.columns(3)
    with d:
        en_az_fn = st.number_input("En az kaçırma (FN)", 0, 100, 0)
    with e:
        en_az_fp = st.number_input("En az fazladan kutu (FP)", 0, 100, 0)
    with f:
        kaynaklar = sorted({
            (_gt_bilgisi(k.get("source", "")) or {}).get("kaynak", "bilinmeyen")
            for k in kayitlar[:60]
        })
        kaynak = st.selectbox("Kaynak", ["hepsi"] + [k for k in kaynaklar if k])

    anahtar = SIRALAMA[siralama]
    ters = anahtar != "mean_iou"          # dusuk IoU'da kucukten buyuge
    sirali = sorted(
        [k for k in kayitlar if anahtar in k],
        key=lambda k: k[anahtar], reverse=ters,
    )
    sirali = [
        k for k in sirali
        if (k.get("false_negatives") or 0) >= en_az_fn
        and (k.get("false_positives") or 0) >= en_az_fp
    ]
    if kaynak != "hepsi":
        sirali = [
            k for k in sirali
            if (_gt_bilgisi(k.get("source", "")) or {}).get("kaynak") == kaynak
        ]

    if not sirali:
        st.warning("Bu filtrelerle kayıt bulunamadı; filtreleri gevşetin.")
        return

    st.markdown("### Özet")
    st.dataframe(
        pd.DataFrame([{
            "görüntü": k.get("source", k.get("image", "")).split("/")[-1][:44],
            "yanlış negatif": k.get("false_negatives"),
            "yanlış pozitif": k.get("false_positives"),
            "ortalama IoU": round(k["mean_iou"], 3) if k.get("mean_iou") else None,
            "skor": round(k["score"], 2) if k.get("score") else None,
            "sağlıklı modelde de var": (
                "evet" if k.get("source") in saglikli else "hayır"
            ),
        } for k in sirali[:int(adet)]]),
        hide_index=True, width="stretch",
    )
    stil.yorum(
        "Skor, yanlış negatif ve yanlış pozitif sayılarıyla düşük IoU'yu "
        "birleştiren sıralama ölçütüdür; en sorunlu kareleri öne çıkarır."
    )

    st.markdown("### Örnekler")
    stil.kutu(
        "<b>Görsel renk sözleşmesi:</b> "
        f'<span style="color:#00d200">yeşil = gerçek etiket (GT)</span> · '
        f'<span style="color:#e61e1e">kırmızı = modelin tahmini</span>. '
        "Yeşil bir kutunun yanında kırmızı yoksa o nesne <b>kaçırılmış</b>; "
        "kırmızı bir kutunun yanında yeşil yoksa <b>fazladan üretilmiş</b> "
        "demektir. Görseller önceden çizildiği için ayrı bir TP/FP/FN renk "
        "ayrımı taşımıyor — sayısal ayrım yukarıdaki tabloda."
    )

    klasor = galeri["folder"]
    saglikli_klasor = saglikli_galeri.get("folder")
    for kayit in sirali[: int(adet)]:
        yol = kayit.get("image")
        if not yol:
            continue
        tam = gorsel_coz(klasor / yol) if klasor else None
        eslesen = saglikli.get(kayit.get("source"))
        st.markdown("---")
        if eslesen and saglikli_klasor:
            a, b = st.columns(2)
            with a:
                stil.ust_baslik("sağlıklı referans modeli")
                s_yol = gorsel_coz(saglikli_klasor / eslesen["image"])
                if s_yol is not None:
                    st.image(str(s_yol), width="stretch")
                else:
                    st.info("Sağlıklı modelin bu karesi bulunamadı.")
                stil.yorum(
                    f"FN {eslesen.get('false_negatives')} · "
                    f"FP {eslesen.get('false_positives')} · "
                    f"IoU {eslesen.get('mean_iou', 0):.2f}"
                )
            with b:
                stil.ust_baslik(f"{senaryo} koşusu")
                if tam is not None:
                    st.image(str(tam), width="stretch")
                else:
                    st.warning(f"Görsel dosyası bulunamadı: `{yol}`")
                stil.yorum(
                    f"FN {kayit.get('false_negatives')} · "
                    f"FP {kayit.get('false_positives')} · "
                    f"IoU {kayit.get('mean_iou', 0):.2f}"
                )
        else:
            stil.ust_baslik(f"{senaryo} koşusu")
            if tam is not None:
                st.image(str(tam), width="stretch")
            else:
                st.warning(f"Görsel dosyası bulunamadı: `{yol}`")
            stil.yorum(
                "Bu kare sağlıklı modelin galerisinde yok — galeriler her "
                "koşunun KENDİ en sorunlu kareleriyle üretilir, bu yüzden "
                "listeler her zaman örtüşmez."
            )
        stil.kutu(_aciklama(kayit, _gt_bilgisi(kayit.get("source", "")),
                            eslesen))

    matris = [g for g in images_for(senaryo) if "confusion" in g.name.lower()]
    if matris:
        st.markdown("---")
        st.markdown("### Confusion matrix")
        st.image(str(matris[0]), width="stretch")
        stil.kutu(
            "<b>Metodolojik not.</b> Bu görseli Ultralytics üretir ve bir kez "
            "kendi raporladığı recall ile çelişti (D3b). Bu yüzden projedeki "
            "karışıklık iddiaları bu görselden değil, bağımsız ölçümden "
            "(<code>teshis/degerlendirme/metrikler.py</code>) üretilir. "
            "Görsel burada yalnızca hızlı bir bakış içindir."
        )

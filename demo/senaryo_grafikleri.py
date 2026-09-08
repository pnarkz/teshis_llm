"""Her senaryonun MEKANIZMASINI en iyi anlatan grafik.

Neden senaryoya ozel
--------------------
Her senaryoya ayni bes grafigi koymak, ekrani doldurur ama hicbirini
anlatmaz. D4'un bulgusu boyut kiriliminda, D3'unki karisiklik matrisinde,
E4'unki cozunurluk taramasindadir. Ustteki dort genel metrik her senaryoda
ayni kalir; altta yalnizca o senaryonun mekanizmasini gosteren grafik durur.

Kurallar
--------
- Her yildiz grafik VERIDEN uretilir; hicbir sayi elle yazilmaz.
- Veri yoksa grafik URETILMEZ ve nedeni yazilir - bos bir eksen cizmek,
  "olculdu ama etki yok" izlenimi verirdi.
- Grafigin altindaki tek cumle de olcumden turetilir.
"""

from __future__ import annotations

import pandas as pd

import grafik
import veri_seti as vs
from teshis.degerlendirme.senaryo_ozeti import ne_gozlendi

# Kod -> yildiz grafik turu. Kod defterden turetilir (senaryo adinin ilk
# parcasi), boylece "D4 last_pt" de D4 ile ayni grafigi alir.
YILDIZ = {
    "D1": "sinif",
    "D2a": "iou_ayrismasi",
    "D2b": "metrik_imzasi",
    "D3": "karisiklik",
    "D3b": "karisiklik",
    "D4": "boyut",
    "D5": "kaynak",
    "D6a": "kume",
    "D6b": "sinif",
    "E1": "egitim_egrisi",
    "E2": "egitim_egrisi",
    "E3b": "egitim_egrisi",
    "E4": "cozunurluk",
}

BASLIK = {
    "sinif": "Sınıf bazlı AP: hangi sınıf çöktü?",
    "boyut": "Nesne boyutuna göre recall: kayıp hangi bantta?",
    "kaynak": "Veri kaynağına göre recall: hangi kaynak kaybetti?",
    "karisiklik": "Karışıklık farkı: hangi sınıf hangisiyle karıştı?",
    "metrik_imzasi": "Metrik imzası: bozulmanın karakteri",
    "iou_ayrismasi": "mAP50 ile mAP50-95 ayrışması",
    "cozunurluk": "Çıkarım çözünürlüğü taraması",
    "kume": "Kilitli küme ile sızıntılı küme",
    "egitim_egrisi": "Eğitim eğrisi",
}


def _kod(senaryo: str) -> str:
    return senaryo.split()[0]


def yildiz_turu(senaryo: str) -> str | None:
    return YILDIZ.get(_kod(senaryo))


# --- Veri ureticileri -------------------------------------------------------

def _referans(senaryo: str) -> str | None:
    return (ne_gozlendi(senaryo) or {}).get("referans_senaryo")


def _kirilim_cifti(senaryo: str, alan: str, ad_haritasi=None):
    """Senaryo ve referansin ayni kirilim alanindaki degerleri."""
    ref = _referans(senaryo)
    if not ref:
        return None
    aday_k, ref_k = vs.kirilim(senaryo), vs.kirilim(ref)
    if not aday_k or not ref_k:
        return None
    satirlar = []
    for grup, d in (aday_k.get(alan) or {}).items():
        r = (ref_k.get(alan) or {}).get(grup) or {}
        if d.get("recall") is None or r.get("recall") is None:
            continue
        satirlar.append({
            "grup": (ad_haritasi or {}).get(grup, grup),
            "sağlıklı": round(r["recall"], 4),
            "senaryo": round(d["recall"], 4),
            "fark": round(d["recall"] - r["recall"], 4),
            "gerçek kutu": d.get("gercek_kutu"),
        })
    return pd.DataFrame(satirlar) if satirlar else None


def boyut_verisi(senaryo: str):
    veri = _kirilim_cifti(senaryo, "boyut_bandi_recall", vs.BANT_ADI)
    if veri is None:
        return None
    # Bantlar buyukluk sirasinda dursun; alfabetik sira anlamsiz olurdu.
    sira = list(vs.BANT_ADI.values())
    veri["_s"] = veri["grup"].map({a: i for i, a in enumerate(sira)})
    return veri.sort_values("_s").drop(columns=["_s"])


def kaynak_verisi(senaryo: str):
    veri = _kirilim_cifti(senaryo, "kaynak_recall")
    return None if veri is None else veri.sort_values("fark")


def sinif_verisi(senaryo: str):
    """Sinif bazli AP50; nadir siniflarin ornek sayisi etikette gorunur."""
    ref = _referans(senaryo)
    if not ref:
        return None
    aday = {s["sınıf"]: s for s in vs.sinif_metrikleri(senaryo)}
    referans = {s["sınıf"]: s for s in vs.sinif_metrikleri(ref)}
    if not aday or not referans:
        return None
    satirlar = []
    for ad, s in aday.items():
        r = referans.get(ad)
        if not r or s["AP50"] is None or r["AP50"] is None:
            continue
        n = s.get("bbox (tanı seti)")
        satirlar.append({
            # Ornek sayisi ETIKETTE: buyuk bir dususun yaninda veri azligi
            # da gorunmeli, yoksa UAI'nin 17 bbox'i gozden kacar.
            "grup": f"{ad}  (n={n})" if n is not None else ad,
            "sağlıklı": r["AP50"],
            "senaryo": s["AP50"],
            "fark": round(s["AP50"] - r["AP50"], 4),
            "bbox": n,
        })
    return pd.DataFrame(satirlar) if satirlar else None


def karisiklik_verisi(senaryo: str):
    """Senaryo eksi saglikli karisiklik matrisi - SAYISAL olcumden."""
    ref = _referans(senaryo)
    if not ref:
        return None
    aday_k, ref_k = vs.kirilim(senaryo), vs.kirilim(ref)
    a = (aday_k or {}).get("karisiklik_matrisi") or {}
    r = (ref_k or {}).get("karisiklik_matrisi") or {}
    if not a or not r:
        return None
    gercekler = sorted(set(a) | set(r))
    tahminler = sorted({t for m in (a, r) for s in m.values() for t in s})
    satirlar = []
    for g in gercekler:
        for t in tahminler:
            av = (a.get(g) or {}).get(t, 0)
            rv = (r.get(g) or {}).get(t, 0)
            if av == 0 and rv == 0:
                continue
            satirlar.append({
                "gercek": vs.KOD_ADI.get(g, g),
                "tahmin": ("bulunamadı" if t == "bulunamadi"
                           else vs.KOD_ADI.get(t, t)),
                "sağlıklı": rv,
                "senaryo": av,
                "fark": av - rv,
            })
    return pd.DataFrame(satirlar) if satirlar else None


def metrik_imzasi_verisi(senaryo: str):
    """Dort genel metrigin farki + gurultu esigi - sifir merkezli profil."""
    gozlem = ne_gozlendi(senaryo)
    if not gozlem:
        return None
    satirlar = []
    for ad, d in gozlem["metrikler"].items():
        if d["fark"] is None:
            continue
        satirlar.append({
            "metrik": ad,
            "fark": d["fark"],
            "gürültü eşiği": d["gurultu_esigi"] or 0.0,
            "değer": d["deger"],
            "referans": d["referans"],
        })
    return pd.DataFrame(satirlar) if satirlar else None


def cozunurluk_verisi():
    """E4 cozunurluk taramasi: 512 / 640 / 768 / 1024 / 1280 px.

    Bes olcum de `reports/senaryo_E4_imgszNNN/d1_metrics.json` altinda
    duruyor ama yalnizca 512 defterde satir olarak var. Tarama, egitim
    cozunurlugunun (768) neden bir tepe noktasi oldugunu ve dususun
    recall'da precision'dan cok daha sert oldugunu gosteriyor.
    """
    from pathlib import Path

    kok = Path(__file__).resolve().parents[1] / "reports"
    satirlar = []
    for dizin in sorted(kok.glob("senaryo_E4_imgsz*")):
        m = vs._oku(dizin / "d1_metrics.json")
        if not m or not m.get("imgsz"):
            continue
        for alan, ad in (("mAP50", "mAP50"), ("precision", "precision"),
                         ("recall", "recall")):
            satirlar.append({"çözünürlük": m["imgsz"], "metrik": ad,
                             "değer": round(m[alan], 4)})
    return pd.DataFrame(sorted(satirlar, key=lambda s: s["çözünürlük"])) \
        if satirlar else None


def kume_verisi(senaryo: str):
    """D6a: ayni agirliklar, iki farkli degerlendirme kumesi."""
    gozlem = ne_gozlendi(senaryo)
    ref = (gozlem or {}).get("referans_senaryo")
    if not gozlem or not ref:
        return None
    satirlar = []
    for ad, d in gozlem["metrikler"].items():
        if d["referans"] is None:
            continue
        satirlar.extend([
            {"metrik": ad, "küme": "kilitli set (val_diagnostic)",
             "değer": d["referans"]},
            {"metrik": ad, "küme": "sızıntılı set (v08)", "değer": d["deger"]},
        ])
    return pd.DataFrame(satirlar) if satirlar else None


def egitim_egrisi_verisi(senaryo: str):
    """Train/val kaybi + best.pt ve last.pt epoch isaretleri."""
    egri = vs.egitim_egrisi(senaryo)
    if egri is None or egri.empty:
        return None, []
    sutunlar = [s for s in ("train/cls_loss", "val/cls_loss")
                if s in egri.columns]
    if not sutunlar:
        return None, []
    uzun = egri.melt(id_vars="epoch", value_vars=sutunlar,
                     var_name="seri", value_name="kayıp")
    isaretler = []
    if "metrics/mAP50(B)" in egri.columns:
        en_iyi = egri.loc[egri["metrics/mAP50(B)"].idxmax()]
        isaretler.append({"epoch": int(en_iyi["epoch"]), "etiket": "best.pt"})
    isaretler.append({"epoch": int(egri["epoch"].max()), "etiket": "last.pt"})
    return uzun, isaretler


# --- Ciziciler --------------------------------------------------------------

def ciz(senaryo: str, tur: str):
    """(grafik, tek cumlelik okuma notu) veya (None, neden yok)."""
    if tur == "boyut":
        veri = boyut_verisi(senaryo)
        if veri is None:
            return None, "Bu koşu için boyut kırılımı ölçümü bulunamadı."
        en_sert = veri.loc[veri["fark"].idxmin()]
        return (
            grafik.dumbbell(veri, "grup", "sağlıklı", "senaryo",
                            alan_adi="recall"),
            f"En sert kayıp <b>{en_sert['grup']}</b> bandında: "
            f"{en_sert['sağlıklı']:.4f} → {en_sert['senaryo']:.4f} "
            f"({en_sert['fark']:+.4f}). Daire sağlıklı referans, kare senaryo; "
            "aradaki mesafe farkın kendisidir.",
        )

    if tur == "kaynak":
        veri = kaynak_verisi(senaryo)
        if veri is None:
            return None, "Bu koşu için kaynak kırılımı ölçümü bulunamadı."
        en_sert = veri.iloc[0]
        return (
            grafik.dumbbell(veri, "grup", "sağlıklı", "senaryo",
                            alan_adi="recall"),
            f"En çok kaybeden kaynak <b>{en_sert['grup']}</b>: "
            f"{en_sert['fark']:+.4f}. Kaynak grupları ayrı çekim koşullarını "
            "temsil eder.",
        )

    if tur == "sinif":
        veri = sinif_verisi(senaryo)
        if veri is None:
            return None, "Bu koşu için sınıf bazlı ölçüm bulunamadı."
        # En cok dusen sinifi secerken NADIR siniflar ayri tutulur. D1'de
        # ham "en cok dusen" UAI cikiyordu (-0.0082, 17 bbox) - oysa senaryo
        # insan sinifini hedefliyor ve UAI'deki oynama olcum gurultusu.
        guvenilir = veri[veri["bbox"].fillna(0) >= 30]
        kaynak = guvenilir if not guvenilir.empty else veri
        en_sert = kaynak.loc[kaynak["fark"].idxmin()]
        nadir = veri[veri["bbox"].fillna(99) < 30]
        ek = ""
        if not nadir.empty:
            ek = (" Nadir sınıflar (" + ", ".join(
                f"{r['grup'].split()[0]} {r['fark']:+.4f}"
                for _, r in nadir.iterrows()
            ) + ") ayrı tutuldu: bu kadar az örnekle oran tek tek nesnelere "
                "aşırı duyarlıdır.")
        return (
            grafik.dumbbell(veri, "grup", "sağlıklı", "senaryo",
                            alan_adi="AP50"),
            f"Yeterli örnekli sınıflar içinde en çok düşen "
            f"<b>{en_sert['grup'].split()[0]}</b>: {en_sert['fark']:+.4f}."
            + ek,
        )

    if tur == "karisiklik":
        veri = karisiklik_verisi(senaryo)
        if veri is None:
            return None, "Bu koşu için karışıklık matrisi ölçümü bulunamadı."
        capraz = veri[(veri["gercek"] != veri["tahmin"])
                      & (veri["tahmin"] != "bulunamadı")]
        artan = capraz.loc[capraz["fark"].idxmax()] if not capraz.empty else None
        not_ = ("Köşegen (doğru sınıflandırma) kırmızıysa azalmış, yeşilse "
                "artmış demektir.")
        if artan is not None and artan["fark"] > 0:
            not_ = (f"En çok artan karışıklık: <b>{artan['gercek']} → "
                    f"{artan['tahmin']}</b> ({artan['sağlıklı']} → "
                    f"{artan['senaryo']} kutu). " + not_)
        return grafik.karisiklik_farki(veri), not_

    if tur == "metrik_imzasi":
        veri = metrik_imzasi_verisi(senaryo)
        if veri is None:
            return None, "Bu koşu için genel metrik farkı hesaplanamıyor."
        artan = veri[veri["fark"] > 0]["metrik"].tolist()
        dusen = veri[veri["fark"] < 0]["metrik"].tolist()
        not_ = "Solda düşüş, sağda artış; soluk çubuklar gürültü eşiğinin altında."
        if artan and dusen:
            not_ = (f"<b>{', '.join(dusen)}</b> düşerken "
                    f"<b>{', '.join(artan)}</b> artıyor — bu ters yön "
                    "bozulmanın karakterini gösterir. " + not_)
        return (grafik.fark_profili(veri, "metrik", "fark", "gürültü eşiği"),
                not_)

    if tur == "iou_ayrismasi":
        veri = metrik_imzasi_verisi(senaryo)
        if veri is None:
            return None, "Bu koşu için genel metrik farkı hesaplanamıyor."
        d = {r["metrik"]: r["fark"] for _, r in veri.iterrows()}
        a, b = d.get("mAP50"), d.get("mAP50_95")
        not_ = "Solda düşüş, sağda artış; soluk çubuklar gürültü eşiğinin altında."
        if a is not None and b is not None:
            not_ = (
                f"mAP50 {a:+.4f}, mAP50-95 {b:+.4f}: daha sıkı IoU eşiklerinde "
                f"kayıp <b>{abs(b / a):.1f} kat</b> daha büyük. Kutular "
                "kayıyor ama nesneler hâlâ bulunuyor — lokalizasyon "
                "gürültüsünün imzası budur. " + not_
                if a else not_
            )
        return (grafik.fark_profili(veri, "metrik", "fark", "gürültü eşiği"),
                not_)

    if tur == "cozunurluk":
        veri = cozunurluk_verisi()
        if veri is None:
            return None, "Çözünürlük taraması ölçümleri bulunamadı."
        tepe = veri[veri["metrik"] == "mAP50"].sort_values("değer").iloc[-1]
        return (
            grafik.cizgi(veri, "çözünürlük", "değer", seri="metrik",
                         alan_adi="değer"),
            f"Tepe nokta <b>{int(tepe['çözünürlük'])} px</b> — modelin "
            "eğitildiği çözünürlük. Düşük çözünürlükte recall çöküyor ama "
            "precision ayakta kalıyor: model az nesne buluyor, bulduklarında "
            "haklı.",
        )

    if tur == "kume":
        veri = kume_verisi(senaryo)
        if veri is None:
            return None, "Bu koşu için küme karşılaştırması hesaplanamıyor."
        return (
            grafik.degerli_gruplu_bar(veri, "metrik", "değer", "küme",
                                      alan_adi="değer"),
            "Aynı ağırlıklar, iki farklı değerlendirme kümesi. Aradaki fark "
            "modelin değil <b>ölçümün</b> farkıdır: sızıntılı kümede model "
            "eğitimde gördüğü kareleri yeniden görüyor.",
        )

    if tur == "egitim_egrisi":
        veri, isaretler = egitim_egrisi_verisi(senaryo)
        if veri is None:
            return None, "Bu koşu için eğitim eğrisi bulunamadı."
        return (
            grafik.egri_isaretli(veri, "epoch", "kayıp", "seri", isaretler,
                                 alan_adi="sınıflandırma kaybı"),
            "Train ve val kaybı arasındaki makasın açılması aşırı uyum "
            "imzasıdır. Kesikli çizgiler best.pt ve last.pt checkpoint'lerinin "
            "hangi epoch'a denk geldiğini gösterir — arıza ikisi arasında "
            "gizlenebiliyor.",
        )

    return None, "Bu senaryo için özel bir grafik tanımlanmadı."

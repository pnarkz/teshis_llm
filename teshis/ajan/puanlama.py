"""Ajan cevabini gizli cevap anahtarina karsi puanlayan pilot rubrik.

Bu, bir bilimsel karsilastirma degil (README'de de boyle belirtiliyor);
diagnosis, evidence ve limitations alanlarini esit agirlikla, anahtar
kelime/regex eslesmesiyle puanlar. Mantik onceden yalnizca
``scripts/ajan_puanla.py`` icinde tekrarlanmaz bir sekilde duruyordu;
bu modul onu tek kaynak haline getirir.

Not: Dosyanin ilk halinde bahsedilen T0/T1/T2 katmanli degerlendirme
semasi hicbir yerde tanimlanmamisti; burada icerilmiyor. Ilerde
katmanli bir rubrik gerekiyorsa once o katmanlarin ne olcecegi
netlestirilmelidir.
"""

from __future__ import annotations

import re
from typing import Any

# Senaryo kodu -> cevap anahtarindaki "expected" etiketi. scripts/ajan_paket_hazirla.py
# cevap anahtarini bu tablodan uretir; puanlama da ayni tabloyu okur, boylece
# paket ile puanlama birbirinden ayrisamaz.
SENARYO_BEKLENEN: dict[str, str] = {
    "Baseline": "saglikli_referans",
    # v00: temiz veriyle, senaryolarla ayni protokolde egitilmis saglikli
    # referans. Ajan icin de saglikli referans olarak puanlanir.
    "v00_saglikli": "saglikli_referans",
    "D3b": "tasit_insan_sinif_karisikligi",
    "D4": "kucuk_nesne_sinyal_kaybi",
    "D5": "kaynak_alani_kaymasi",
    "D6b": "tekrar_agirligi",
    "D1": "sinif_yetersizligi",
    "D2a": "lokalizasyon_etiket_gurultusu",
    "D2b": "eksik_etiket",
    # Ayni eksik-etiket bozulmasi, farkli baslangic modeli: beklenen teshis aynidir.
    "D2b final_best": "eksik_etiket",
    "D3": "uap_uai_sinif_karisikligi",
    # E4: veri tertemiz; bozulma cikarim cozunurlugunde. Ajanin kosu
    # listesinde YER ALMAZ (araclar.py imgsz filtresi), ancak results.csv'de
    # bir satiri oldugu icin beklenen teshisi burada tanimli olmalidir -
    # aksi halde paket hazirlayici "beklenen teshis yok" diye durur.
    "E4 imgsz512": "cozunurluk_uyumsuzlugu",
    # E2 olculdu ve underfitting URETMEDI (bkz. BULGULAR.md). Beklenen
    # etiket yine de tanimli: ajana verildigi gun dogru cevap "anlamli
    # degisim yok" olmalidir, "yetersiz egitim" degil.
    "E2": "anlamli_degisim_yok",
    # E1 best.pt asiri uyumu GIZLIYOR (epoch 1 checkpoint'i); dogru cevap
    # "degisim yok". Asiri uyum yalnizca last.pt ve egitim egrisinde gorunur.
    "E1": "anlamli_degisim_yok",
    "E1 last_pt": "asiri_uyum",
    # E3b: 10 kat yuksek lr. Beklenen imza kararsizlik - iki seed arasinda
    # buyuk oynaklik ve gurultulu egitim egrisi.
    "E3b seed42": "egitim_kararsizligi",
    "E3b seed43": "egitim_kararsizligi",
    # C2: hicbir bozulma yok, yalnizca farkli seed. Dogru cevap "degisim yok";
    # baska her cevap ajanin YANLIS POZITIF uretmesi demektir.
    "C2 seed7": "anlamli_degisim_yok",
    # Ek kontrol kosulari: hicbir bozulma yok, yalnizca farkli seed. Bunlar
    # tekrar degil BAGIMSIZ kontrollerdir (farkli modeller, farkli metrikler);
    # istatistiksel olarak ayni girdinin tekrarindan daha degerlidir.
    "C2 seed13": "anlamli_degisim_yok",
    "C2 seed21": "anlamli_degisim_yok",
    # last_pt varyantlari ayri senaryo degildir; ajana verilmezler (bkz.
    # araclar.ajana_uygun_mu). Beklenen teshis, ait olduklari kosununkiyle
    # aynidir - paket hazirlayici her results.csv satiri icin bir karsilik
    # aradigi icin burada tanimli olmalari gerekir.
    "v00_saglikli last_pt": "saglikli_referans",
    "D4 last_pt": "kucuk_nesne_sinyal_kaybi",
    "D5 last_pt": "kaynak_alani_kaymasi",
    "D6b last_pt": "tekrar_agirligi",
}

# beklenen etiket -> (diagnosis alaninda aranan regex, okunabilir etiket).
#
# Anahtar olarak kosu_NN degil beklenen etiket kullanilir: kosu_NN numaralari
# results.csv satir sirasina bagli oldugu icin araya yeni bir kosu eklenirse
# kayar; beklenen etiket ise senaryonun kendisine baglidir ve kaymaz.
# Yeni bir senaryo eklendiginde SENARYO_BEKLENEN'e ve buraya birlikte satir
# eklenmelidir; eksikse teshis_puani (0.0, None) doner, yani "degerlendirilemedi"
# olarak ayirt edilir ve sessizce yanlis puanlanmaz.
ANAHTAR_KALIPLAR: dict[str, tuple[str, str]] = {
    "saglikli_referans": (r"baseline|saglikli|referans|optimal|dengeli|en yuksek", "saglikli referans"),
    "egitim_kararsizligi": (
        r"kararsiz|instabil|instabil|oynak|dalgalan|yuksek.*ogrenme|lr.*yuksek|"
        r"diverg|iraksa|seed.*fark",
        "egitim kararsizligi",
    ),
    "asiri_uyum": (
        r"asiri uyum|overfit|ezber|memoriz|train.*val.*fark|genelleme kayb",
        "asiri uyum",
    ),
    "cozunurluk_uyumsuzlugu": (
        r"cozunurluk|imgsz|resolution|olcek|kucult|downscal|upscal|boyut uyumsuz",
        "cozunurluk uyumsuzlugu",
    ),
    "sinif_yetersizligi": (r"sinif|yetersiz|insan.*(az|dus|kayb)|temsil", "sinif yetersizligi"),
    "lokalizasyon_etiket_gurultusu": (r"konum|lokal|box|iou|yerlestir|konumlandirma|gurultu", "lokalizasyon gurultusu"),
    "eksik_etiket": (r"eksik|etiket|anot|annotation|yanlis pozitif|false positive", "eksik etiket"),
    "uap_uai_sinif_karisikligi": (r"sinif|karis|uap|uai|id hatas|capraz|confusion", "UAP/UAI sinif karisikligi"),
    "tasit_insan_sinif_karisikligi": (r"sinif|karis|tasit|insan|id hatas|capraz|confusion", "tasit/insan sinif karisikligi"),
    "kucuk_nesne_sinyal_kaybi": (r"kucuk|boyut|small|olcek|piksel|cozunurluk|minik", "kucuk nesne sinyal kaybi"),
    "kaynak_alani_kaymasi": (r"kaynak|alan|domain|dagilim|kayma|shift|sensor|cesitlilik", "kaynak/alan kaymasi"),
    "tekrar_agirligi": (r"tekrar|asiri temsil|dengesiz|dagilim|agirlik|baskin|over.?represent|duplicate", "tekrar agirligi"),
    # Bozulmanin tanı setinde anlamli iz birakmadigi kosular icin gecerli cevap.
    "anlamli_degisim_yok": (
        r"stabil|degisim yok|fark yok|anlamli\w*\s*(degil|yok)|iyilesme|korunmus|"
        r"bozulma\s*(tespit\s*edilmedi|bulunamadi|gozlenmedi|saptanmadi)|"
        r"saglikl|referans|bozulma yok|normal",
        "anlamli degisim yok",
    ),
}

# Kendi istatistiksel analizimize gore bozulmanin kilitli tanı setinde ANLAMLI
# bir iz birakmadigi senaryolar (bkz. README "v00 Saglikli Referans" ve
# "D3b Sonucu"). Bu kosularda "anlamli degisim yok" demek DOGRU cevaptir;
# bozulmanin adini soylemek ise eldeki kanitin otesine gecmektir.
#
# Cevap anahtari "hangi bozulmayi uyguladik" bilgisini tutar; bu tablo ise
# "o bozulma kanitta gorunuyor mu" sorusunu ayirir. Ikisi ayni sey degildir
# ve karistirilirsa ajan, dogru cevap verdigi icin cezalandirilir.
TESPIT_EDILEMEYEN: dict[str, str] = {
    "D1": (
        "insan recall -0.0147 (z=-1.22, anlamli degil); hicbir sinifta anlamli "
        "fark yok. Bkz. README 2026-08-26 (3). "
        "Olcum: reports/senaryo_D1/d1_metrics.json (Ultralytics val). "
        "metrikler.py ayni sinifi conf=0.25/IoU=0.5 ile olcer ve farkli bir "
        "deger verir (orada fark 0.0000); iki olcum karistirilmamalidir."
    ),
    "D3b": (
        "capraz sinif hatasi 2 -> 4 kutu; bozulma sogurulmus. "
        "Bkz. README 2026-08-26 (7)."
    ),
    # C2 kontrolu (2026-09-01) ile eklendi. Bu, ajani kayirmak icin degil,
    # D1 ve D3b'ye zaten uygulanan olcutun artik SAYISAL bir tabani oldugu
    # icin: D6b'nin genel metrik farklari olculmus seed gurultusunun icinde.
    # Ajanin gordugu kanit (genel + kirilim metrikleri) bu kosuda bozulmayi
    # gostermiyor; "saglikli" demek o kanitla tutarli bir okuma.
    "D6b": (
        "genel metrik farklari seed gurultusunun icinde: precision -0.0007 ve "
        "recall -0.0012, C2 esigi 0.0184 / 0.0114. mAP50 farki (+0.0033) esigin "
        "yalnizca iki kati. Bkz. docs/BULGULAR.md 'C2 Kontrolu'."
    ),
}


def _normalize(metin: Any) -> str:
    """Teshis metnini eslestirme icin normallestirir.

    Model teshisleri snake_case yaziyor ("bozulma_tespit_edilmedi"), kaliplar
    ise dogal dilde ("bozulma yok"). Alt cizgi ve tireler bosluga cevrilmezse
    dogru cevaplar eslesmeden gecer: gercek bir kosuda ajan D1 icin
    "bozulma_tespit_edilmedi" dedi - bizim analizimize gore DOGRU cevap - ama
    kalip "bozulma yok" aradigi icin sifir aldi.
    """
    return re.sub(r"[_\-]+", " ", str(metin)).lower().strip()


def metin_ozeti(cevap: dict[str, Any]) -> str:
    """Puanlama icin cevabin metin alanlarini kucuk harfli tek dizeye indirger."""
    parcalar = [cevap.get("diagnosis", ""), *cevap.get("evidence", []), *cevap.get("limitations", [])]
    parcalar.append(cevap.get("next_measurement", ""))
    return " ".join(str(parca).lower() for parca in parcalar)


def teshis_puani(beklenen_etiket: str, cevap: dict[str, Any]) -> tuple[float, str | None]:
    """Diagnosis metninin beklenen teshise uyup uymadigini 0/0.5/1 olarak puanlar.

    beklenen_etiket, cevap anahtarindaki "expected" degeridir. ANAHTAR_KALIPLAR'da
    kayit yoksa (henuz kalip yazilmamis yeni bir senaryo) puan hesaplanamaz;
    (0.0, None) doner ve cagiran taraf bunu "degerlendirilemedi" olarak ayirt eder.
    """
    if beklenen_etiket not in ANAHTAR_KALIPLAR:
        return 0.0, None
    kalip, etiket = ANAHTAR_KALIPLAR[beklenen_etiket]
    metin = _normalize(cevap.get("diagnosis", ""))
    if re.search(kalip, metin):
        return 1.0, etiket
    # Eksik etikette model nedeni adlandiramasa bile precision/recall
    # dengesizligini dogru tarif etmisse kismi puan alir. Model Turkce
    # cevapladigi icin karsiliklari da kabul edilir; onceki surum yalnizca
    # Ingilizce terimleri ariyordu ve "hassasiyet kaybi" gibi dogru bir
    # tarifi kaciriyordu.
    if beklenen_etiket == "eksik_etiket":
        kesinlik = re.search(r"precision|hassasiyet|kesinlik", metin)
        duyarlilik = re.search(r"recall|duyarlilik|anma", metin)
        if kesinlik and duyarlilik:
            return 0.5, etiket
    return 0.0, etiket


def kaniti_puanla(cevap: dict[str, Any]) -> float:
    """En az iki kanit ve en az birinde bir rakam var mi diye kontrol eder."""
    kanitlar = cevap.get("evidence", [])
    if not isinstance(kanitlar, list) or len(kanitlar) < 2:
        return 0.0
    return 1.0 if any(re.search(r"\d", str(deger)) for deger in kanitlar) else 0.0


def siniri_puanla(cevap: dict[str, Any]) -> float:
    """Modelin kucuk ornek belirsizligini bir SAYIYLA birlikte belirtip belirtmedigini olcer.

    Onceki surum yalnizca UAP/UAI + (15|17) kalibini kabul ediyordu. Bu, kucuk
    ornek uyarisinin baska bir grup icin gecerli oldugu kosularda (orn. kaynak
    grubu n=106, ya da bir boyut bandi) dogru yazilmis sinirlamalari
    cezalandiriyordu. Sart artik sudur: bir grup adlandirilmali ve yaninda bir
    sayi verilmelidir; yani "az ornek var" demek yetmez, kac oldugu soylenmeli.
    """
    parcalar = [str(oge).lower() for oge in cevap.get("limitations", [])]
    grup_kalibi = re.compile(r"uap|uai|kaynak_[a-z]|bant|bbox|n\s*=|ornek|kutu")
    for parca in parcalar:
        if grup_kalibi.search(parca) and re.search(r"\d", parca):
            return 1.0
    return 0.0


def kosuyu_puanla(kosu_id: str, beklenen: dict[str, Any], cevap: dict[str, Any]) -> dict[str, Any]:
    """Tek bir kosu icin diagnosis/evidence/limitation alt puanlarini ve toplami dondurur.

    Iki teshis puani hesaplanir:

    - ``diagnosis_score``: kati puan. "Hangi bozulmayi uyguladik" sorusuna gore.
    - ``diagnosis_score_tespit``: tespit-farkindalikli puan. Bozulmanin kanitta
      anlamli iz birakmadigi kosularda (bkz. TESPIT_EDILEMEYEN) "anlamli
      degisim yok" cevabi da dogru sayilir.

    Ikisi ayri raporlanir; hangisinin kullanilacagi okuyucunun karari olur.
    """
    beklenen_etiket = str(beklenen.get("expected", ""))
    teshis, etiket = teshis_puani(beklenen_etiket, cevap)
    kanit = kaniti_puanla(cevap)
    sinir = siniri_puanla(cevap)

    senaryo = str(beklenen.get("hidden_role", ""))
    gerekce = TESPIT_EDILEMEYEN.get(senaryo)
    teshis_tespit = teshis
    if gerekce is not None:
        alternatif, _ = teshis_puani("anlamli_degisim_yok", cevap)
        teshis_tespit = max(teshis, alternatif)

    return {
        "run_id": kosu_id,
        "expected": beklenen.get("expected"),
        "expected_label": etiket,
        "diagnosis_score": teshis,
        "diagnosis_score_tespit": teshis_tespit,
        "evidence_score": kanit,
        "limitation_score": sinir,
        "total": round((teshis + kanit + sinir) / 3, 3),
        "total_tespit": round((teshis_tespit + kanit + sinir) / 3, 3),
        "tespit_edilebilir": gerekce is None,
        "tespit_notu": gerekce,
        "model_diagnosis": cevap.get("diagnosis", "missing"),
    }


def paketi_puanla(cevaplar: list[dict[str, Any]], cevap_anahtari: dict[str, Any]) -> dict[str, Any]:
    """Tum kor deneme paketini puanlar; cevap_anahtari={run_id: {"expected": ...}}."""
    id_ile_cevap = {oge.get("run_id"): oge for oge in cevaplar}
    satirlar = [
        kosuyu_puanla(kosu_id, beklenen, id_ile_cevap.get(kosu_id, {}))
        for kosu_id, beklenen in cevap_anahtari.items()
    ]
    return {
        "metric_definition": (
            "diagnosis, evidence and limitations each score 0..1; "
            "this is a pilot rubric, not a scientific benchmark"
        ),
        "mean_score_aciklama": {
            "mean_score": (
                "Kati puan: cevap, UYGULANAN bozulmanin adiyla eslesiyor mu? "
                "Bozulmanin kanitta gorunmedigi kosularda model dogru cevap "
                "verse bile (anlamli degisim yok) sifir alir."
            ),
            "mean_score_tespit": (
                "Tespit-farkindalikli puan: bozulmanin kilitli tanı setinde "
                "anlamli iz birakmadigi kosularda 'anlamli degisim yok' cevabi "
                "da dogru sayilir. Hangi kosularin bu kapsamda oldugu "
                "teshis/ajan/puanlama.py::TESPIT_EDILEMEYEN icinde, projenin "
                "kendi istatistiksel analizine dayanarak listelenir."
            ),
        },
        "runs": satirlar,
        "mean_score": round(sum(satir["total"] for satir in satirlar) / len(satirlar), 3) if satirlar else 0.0,
        "mean_score_tespit": (
            round(sum(satir["total_tespit"] for satir in satirlar) / len(satirlar), 3)
            if satirlar else 0.0
        ),
    }

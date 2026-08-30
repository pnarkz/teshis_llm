"""docs/BULGULAR.md'deki E4 sayilarini HAM olcum dosyalarina karsi dogrular.

Neden ham dosyaya karsi?
------------------------
Rapor JSON'lari okunabilirlik icin 4 haneye yuvarlanir. Belgeye 3 hane
yaziliyor. Ozet JSON'dan okuyup tekrar yuvarlamak *cift yuvarlama* uretir ve
sessizce yanlis rakam verir: UAP recall@512 gercekte 0.24153'tur, dogru 3
haneli hali 0.242'dir; ama 4 haneye yuvarlanmis 0.2415 uzerinden bicimlemek
0.241 verir (ikili gosterimde 0.2415 aslinda 0.24149...'dir).

Bu tam olarak bir kez oldu ve belgeye yanlis rakam girdi. Test, dogrulamanin
her zaman `reports/senaryo_E4_imgszNNN/d1_metrics.json` icindeki tam
hassasiyetli degerlere karsi yapilmasini zorunlu kilar.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BULGULAR = ROOT / "docs/BULGULAR.md"
SINIFLAR = ["tasit", "insan", "UAP", "UAI"]
EGITIM_IMGSZ = 768


def _ham(imgsz: int) -> dict:
    yol = ROOT / f"reports/senaryo_E4_imgsz{imgsz}/d1_metrics.json"
    if not yol.is_file():
        pytest.skip(f"{yol} yok; E4 olcumleri henuz uretilmemis")
    return json.loads(yol.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def e4_bolumu() -> str:
    metin = BULGULAR.read_text(encoding="utf-8")
    if "### E4 Sonucu" not in metin:
        pytest.skip("BULGULAR.md'de E4 bolumu yok")
    return metin[metin.index("### E4 Sonucu"):]


@pytest.fixture(scope="module")
def olcumler() -> list[int]:
    from teshis.egitim.protokol import e_senaryo_ayarlari
    return e_senaryo_ayarlari("E4")["kosu_ayarlari"]["olcum_imgsz"]


def test_tarama_tablosu_ham_olcumlerle_uyusuyor(e4_bolumu, olcumler):
    """Her cozunurluk satirindaki dort metrik ham dosyadan dogrulanir."""
    eksik = []
    for imgsz in olcumler:
        m = _ham(imgsz)
        for ad in ("mAP50", "mAP50_95", "precision", "recall"):
            deger = f"{m[ad]:.3f}"
            if deger not in e4_bolumu:
                eksik.append(f"imgsz={imgsz} {ad}={deger}")
    assert not eksik, f"Belgede bulunamayan E4 degerleri: {eksik}"


def test_map50_farklari_ham_olcumlerden_hesaplaniyor(e4_bolumu, olcumler):
    taban = _ham(EGITIM_IMGSZ)["mAP50"]
    eksik = []
    for imgsz in olcumler:
        if imgsz == EGITIM_IMGSZ:
            continue
        fark = f"{_ham(imgsz)['mAP50'] - taban:.3f}"
        if fark not in e4_bolumu:
            eksik.append(f"imgsz={imgsz} mAP50 farki={fark}")
    assert not eksik, f"Belgede bulunamayan farklar: {eksik}"


def test_sinif_recall_tablosu_ham_olcumlerle_uyusuyor(e4_bolumu, olcumler):
    dusuk = _ham(min(olcumler))
    taban = _ham(EGITIM_IMGSZ)
    eksik = []
    for i, sinif in enumerate(SINIFLAR):
        r1, r2 = taban["class_recall"][i], dusuk["class_recall"][i]
        for etiket, deger in ((f"{sinif} recall@{EGITIM_IMGSZ}", r1),
                              (f"{sinif} recall@{min(olcumler)}", r2),
                              (f"{sinif} fark", r2 - r1)):
            if f"{deger:.3f}" not in e4_bolumu:
                eksik.append(f"{etiket}={deger:.3f}")
    assert not eksik, f"Belgede bulunamayan sinif degerleri: {eksik}"


def test_tepe_egitim_cozunurlugunde(olcumler):
    """Taramanin ic kontrolu: tepe egitim cozunurlugunde olmali.

    Baska yerde cikarsa ya egitim cozunurlugu yanlis kaydedilmis ya olcum
    bozuktur; her iki durumda da E4 anlatisi gecersizdir.
    """
    en_iyi = max(olcumler, key=lambda r: _ham(r)["mAP50"])
    assert en_iyi == EGITIM_IMGSZ, (
        f"mAP50 tepesi {en_iyi}'de, egitim cozunurlugu {EGITIM_IMGSZ}"
    )


def test_precision_recall_ayrisimi_gercekten_var(olcumler):
    """Belgenin manset iddiasi: precision stabil, recall coker.

    Iddia sayilardan bagimsiz yazilmis bir yorum olmamali; olcum destekliyor mu?
    """
    p_araligi = [_ham(r)["precision"] for r in olcumler]
    r_araligi = [_ham(r)["recall"] for r in olcumler]
    p_yayilim = max(p_araligi) - min(p_araligi)
    r_yayilim = max(r_araligi) - min(r_araligi)
    assert r_yayilim > 3 * p_yayilim, (
        f"recall yayilimi ({r_yayilim:.3f}) precision yayiliminin ({p_yayilim:.3f}) "
        "belirgin sekilde uzerinde degil; 'kor eder, yaniltmaz' iddiasi zayiflar"
    )


def test_kucultme_buyutmeden_pahali(olcumler):
    """Asimetri iddiasi: ayni orandaki kucultme, buyutmeden cok daha maliyetli."""
    taban = _ham(EGITIM_IMGSZ)["mAP50"]
    kucuk = [r for r in olcumler if r < EGITIM_IMGSZ]
    buyuk = [r for r in olcumler if r > EGITIM_IMGSZ]
    if not (kucuk and buyuk):
        pytest.skip("tarama tek yonlu")
    en_kotu_kucuk = taban - min(_ham(r)["mAP50"] for r in kucuk)
    en_kotu_buyuk = taban - min(_ham(r)["mAP50"] for r in buyuk)
    assert en_kotu_kucuk > en_kotu_buyuk, (
        f"kucultme kaybi ({en_kotu_kucuk:.3f}) buyutme kaybindan "
        f"({en_kotu_buyuk:.3f}) buyuk degil"
    )

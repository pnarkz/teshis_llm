"""Konsolun veri katmani: gosterilen her sayi kaynagina baglaniyor mu?

Bu proje icin en onemli kural, hicbir metrigin veya sayimin uydurulmamasi.
Buradaki testler konsolun gosterdigi degerleri KAYNAK dosyalarla karsilastirir:
`reports/veri_raporu.json`, `val_diagnostic/manifest.json`, `results.csv`,
kosu `args.yaml` ve kirilim olcumleri.

Ayrica gorsel katmani icin fallback davranisi test edilir: kilitli tani seti
Git disidir, dolayisiyla taze bir klonda galeri bos kalabilir. Sayfa o
durumda sessizce bos kalmamali.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

import gorseller  # noqa: E402
import veri_seti as vs  # noqa: E402


@pytest.fixture(scope="module")
def rapor() -> dict:
    yol = ROOT / "reports/veri_raporu.json"
    if not yol.is_file():
        pytest.skip("veri_raporu.json yok")
    return json.loads(yol.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tani() -> dict:
    yol = ROOT / "val_diagnostic/manifest.json"
    if not yol.is_file():
        pytest.skip("val_diagnostic manifesti yok")
    return json.loads(yol.read_text(encoding="utf-8"))


# --- Veri seti sayilari -----------------------------------------------------

def test_split_dagilimi_rapordan_geliyor(rapor, tani):
    satirlar = {s["bölüm"]: s for s in vs.split_dagilimi()}
    for ad, d in rapor["splitler"].items():
        assert satirlar[ad]["görüntü"] == d["goruntu"], ad
        assert satirlar[ad]["bbox"] == d["bbox"], ad
    kilitli = satirlar["val_diagnostic (kilitli)"]
    assert kilitli["görüntü"] == tani["goruntu_sayisi"]
    assert kilitli["bbox"] == tani["bbox_sayisi"]


def test_tani_seti_sinif_sayilari_bootstrap_ile_ayni(tani):
    """Iki yerde tutulan bir sayi er gec ayrisir; ikisi de ayni olmali."""
    from teshis.degerlendirme.bootstrap import VAL_DIAGNOSTIC_BBOX_N

    assert tani["sinif_bbox"] == VAL_DIAGNOSTIC_BBOX_N


def test_sinif_paylari_yuzde_yuz_ediyor():
    for kapsam in ("toplam", "tani", "train"):
        paylar = [s["pay"] for s in vs.sinif_dagilimi(kapsam)]
        if paylar:
            assert abs(sum(paylar) - 100) < 0.05, kapsam


def test_saglik_uyarilari_yalnizca_sifirdan_farkli(rapor):
    uyarilar = vs.saglik_uyarilari()
    for u in uyarilar:
        assert u["adet"] > 0
        assert rapor["saglik"][u["anahtar"]] == u["adet"]
    sifir_olanlar = {k for k, v in rapor["saglik"].items() if not v}
    assert not sifir_olanlar & {u["anahtar"] for u in uyarilar}


# --- Model kunyesi ----------------------------------------------------------

def test_kunye_kosunun_kendi_args_yaml_dosyasindan_geliyor():
    """Protokolde ne yazdigi degil, egitimin ne ile kostugu onemli."""
    import yaml

    kunye = vs.egitim_ayarlari("v00_saglikli")
    dizin = vs.kosu_dizini("v00_saglikli")
    assert dizin is not None
    args = yaml.safe_load((dizin / "args.yaml").read_text(encoding="utf-8"))
    assert kunye["optimizer"] == args["optimizer"]
    assert kunye["seed"] == args["seed"]
    assert kunye["eğitim çözünürlüğü"] == args["imgsz"]
    assert kunye["erken durdurma sabrı"] == args["patience"]


def test_kunyede_mutlak_yol_gorunmuyor():
    """Sunum ekraninda kullanici yolu gorunmemeli."""
    for deger in vs.egitim_ayarlari("v00_saglikli").values():
        metin = str(deger)
        assert "\\Users\\" not in metin and "/Users/" not in metin, metin


def test_optimizer_auto_uyarisi_veriden_geliyor():
    """`optimizer: auto` lr0'i yok sayar; kunye bunu soylemeli.

    Bu, E3'u sessizce gecersiz kilabilecek bir tuzakti: protokol lr0'i 100
    kat yukselttigini beyan ediyor, Ultralytics auto modda onu yok sayiyor.
    """
    ayar = vs.egitim_ayarlari("v00_saglikli")
    notu = vs.optimizer_notu("v00_saglikli")
    if str(ayar.get("optimizer")).lower() == "auto":
        assert notu and "bağlayıcı değildir" in notu
    else:
        assert notu is None


def test_sinif_metrikleri_olcum_dosyasiyla_ayni():
    m = vs.metrikler("v00_saglikli")
    satirlar = vs.sinif_metrikleri("v00_saglikli")
    assert len(satirlar) == len(m["class_names"])
    for i, ad in enumerate(m["class_names"]):
        satir = satirlar[i]
        assert satir["AP50"] == round(m["class_ap50"][i], 4)
        assert satir["recall"] == round(m["class_recall"][i], 4)


def test_az_ornekli_siniflar_isaretleniyor():
    """UAP/UAI'nin yuksek gorunen skorlari 15-17 bbox uzerinden olculuyor."""
    az = [s for s in vs.sinif_metrikleri("v00_saglikli") if s["az örnek"] == "evet"]
    assert {s["sınıf"] for s in az} == {"UAP", "UAI"}


# --- Gorsel katmani ---------------------------------------------------------

def test_gorsel_kaynagi_her_zaman_bir_sey_soyluyor():
    d = gorseller.durum()
    assert d["kaynak"] in ("tam", "yedek", "yok")
    assert d["açıklama"]


def test_tasinabilir_ornek_seti_depoda_var():
    """val_diagnostic Git disidir; taze bir klonda galeri bos kalmamali."""
    kok = ROOT / "demo/assets/ornekler"
    assert (kok / "images").is_dir(), (
        "Tasinabilir ornek seti yok. Uretmek icin: python demo/gorseller.py"
    )
    gorsel = list((kok / "images").glob("*.jpg"))
    assert len(gorsel) >= 4, gorsel
    for g in gorsel:
        assert (kok / "labels" / f"{g.stem}.txt").is_file(), g.name


def test_tasinabilir_sette_dort_sinif_da_var():
    """Rastgele secim yapilsaydi 12 goruntunun hepsi insan/tasit cikardi."""
    kok = ROOT / "demo/assets/ornekler/labels"
    if not kok.is_dir():
        pytest.skip("tasinabilir set yok")
    siniflar = set()
    for etiket in kok.glob("*.txt"):
        for satir in etiket.read_text(encoding="utf-8").splitlines():
            if satir.split():
                siniflar.add(int(float(satir.split()[0])))
    assert siniflar == {0, 1, 2, 3}, siniflar


def test_kaynak_grubu_olcum_katmaniyla_ayni_kurali_kullaniyor():
    """Galeri filtresi ile kirilim olcumu ayni gruplari uretmeli.

    GERCEK HATA: galeri once kendi kuralini kullaniyordu ("__ oncesi") ve
    `frame_008172_...` dosyalarini "bilinmiyor" sayiyordu; oysa onlar tf2026
    grubuna ait. Filtre ile metrik tablosu birbirini tutmuyordu.
    """
    from teshis.veri.istatistik import kaynak_adi

    katalog = gorseller.katalog()
    if not katalog:
        pytest.skip("gorsel bulunamadi")
    for kayit in katalog[:50]:
        assert kayit["kaynak"] == kaynak_adi(kayit["dosya"])

    d = gorseller.durum()
    if d["kaynak"] == "tam":
        tani = json.loads(
            (ROOT / "val_diagnostic/manifest.json").read_text(encoding="utf-8")
        )
        assert {k["kaynak"] for k in katalog} == set(tani["kaynak_grubu"])


def test_bbox_cizimi_gorsel_uretiyor():
    katalog = gorseller.katalog()
    if not katalog:
        pytest.skip("gorsel bulunamadi")
    pytest.importorskip("PIL")
    kayit = next(k for k in katalog if k["kutular"])
    gorsel = gorseller.kutulu_gorsel(kayit)
    assert gorsel is not None and gorsel.size[0] > 0


def test_defterdeki_her_kosunun_kimligi_var():
    """Senaryo kartlari kimligi olan kosulardan turetilir."""
    from teshis.degerlendirme.karsilastirilabilirlik import kimlik

    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        for satir in csv.DictReader(f):
            assert kimlik(satir["scenario"]) is not None, satir["scenario"]


# --- Taze klon davranisi ----------------------------------------------------

def test_tani_seti_manifest_olmadan_da_dogru_sayilari_veriyor():
    """`val_diagnostic/` Git disidir; taze bir klonda manifest bulunmaz.

    GERCEK HATA: o durumda konsol "Kilitli tanı seti: 0 — 0 bbox" yaziyordu,
    yani setin BOS oldugu izlenimini veriyordu. Sayilar aslinda depoyla
    gelen olcum dosyalarindan yeniden kurulabiliyor.
    """
    yeniden = vs._tani_seti_yeniden_kur()
    assert yeniden, "kunye yeniden kurulamadi"
    assert yeniden["goruntu_sayisi"] == 1056
    assert yeniden["_yeniden_kuruldu"] is True

    from teshis.degerlendirme.bootstrap import VAL_DIAGNOSTIC_BBOX_N

    assert yeniden["bbox_sayisi"] == sum(VAL_DIAGNOSTIC_BBOX_N.values())
    assert yeniden["sinif_bbox"] == VAL_DIAGNOSTIC_BBOX_N


def test_yeniden_kurulan_kunye_manifestle_ayni(tani):
    """Yeniden kurulan degerler gercek manifestle CAKISMAMALI.

    Bu makinede ikisi de var; ayrisirlarsa taze klonda gosterilen sayilar
    burada gosterilenlerden farkli olurdu.
    """
    yeniden = vs._tani_seti_yeniden_kur()
    assert yeniden["goruntu_sayisi"] == tani["goruntu_sayisi"]
    assert yeniden["bbox_sayisi"] == tani["bbox_sayisi"]
    assert yeniden["sinif_bbox"] == tani["sinif_bbox"]
    assert set(yeniden["kaynak_grubu"]) == set(tani["kaynak_grubu"])
    for grup, d in tani["kaynak_grubu"].items():
        assert yeniden["kaynak_grubu"][grup]["goruntu"] == d["goruntu"], grup


def test_bilinmeyen_deger_uydurulmuyor():
    """Kaynak grubu basina bbox hicbir izlenen dosyada tam durmuyor."""
    yeniden = vs._tani_seti_yeniden_kur()
    for grup, d in yeniden["kaynak_grubu"].items():
        assert d["bbox"] is None, f"{grup}: uydurulmus bbox sayisi"


def test_sunum_gorsel_seti_depoda_var():
    """Taze bir klonda Hata Analizi bos kalmamali."""
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "demo"))
    from data_loader import SUNUM_GORSELLERI

    assert SUNUM_GORSELLERI.is_dir(), (
        "Sunum gorsel seti yok. Uretmek icin: "
        "python scripts/sunum_gorselleri_hazirla.py"
    )
    kareler = list(SUNUM_GORSELLERI.rglob("*.jpg"))
    assert len(kareler) > 300, len(kareler)
    # Windows'un 260 karakterlik yol siniri klonu kirmisti.
    for kare in kareler:
        goreli = kare.relative_to(ROOT)
        assert len(str(goreli)) < 120, f"yol cok uzun: {goreli}"


def test_gorsel_cozumleyici_aynayi_buluyor():
    """Orijinal gorsel yoksa ayna kopyasi bulunmali."""
    import json as _json
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "demo"))
    from data_loader import _ayna_yolu

    manifest = ROOT / "reports/hata_galerisi_D4/gallery.json"
    if not manifest.is_file():
        pytest.skip("D4 galerisi yok")
    kayitlar = _json.loads(manifest.read_text(encoding="utf-8"))
    bulunan = sum(
        1 for k in kayitlar
        if k.get("image") and _ayna_yolu(Path("hata_galerisi_D4") / k["image"]).is_file()
    )
    assert bulunan >= 8, f"aynada yalnizca {bulunan} kare var"


def test_kisa_yol_isletim_sisteminden_bagimsiz():
    """Windows'ta uretilmis yollar Linux'ta da dosya adina inmeli.

    Kayitlar Windows'ta uretildigi icin ters bolulu. `Path(...).name` Linux'ta
    boyle bir yolu tek parca sayar ve tam yolu dondururdu; konsol baska bir
    isletim sisteminde acildiginda ekranda KULLANICI ADI dahil tam yol
    goruntuleniyordu.
    """
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    from veri_seti import kisa_yol

    windows = r"C:\Users\ASUS\Desktop\termal_teshis\runs\v00\weights\best.pt"
    assert kisa_yol(windows) == "best.pt"
    assert "Users" not in kisa_yol(windows)
    assert kisa_yol("/home/biri/proje/data/veri.yaml") == "veri.yaml"
    assert kisa_yol("yolov8n.pt") == "yolov8n.pt"     # yol degil, dokunma
    assert kisa_yol(640) == 640                        # sayi, dokunma
    assert kisa_yol(None) is None


def test_egitim_ayarlarinda_mutlak_yol_gorunmuyor():
    """Ekranda gosterilen hicbir ayar tam yol veya kullanici adi tasimamali."""
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    from veri_seti import egitim_ayarlari

    for senaryo in ("v00_saglikli", "D2b", "D4"):
        for alan, deger in egitim_ayarlari(senaryo).items():
            if isinstance(deger, str):
                assert "\\" not in deger, f"{senaryo}/{alan}: {deger}"
                assert not deger.startswith("/"), f"{senaryo}/{alan}: {deger}"
                assert "Users" not in deger, f"{senaryo}/{alan}: {deger}"

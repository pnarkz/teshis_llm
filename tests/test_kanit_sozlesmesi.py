"""Sartname bolum 9 kanit sozlesmesinin (kanit.json) testleri.

Sozlesmenin kurali net: **"Tek basina mAP hicbir teshis icin yeterli kanit
sayilmaz."** Bu yuzden buradaki testler yalnizca "dosya uretiliyor mu" diye
sormaz; dosyanin DOGRU kosuya ait oldugunu ve eksiklerini gizlemedigini de
dogrular.
"""

import csv
import json
from pathlib import Path

import pytest

from teshis.degerlendirme import kanit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def kosular() -> list[str]:
    return kanit.tum_kosular()


def test_her_kosu_icin_kanit_uretilebiliyor(kosular):
    """results.csv'deki her satir icin sozlesme toplanabilmeli."""
    basarisiz = []
    for run_id in kosular:
        try:
            kanit.kanit_uret(run_id)
        except Exception as sorun:  # noqa: BLE001 - hangi hata olursa olsun rapor et
            basarisiz.append(f"{run_id}: {type(sorun).__name__} {sorun}")
    assert not basarisiz, f"Kanit uretilemeyen kosular: {basarisiz}"


def test_kanit_dosyasi_kendi_kosusunun_metriklerini_tasir(kosular):
    """Dosya adi dogru ama icerigi baska kosuya ait olmamali.

    GERCEK HATA: D6a ve E4, v00'un agirliklarini yeniden degerlendirir; ucu de
    results.csv'de ayni weights_path'i gosterir. Ilk surumde kanit dosyasi
    weights_path'ten turetiliyordu, bu yuzden E4'un kaniti v00'un dizinine
    yazildi ve v00'unkini EZDI. v00'un kanit.json'i UAP recall 0.2415
    gosteriyordu - bu E4'un imgsz=512 degeri; v00'un gercek degeri 1.0.
    """
    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        satirlar = {s["run_id"]: s for s in csv.DictReader(f)}

    hatali = []
    for run_id in kosular:
        yol = kanit.kanit_yolu(run_id)
        if not yol.is_file():
            continue
        icerik = json.loads(yol.read_text(encoding="utf-8"))
        if icerik["run_id"] != run_id:
            hatali.append(f"{yol.name} icinde run_id={icerik['run_id']}, beklenen {run_id}")
            continue
        beklenen = round(float(satirlar[run_id]["mAP50"]), 4)
        gercek = icerik["genel_metrikler"]["mAP50"]
        if abs(gercek - beklenen) > 5e-4:
            hatali.append(f"{run_id}: kanit mAP50={gercek}, results.csv={beklenen}")
    assert not hatali, hatali


def test_iki_kosu_ayni_kanit_dosyasini_paylasmaz(kosular):
    """Her kosunun kanit dosyasi benzersiz olmali; biri digerini ezemez."""
    yollar = {}
    for run_id in kosular:
        yol = kanit.kanit_yolu(run_id)
        if yol in yollar:
            pytest.fail(f"{run_id} ve {yollar[yol]} ayni dosyayi yaziyor: {yol}")
        yollar[yol] = run_id


def test_paylasilan_dizinde_yalnizca_egiten_kosu_yazar():
    """Bir dizini birden fazla kosu paylasiyorsa, yalnizca EGITEN sahiplenir.

    Dizini tek basina kullanan bir kosu, duration_min bos olsa bile kendi
    dizinine yazar - orada ezilecek baska bir kanit yoktur (orn. D2a).
    Kisitlama yalnizca paylasilan dizinler icin anlamlidir.
    """
    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        satirlar = list(csv.DictReader(f))

    dizinler: dict[Path, list[dict]] = {}
    for satir in satirlar:
        dizin = ROOT / Path(satir["weights_path"]).parent.parent
        dizinler.setdefault(dizin, []).append(satir)

    paylasilan = {d: v for d, v in dizinler.items() if len(v) > 1}
    if not paylasilan:
        pytest.skip("hicbir dizin paylasilmiyor")

    for dizin, paylasanlar in paylasilan.items():
        yazanlar = [
            s["run_id"] for s in paylasanlar
            if kanit.kanit_yolu(s["run_id"]).parent == dizin
        ]
        assert len(yazanlar) <= 1, (
            f"{dizin.name} dizinine birden fazla kosu kanit yaziyor: {yazanlar}"
        )
        egitenler = [
            s["run_id"] for s in paylasanlar
            if (s["duration_min"] or "0").strip() not in ("", "0")
        ]
        if egitenler:
            assert yazanlar == egitenler[:1], (
                f"{dizin.name}: dizini egiten {egitenler[0]} degil {yazanlar} sahiplenmis"
            )


def test_eksik_maddeler_gizlenmez(kosular):
    """Sozlesme eksikse bu acikca yazilmali; dosya tam gorunmemeli.

    Yarim bir kanit dosyasinin tam gorunmesi, hic olmamasindan tehlikelidir.
    """
    for run_id in kosular:
        icerik = kanit.kanit_uret(run_id)
        durum = icerik["sozlesme_durumu"]
        assert isinstance(durum["tam_mi"], bool)
        if not durum["tam_mi"]:
            assert durum["eksikler"], f"{run_id}: tam degil ama eksik listesi bos"
            for madde in durum["eksikler"]:
                assert ":" in madde, f"{run_id}: eksik maddesi neyin eksik oldugunu soylemiyor"


def test_ogrenme_orani_hem_beyani_hem_gecerli_degeri_tasir(kosular):
    """optimizer=auto lr0'i yok sayar; yalnizca beyani yazmak yaniltici olur."""
    for run_id in kosular:
        yapilandirma = kanit.kanit_uret(run_id)["yapilandirma"]
        assert "lr0_beyan_edilen" in yapilandirma
        assert "lr0_gecerli" in yapilandirma
        assert yapilandirma["lr0_gecerli"] == kanit.AUTO_OPTIMIZER_LR
        assert "UYGULANMADI" in yapilandirma["lr0_notu"]


def test_guven_araligi_yontemi_her_zaman_etiketli(kosular):
    """Wilson mu, sartnameye uygun bootstrap mu - okuyan kisi bilmeli."""
    for run_id in kosular:
        for sinif in kanit.kanit_uret(run_id)["sinif_metrikleri"]:
            if "recall_ga" not in sinif:
                continue
            yontem = sinif.get("ga_yontemi", "")
            assert yontem, f"{run_id}/{sinif['sinif']}: GA var ama yontemi yazmiyor"
            assert ("bootstrap" in yontem) or ("wilson" in yontem.lower())


def test_sinif_metrikleri_bbox_sayisi_tasir(kosular):
    """Sartname: sinif metrikleri 'her biri bbox sayisi ve GA ile' verilmeli."""
    for run_id in kosular:
        for sinif in kanit.kanit_uret(run_id)["sinif_metrikleri"]:
            assert sinif["bbox_n"], f"{run_id}/{sinif['sinif']}: bbox_n yok"
            assert "f1" in sinif


def test_kanit_yalnizca_map_ile_yetinmiyor(kosular):
    """Sozlesmenin kendi kurali: mAP tek basina yeterli kanit degil."""
    for run_id in kosular:
        icerik = kanit.kanit_uret(run_id)
        assert icerik["sinif_metrikleri"], run_id
        assert icerik["kural"].startswith("Tek basina mAP")
        # En az bir kirilim veya egitim egrisi bulunmali
        assert any(
            anahtar in icerik
            for anahtar in ("boyut_bandi_recall", "kaynak_recall", "egitim_egrisi")
        ), f"{run_id}: mAP ve sinif metrikleri disinda hicbir kanit yok"


def test_rapor_klasoru_kurali_tek_kaynaktan_gelir():
    """demo ve kanit ayni kurali kullanmali; iki kopya kacinilmaz olarak ayrilir.

    GERCEK HATA: kural iki yerde ayri yaziliydi. C2 kontrol kosusu
    eklendiginde demo onu `kontrol_C2_seed7` olarak buluyordu, kanit uretici
    `senaryo_C2_seed7` ariyor ve bulamiyordu.
    """
    import sys

    from teshis.degerlendirme import kanit as kanit_modulu
    from teshis.degerlendirme.raporlar import klasor_adi

    sys.path.insert(0, str(ROOT / "demo"))
    from data_loader import rapor_klasoru as demo_klasor

    kaynak = (ROOT / "teshis/degerlendirme/kanit.py").read_text(encoding="utf-8")
    assert "OZEL_KLASOR" not in kaynak, "kanit.py kendi klasor haritasini tutuyor"
    assert "raporlar import" in kaynak

    for senaryo in ("D4", "E1 last_pt", "C2 seed7", "v00_saglikli", "D2b final_best"):
        beklenen = klasor_adi(senaryo)
        demo_yol = demo_klasor(senaryo)
        assert demo_yol is not None and demo_yol.name == beklenen, senaryo
        assert kanit_modulu.rapor_klasoru(senaryo).name == beklenen, senaryo


def test_kontrol_kosullari_kontrol_onekini_alir():
    """Sartname bolum 8 kontrol kosullari senaryo klasoru gibi adlandirilmamali."""
    from teshis.degerlendirme.raporlar import klasor_adi

    for kod in ("C1", "C2 seed7", "C3 bootstrap"):
        assert klasor_adi(kod).startswith("kontrol_"), kod
    for kod in ("D1", "E1", "E4 imgsz512"):
        assert klasor_adi(kod).startswith("senaryo_"), kod


def test_nokta_tahmin_kendi_guven_araliginin_icinde(kosular):
    """Bir guven araligi, ait oldugu olcumu ICERMEK zorundadir.

    GERCEK HATA: kanit.py, Ultralytics'in recall'unu (d1_sonuc.py, kendi conf
    esigi) metrikler.py'nin araligiyla (conf=0.25, IoU=0.5 eslestirme)
    eslestiriyordu. Iki farkli olcum oldugu icin nokta tahmin araligin
    DISINDA kaliyordu: tasit recall 0.8568, aralik [0.8708, 0.9088].

    Bu test, hangi olcumun hangi araliga ait oldugunu `ga_hangi_olcum`
    alanindan okur ve tutarliligi dogrular.
    """
    hatali = []
    for run_id in kosular:
        for sinif in kanit.kanit_uret(run_id)["sinif_metrikleri"]:
            if "recall_ga" not in sinif:
                continue
            hangi = sinif.get("ga_hangi_olcum")
            assert hangi, f"{run_id}/{sinif['sinif']}: GA var ama hangi olcume ait yazmiyor"
            assert hangi in sinif, f"{run_id}/{sinif['sinif']}: '{hangi}' alani yok"
            nokta, (alt, ust) = sinif[hangi], sinif["recall_ga"]
            if not (alt <= nokta <= ust):
                hatali.append(
                    f"{run_id}/{sinif['sinif']}: {hangi}={nokta} "
                    f"araligin [{alt}, {ust}] disinda"
                )
    assert not hatali, hatali


def test_iki_recall_olcumu_ayri_alanlarda():
    """Farkli esiklerle olculen iki recall ayni alanda toplanmamali."""
    kaynak = (ROOT / "teshis/degerlendirme/kanit.py").read_text(encoding="utf-8")
    assert "recall_kirilim" in kaynak
    assert "ga_hangi_olcum" in kaynak


def test_galeri_adi_tek_kaynaktan_gelir():
    """Galeri klasoru adi iki yerde ayri yazilmamali.

    GERCEK HATA: toplu uretici boslugu alt cizgiye ceviriyordu, kanit uretici
    cevirmiyordu. 10 kosunun galerisi uretildigi halde "eksik" gorunuyordu
    (orn. 'C2 seed21' -> 'hata_galerisi_C2_seed21' vs '...C2 seed21').
    """
    import importlib.util

    from teshis.degerlendirme.raporlar import galeri_adi

    assert galeri_adi("C2 seed21") == "hata_galerisi_C2_seed21"
    assert galeri_adi("D4") == "hata_galerisi_D4"

    kaynak = (ROOT / "teshis/degerlendirme/kanit.py").read_text(encoding="utf-8")
    assert 'f"reports/hata_galerisi_{senaryo}' not in kaynak, (
        "kanit.py galeri adini kendi kuruyor; raporlar.galeri_adi kullanilmali"
    )

    spec = importlib.util.spec_from_file_location(
        "galeri_toplu", ROOT / "scripts/hata_galerisi_toplu.py"
    )
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    for senaryo in ("C2 seed21", "D2b final_best", "E1 last_pt", "D4"):
        assert modul.galeri_yolu(senaryo).name == galeri_adi(senaryo), senaryo


def test_kanit_sozlesmesi_tam(kosular):
    """Her kosu sartname bolum 9'daki tum maddeleri karsilamali.

    Bu test bilerek KATI: sozlesme bir kez tam hale geldikten sonra, yeni bir
    kosu eklendiginde kirilim veya galeri unutulursa hemen gorunur olmali.
    Gecici bir eksik varsa DEFTER_DISI benzeri bir gerekce mekanizmasi degil,
    olcumun kendisi uretilmelidir.
    """
    eksikler = {}
    for run_id in kosular:
        durum = kanit.kanit_uret(run_id)["sozlesme_durumu"]
        if not durum["tam_mi"]:
            eksikler[run_id] = durum["eksikler"]
    assert not eksikler, f"Sozlesmeyi tam karsilamayan kosular: {eksikler}"

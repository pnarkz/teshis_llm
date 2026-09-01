"""results.csv kayit defterinin degismezleri.

Defter uc bilesenin ortak kaynagidir: demo, ajan arac katmani ve kanit
sozlesmesi hepsi buradan beslenir. Bu yuzden defterdeki bir tutarsizlik
sessizce ucune birden yayilir.
"""

import csv
from pathlib import Path

import pytest

from teshis.degerlendirme import kayit_defteri as kd

ROOT = Path(__file__).resolve().parents[1]


def test_sutun_duzeni_beklenen_gibi():
    assert list(kd.oku()[0]) == kd.SUTUNLAR


def test_run_id_ve_senaryo_adlari_benzersiz():
    """Iki kosu ayni adi tasirsa demo ve kanit ureticisi ayirt edemez.

    GERCEK HATA: d2b_20260820_main ve d2b_20260820_final ikisi de 'D2b' idi;
    demo bunu run_id'ye gore elle yamaliyordu, kanit ureticisi yanlis rapor
    klasorunu buluyordu.
    """
    satirlar = kd.oku()
    for alan in ("run_id", "scenario"):
        degerler = [s[alan] for s in satirlar]
        tekrar = {d for d in degerler if degerler.count(d) > 1}
        assert not tekrar, f"{alan} tekrar ediyor: {sorted(tekrar)}"


def test_uretilmis_her_olcumun_defterde_karsiligi_var():
    """Olcum uretilip deftere yazilmazsa demo ve ajan onu goremez.

    Bilincli istisnalar kd.DEFTER_DISI altinda GEREKCESIYLE yazilir.
    """
    eksik = kd.kayitsiz_olcumler()
    assert not eksik, (
        f"Bu olcumler uretilmis ama defterde satiri yok: {eksik}. "
        "Ya satir ekleyin ya da DEFTER_DISI'na gerekcesiyle yazin."
    )


def test_defter_disi_istisnalar_gerekce_tasir():
    for klasor, gerekce in kd.DEFTER_DISI.items():
        assert len(gerekce) > 40, f"{klasor}: gerekce cok kisa veya bos"


def test_her_satirin_olcum_dosyasi_mevcut_ve_uyusuyor():
    """Defterdeki sayi ile raporun sayisi ayrisamamali."""
    import json

    from teshis.degerlendirme.raporlar import rapor_klasoru

    hatali = []
    for s in kd.oku():
        klasor = rapor_klasoru(s["scenario"])
        if klasor is None:
            hatali.append(f"{s['scenario']}: rapor klasoru bulunamadi")
            continue
        yol = klasor / "d1_metrics.json"
        if not yol.is_file():
            hatali.append(f"{s['scenario']}: {yol.name} yok")
            continue
        m = json.loads(yol.read_text(encoding="utf-8"))
        if abs(m["mAP50"] - float(s["mAP50"])) > 5e-7:
            hatali.append(
                f"{s['scenario']}: defter {s['mAP50']}, olcum {m['mAP50']:.7f}"
            )
    assert not hatali, hatali


def test_ayni_senaryo_adi_ikinci_kez_eklenemez():
    mevcut = kd.oku()
    satir = dict(mevcut[0])
    satir["run_id"] = "yepyeni_run_id"
    with pytest.raises(ValueError, match="scenario zaten defterde"):
        kd.dogrula(satir, mevcut)


def test_metrikler_elle_verilemez():
    """satir_uret metrikleri olcum dosyasindan okur; imzasinda metrik yok."""
    import inspect

    imza = inspect.signature(kd.satir_uret).parameters
    for yasak in ("mAP50", "precision", "recall", "AP_tasit"):
        assert yasak not in imza, (
            f"satir_uret metrikleri parametre olarak aliyor ({yasak}); "
            "bu, defter ile raporun ayrismasina izin verir"
        )
    assert "metrik_json" in imza


def test_eksik_olcum_dosyasi_reddedilir(tmp_path):
    with pytest.raises(FileNotFoundError):
        kd.satir_uret(
            run_id="x", scenario="X", metrik_json=tmp_path / "yok.json",
            weights_path="w", data_version="v", seed=1, epochs=1,
        )

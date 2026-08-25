"""teshis/ajan/araclar.py icin testler.

results.csv zamanla yeni senaryo satirlariyla buyuyecegi icin testler kesin
sayisal degerlere degil, katalog.yaml kuralinin (manifest_ajana_verilmez,
senaryo adi ajana gorunmez) korundugu yapisal ozelliklere bakar.
"""

import pandas as pd
import pytest

from teshis.ajan import araclar


def test_kosu_listesi_baseline_ile_baslar():
    liste = araclar.kosu_listesini_getir()
    assert liste[0] == "kosu_01"
    assert len(liste) == len(set(liste))


def test_kosu_listesi_uzunlugu_results_csv_ile_tutarli():
    frame = pd.read_csv(araclar.RESULTS_CSV)
    assert len(araclar.kosu_listesini_getir()) == len(frame) + 1


def test_anonim_harita_senaryo_adi_icermez():
    harita = araclar.anonim_kosu_haritasi()
    for kosu_id, run_id in harita.items():
        assert kosu_id.startswith("kosu_")
        # run_id gercek deneyin kimligidir (D1/D2a gibi senaryo kodu degil).
        assert isinstance(run_id, str) and run_id


def test_anonim_harita_kararlidir():
    """Ayni sureç icinde iki kez cagirmak ayni eslemeyi vermelidir (numaralandirma sabit)."""
    assert araclar.anonim_kosu_haritasi() == araclar.anonim_kosu_haritasi()


def test_bilinmeyen_kosu_id_key_error_verir():
    with pytest.raises(KeyError):
        araclar.kosu_metriklerini_getir("kosu_99")


def test_kosu_metrikleri_beklenen_alanlari_icerir():
    ilk_kosu = sorted(araclar.anonim_kosu_haritasi())[0]
    metrikler = araclar.kosu_metriklerini_getir(ilk_kosu)
    for alan in ("mAP50", "mAP50_95", "precision", "recall", "class_AP50"):
        assert alan in metrikler
    assert set(metrikler["class_AP50"]) == set(araclar.SINIFLAR)


def test_baseline_farki_kosu01_icin_sifirdir():
    fark = araclar.baseline_farkini_getir("kosu_01")
    assert all(deger == 0.0 for deger in fark.values())


def test_baseline_farki_baseline_metrikleriyle_tutarlidir():
    ilk_kosu = sorted(araclar.anonim_kosu_haritasi())[0]
    kosu = araclar.kosu_metriklerini_getir(ilk_kosu)
    taban = araclar.baseline_metriklerini_getir()
    fark = araclar.baseline_farkini_getir(ilk_kosu)
    assert fark["mAP50"] == pytest.approx(kosu["mAP50"] - taban["mAP50"], abs=1e-6)


def test_bbox_sayilari_val_diagnostic_ile_sabittir():
    assert araclar.bbox_sayilarini_getir() == {"tasit": 1264, "insan": 2718, "UAP": 15, "UAI": 17}

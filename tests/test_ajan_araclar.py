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
    """Ajan, saglikli referans + her senaryo kosusunu gorur.

    v00 referans kosusu results.csv'de bir satirdir ama ajana ayri bir senaryo
    olarak sunulmaz; kosu_01 olarak karsilastirma tabani rolunu ustlenir.
    """
    frame = pd.read_csv(araclar.RESULTS_CSV)
    senaryo_sayisi = (
        (frame["scenario"] != araclar.REFERANS_SENARYO)
        & (frame["evaluation_set"] == araclar.KILITLI_DEGERLENDIRME_SETI)
    ).sum()
    assert len(araclar.kosu_listesini_getir()) == senaryo_sayisi + 1


def test_referans_kosusu_ajana_senaryo_olarak_sunulmaz():
    """v00 anonim haritada bulunmamali; ajanin teshis edecegi bir senaryo degil."""
    frame = pd.read_csv(araclar.RESULTS_CSV).set_index("run_id")
    for run_id in araclar.anonim_kosu_haritasi().values():
        assert frame.loc[run_id, "scenario"] != araclar.REFERANS_SENARYO


def test_baseline_saglikli_referans_kosusundan_gelir():
    """kosu_01 metrikleri, protokolle egitilmis v00 kosusuyla birebir ayni olmali."""
    frame = pd.read_csv(araclar.RESULTS_CSV)
    v00 = frame[frame["scenario"] == araclar.REFERANS_SENARYO].iloc[0]
    taban = araclar.baseline_metriklerini_getir()
    for alan in ("mAP50", "mAP50_95", "precision", "recall"):
        assert abs(taban[alan] - float(v00[alan])) < 1e-9


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


def test_farkli_degerlendirme_setindeki_kosu_ajana_verilmez():
    """Ajan yalnizca ayni kilitli kumede olculmus kosulari gormeli.

    D6a'nin metrikleri kasitli olarak SIZINTILI bir kumede olculdu. Ayni
    tabloya konup ajana senaryo olarak sunulursa, ajan farkli tabanlari
    kiyaslar ve sizintinin sismesini "bozulma" sanir.
    """
    frame = pd.read_csv(araclar.RESULTS_CSV).set_index("run_id")
    for run_id in araclar.anonim_kosu_haritasi().values():
        assert frame.loc[run_id, "evaluation_set"] == araclar.KILITLI_DEGERLENDIRME_SETI, (
            f"{run_id} farkli bir degerlendirme kumesinde olculmus; ajana verilmemeli"
        )


def test_kilitli_set_disindaki_satirlar_results_csvde_kalir():
    """Disarida birakma yalnizca ajan katmanindadir; kayit silinmez."""
    frame = pd.read_csv(araclar.RESULTS_CSV)
    disaridakiler = frame[frame["evaluation_set"] != araclar.KILITLI_DEGERLENDIRME_SETI]
    if disaridakiler.empty:
        pytest.skip("farkli kumede olculmus kosu yok")
    # Kayit duruyor ama ajan haritasinda yok.
    harita = set(araclar.anonim_kosu_haritasi().values())
    for run_id in disaridakiler["run_id"]:
        assert run_id not in harita

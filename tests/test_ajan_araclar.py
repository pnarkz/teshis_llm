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


def test_ajanin_gordugu_kosular_uygunluk_yuklemiyle_birebir_ortusur():
    """Ajan, uygun olan HER kosuyu gorur ve uygun olmayan HICBIRINI gormez.

    Uygunluk kosullari araclar.ajana_uygun_mu icinde TEK kaynakta durur; bu
    test onu kullanir, kopyalamaz. Kopyalayan onceki surum uretimle birlikte
    guncellenmedigi icin uc sizintiyi de kaciridi (E4, E2, last_pt).
    """
    frame = araclar._scenario_rows()
    uygun = {
        str(r.run_id) for r in frame.itertuples()
        if araclar.ajana_uygun_mu(frame.loc[r.Index])
    }
    gorunen = set(araclar.anonim_kosu_haritasi().values())
    assert gorunen == uygun, (
        f"ajanda fazla: {sorted(gorunen - uygun)} | "
        f"ajanda eksik: {sorted(uygun - gorunen)}"
    )
    assert araclar.kosu_listesini_getir()[0] == "kosu_01"
    assert len(araclar.kosu_listesini_getir()) == len(uygun) + 1


def test_checkpoint_varyantlari_ajana_verilmez():
    """last_pt satirlari ayri senaryo degil; ayni kosunun baska bir anidir.

    Deftere eklendiklerinde ajanin listesine kosu_12..kosu_15 olarak sizdilar.
    Bu, ayni sinifin ucuncu sizintisiydi (once E4, sonra E2), bu yuzden ad
    listesine madde eklemek yerine yapisal olarak kapatildi.
    """
    frame = araclar._scenario_rows()
    gorunen = set(araclar.anonim_kosu_haritasi().values())
    sizan = [
        str(r.run_id) for r in frame.itertuples()
        if not str(r.weights_path).endswith("best.pt") and str(r.run_id) in gorunen
    ]
    assert not sizan, f"last.pt kosulari ajana verilmis: {sizan}"


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


def test_farkli_taban_modelden_gelen_kosu_ajana_verilmez():
    """yolo26n cifti ajana senaryo olarak sunulmamali.

    Ajan her kosuyu kosu_01 (main_model tabanli v00) ile karsilastirir. Farkli
    bir taban modelden gelen kosuda fark, bozulmadan degil model
    kapasitesinden gelir; ajan bunu "bozulma" sanar.
    """
    frame = pd.read_csv(araclar.RESULTS_CSV).set_index("run_id")
    referans_model = frame.loc[
        frame["scenario"] == araclar.REFERANS_SENARYO, "model"
    ].iloc[0]
    for run_id in araclar.anonim_kosu_haritasi().values():
        senaryo = frame.loc[run_id, "scenario"]
        assert senaryo not in araclar.AJANA_VERILMEYEN, f"{run_id} disarida birakilmaliydi"


def test_ajana_verilmeyen_kayitlari_results_csvde_durur():
    """Disarida birakma yalnizca ajan katmanindadir; deney kaydi silinmez."""
    frame = pd.read_csv(araclar.RESULTS_CSV)
    for senaryo in araclar.AJANA_VERILMEYEN:
        assert (frame["scenario"] == senaryo).any(), f"{senaryo} results.csv'de yok"


def test_ajana_verilmeyen_her_kayit_gerekcelidir():
    """Her disarida birakma yazili bir gerekce tasimali."""
    for senaryo, gerekce in araclar.AJANA_VERILMEYEN.items():
        assert gerekce and len(gerekce) > 15, f"{senaryo} icin gerekce yetersiz"


def test_farkli_cozunurlukte_olculen_kosu_ajana_verilmez():
    """Ayni kumede olculmus olmak yetmez; ayni imgsz'de de olculmus olmali.

    E4, v00 modelinin farkli imgsz degerlerinde yeniden degerlendirilmesidir.
    Veri tertemizdir; fark yalnizca cikarim cozunurlugundan gelir. Kilitli
    kumede olculdugu icin evaluation_set filtresini GECER ve ajanin listesine
    "bozulmus bir kosu" gibi girer. Gercekte bu leaked satir eklendiginde
    kosu_11 olarak goruldu ve ancak elle fark edildi.
    """
    import pandas as pd
    from teshis.ajan import araclar

    frame = araclar._scenario_rows()
    referans_imgsz = araclar._referans_satiri()["imgsz_eval"]
    gorunen = set(araclar.anonim_kosu_haritasi().values())

    farkli = frame[
        (frame["evaluation_set"] == araclar.KILITLI_DEGERLENDIRME_SETI)
        & (frame["imgsz_eval"] != referans_imgsz)
    ]
    sizan = [str(r) for r in farkli["run_id"] if str(r) in gorunen]
    assert not sizan, (
        f"Referanstan farkli imgsz ile olculmus kosular ajana verilmis: {sizan}"
    )


def test_ajana_verilen_kosularin_hepsi_ayni_kume_ve_cozunurlukte():
    """Kume ve cozunurluk yapisal olarak garanti; bunlarda istisna yok."""
    from teshis.ajan import araclar

    frame = araclar._scenario_rows().set_index("run_id")
    referans = araclar._referans_satiri()
    for run_id in araclar.anonim_kosu_haritasi().values():
        satir = frame.loc[run_id]
        assert satir["evaluation_set"] == araclar.KILITLI_DEGERLENDIRME_SETI, run_id
        assert satir["imgsz_eval"] == referans["imgsz_eval"], run_id


def test_taban_model_sapmalari_beyan_edilmis_olanlarla_sinirli():
    """Farkli baslangic modelinden gelen kosu, ancak BEYAN EDILMISSE ajana verilebilir.

    Boyle bir kosuda fark iki degiskenden gelir (bozulma + baslangic modeli);
    ajan bunu tek degiskenli bir bozulma gibi okur. Kusur d2b_20260820_final'da
    gercekten mevcut ve kapatilmak yerine sinirlandirildi (bkz. araclar.py
    BILINEN_TABAN_MODEL_SAPMASI). Bu test, listenin sessizce buyumesini onler.
    """
    from teshis.ajan import araclar

    frame = araclar._scenario_rows().set_index("run_id")
    referans_model = araclar._referans_satiri()["model"]
    beyansiz = [
        run_id for run_id in araclar.anonim_kosu_haritasi().values()
        if frame.loc[run_id]["model"] != referans_model
        and run_id not in araclar.BILINEN_TABAN_MODEL_SAPMASI
    ]
    assert not beyansiz, (
        f"Beyan edilmemis taban model sapmasi: {beyansiz}. Ya kosuyu ajandan "
        "cikarin ya da BILINEN_TABAN_MODEL_SAPMASI'na gerekcesiyle ekleyin."
    )


def test_c2_negatif_kontrolu_ajana_veriliyor():
    """C2 ajanin kosu listesinde OLMALI: yanlis pozitif orani ancak boyle olculur.

    C2'de hicbir bozulma yoktur; v00 ile birebir ayni protokol, yalnizca seed
    farkli. Ajan burada bir "sorun" bulursa bu, uydurulmus bir teshistir. Bu
    kontrol olmadan projenin merkezi iddiasi ("ajan bozulmayi metrik
    imzasindan teshis edebilir") tek yonlu olcumus olur: dogru pozitifler
    sayilir, yanlis pozitifler sayilmaz.
    """
    import csv

    from teshis.ajan import araclar

    with open(araclar.RESULTS_CSV, encoding="utf-8") as f:
        c2 = [s for s in csv.DictReader(f) if s["scenario"].startswith("C2")]
    if not c2:
        pytest.skip("C2 kosusu henuz results.csv'de yok")

    gorunen = set(araclar.anonim_kosu_haritasi().values())
    eksik = [s["run_id"] for s in c2 if s["run_id"] not in gorunen]
    assert not eksik, (
        f"C2 negatif kontrolu ajana verilmiyor: {eksik}. "
        "Bu kosu olmadan ajanin yanlis pozitif orani olculemez."
    )


def test_c2_eklenmesi_mevcut_kosu_numaralarini_kaydirmadi():
    """Yeni kosu SONA eklenmeli; ortaya girmesi tamamlanmis denemeyi gecersiz kilar.

    kosu_NN numaralari results.csv satir sirasina dayanir. C2 satiri sona
    eklendigi icin kosu_02..kosu_10 korunur; bir gun basa veya ortaya bir
    satir eklenirse bu test kirilir ve deneme kayitlarinin gecersizlestigi
    fark edilir.
    """
    from teshis.ajan import araclar

    harita = araclar.anonim_kosu_haritasi()
    beklenen = {
        "kosu_02": "d1_v2_20260825",
        "kosu_03": "d2a_20260820",
        "kosu_04": "d2b_20260820_main",
        "kosu_05": "d2b_20260820_final",
        "kosu_06": "d3_v2_20260826",
        "kosu_07": "d3b_20260826",
        "kosu_08": "d4_20260826",
        "kosu_09": "d5_20260826",
        "kosu_10": "d6b_20260828",
    }
    kayan = {k: (v, harita.get(k)) for k, v in beklenen.items() if harita.get(k) != v}
    assert not kayan, (
        f"Tamamlanmis denemenin kosu numaralari kaymis: {kayan}. "
        "reports/ajan_denemesi/ altindaki cevaplar artik gecersizdir."
    )


def test_c2_beklenen_cevabi_degisim_yok():
    """C2'nin dogru cevabi 'anlamli degisim yok'; baska her cevap yanlis pozitiftir."""
    from teshis.ajan.puanlama import SENARYO_BEKLENEN

    assert SENARYO_BEKLENEN["C2 seed7"] == "anlamli_degisim_yok"


def test_protokolden_sapan_kosular_ajana_verilmez():
    """E serisi ortak protokolu bozar; ajanin varsayimini gecersiz kilar.

    Ad listesine E senaryolarini tek tek eklemek BES kez geride kaldi
    (E4, E2, last_pt, E1, E3b). Filtre artik kosunun KENDI manifestini okur:
    manifest e_senaryo veya protokol_sapmalari tasiyorsa kosu ajana verilmez.
    Yeni bir E senaryosu eklendiginde liste kendiliginden guncellenir.
    """
    import json
    from pathlib import Path as _Path

    from teshis.ajan import araclar

    frame = araclar._scenario_rows()
    gorunen = set(araclar.anonim_kosu_haritasi().values())
    sizan = []
    for r in frame.itertuples():
        yol = (araclar.ROOT / _Path(str(r.weights_path)).parent.parent
               / "run_manifest.json")
        if not yol.is_file():
            continue
        m = json.loads(yol.read_text(encoding="utf-8"))
        if (m.get("e_senaryo") or m.get("protokol_sapmalari")) and str(r.run_id) in gorunen:
            sizan.append(str(r.run_id))
    assert not sizan, f"Protokolden sapan kosular ajana verilmis: {sizan}"


def test_ajan_kosu_numaralari_hala_ayni():
    """Tamamlanmis denemenin kosu_NN eslemesi korunmali."""
    from teshis.ajan import araclar

    harita = araclar.anonim_kosu_haritasi()
    for kosu, beklenen in (
        ("kosu_02", "d1_v2_20260825"),
        ("kosu_08", "d4_20260826"),
        ("kosu_11", "c2_seed7_20260901"),
    ):
        assert harita.get(kosu) == beklenen, (
            f"{kosu} kaymis: {harita.get(kosu)} (beklenen {beklenen})"
        )

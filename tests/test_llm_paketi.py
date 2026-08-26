"""LLM deneme paketi ile ajan araclari arasindaki tutarlilik sozlesmeleri.

Bu testler, dogrulama sirasinda bulunan gercek bir bosluktan dogdu: paket
ureticisi dort kosuyu dosya yollariyla hardcode ediyordu, bu yuzden D2b
final_best ve D3 eklendiginde ajan araclari 6 kosu sunarken cevap anahtari
4 kosuda kalmisti; yeni kosular sessizce puanlanamaz durumdaydi.
"""

import json
from pathlib import Path

import pytest

from teshis.ajan import araclar
from teshis.ajan.puanlama import ANAHTAR_KALIPLAR, SENARYO_BEKLENEN

ROOT = Path(__file__).resolve().parents[1]
ANSWER_KEY = ROOT / "reports/llm_trial/answer_key.json"
LLM_INPUT = ROOT / "reports/llm_trial/llm_input.json"


def test_her_senaryonun_beklenen_teshisi_tanimli():
    """results.csv'deki her senaryo icin SENARYO_BEKLENEN kaydi olmalidir."""
    import pandas as pd

    scenarios = set(pd.read_csv(araclar.RESULTS_CSV)["scenario"])
    # results.csv'de iki D2b kosusu ayni senaryo adini tasir; demo/loader
    # bunlardan birini "D2b final_best" olarak yeniden adlandirir.
    scenarios |= {"Baseline", "D2b final_best"}
    eksik = sorted(scenarios - set(SENARYO_BEKLENEN))
    assert not eksik, (
        f"Bu senaryolar icin beklenen teshis tanimli degil: {eksik}. "
        "teshis/ajan/puanlama.py::SENARYO_BEKLENEN ve ANAHTAR_KALIPLAR'a ekleyin."
    )


def test_beklenen_teshislerin_hepsi_puanlanabilir():
    eksik = sorted(set(SENARYO_BEKLENEN.values()) - set(ANAHTAR_KALIPLAR))
    assert not eksik, f"Bu beklenen teshisler icin regex kalibi yok: {eksik}"


@pytest.mark.skipif(not ANSWER_KEY.is_file(), reason="cevap anahtari yok")
def test_cevap_anahtari_anonim_haritayla_uyumlu():
    """Kayitli answer_key.json hala mevcut kosu_NN eslemesiyle ayni senaryolari gosteriyor mu?

    kosu_NN numaralari results.csv satir sirasindan turetildigi icin araya bir
    satir eklenirse kayar. Bu test, kayitli cevap anahtarinin sessizce yanlis
    senaryoyu puanlar hale gelmesini yakalar.
    """
    import pandas as pd

    key = json.loads(ANSWER_KEY.read_text(encoding="utf-8"))
    frame = pd.read_csv(araclar.RESULTS_CSV).set_index("run_id")
    mapping = araclar.anonim_kosu_haritasi()

    for kosu_id, entry in key.items():
        scenario = "Baseline" if kosu_id == "kosu_01" else str(frame.loc[mapping[kosu_id], "scenario"])
        beklenen = SENARYO_BEKLENEN[scenario]
        assert entry["expected"] == beklenen, (
            f"{kosu_id} cevap anahtarinda '{entry['expected']}' diyor ama guncel esleme "
            f"{scenario} -> '{beklenen}' veriyor. Paket kaymis; "
            "scripts/prepare_llm_trial.py --force ile yeniden uretin ve LLM denemesini tekrarlayin."
        )


@pytest.mark.skipif(not LLM_INPUT.is_file(), reason="paket yok")
def test_paket_ajana_senaryo_adi_sizdirmaz():
    """Kosu verilerinde senaryo kodu, veri surumu veya gercek kaynak adi gecmemelidir.

    Tarama yalnizca `runs` bolumunde yapilir: paketin ust duzey gorev tanimi
    ("Termal drone YOLO diagnostigi") alanin kendisini anlatir ve `termal`
    kelimesi orada mesru olarak gecer. Kaynak adlari ise kosu verilerinde
    kaynak_a/kaynak_b diye anonimlestirilmis olmalidir.
    """
    packet = json.loads(LLM_INPUT.read_text(encoding="utf-8"))
    blob = json.dumps(packet["runs"], ensure_ascii=False).lower()
    yasakli = [
        "sinif_yetersizligi", "eksik_etiket", "lokalizasyon_etiket_gurultusu",
        "uap_uai_sinif_karisikligi", "kucuk_nesne", "kaynak_alani", "saglikli",
        "manifest", "v00_", "v01_", "v02_", "v03_", "v04_", "v05_", "v06_", "v07_",
        "run_20", "best.pt",
        # gercek kaynak adlari
        "aaterm", "hituav", "termal", "sentetik", "tf2026",
    ]
    sizan = [word for word in yasakli if word in blob]
    assert not sizan, f"Paket ajana sizdiriyor: {sizan}"


@pytest.mark.skipif(not LLM_INPUT.is_file(), reason="paket yok")
def test_her_kosuda_kirilim_kaniti_var():
    """Toplam metriklerde gorunmeyen senaryolar icin kirilim kaniti sart.

    D3b/D4/D5 tam olarak toplam metriklerde gorunmez; kirilim alanlari
    eksikse ajan bu senaryolari teshis edemez.
    """
    packet = json.loads(LLM_INPUT.read_text(encoding="utf-8"))
    eksik = [
        kosu_id
        for kosu_id, run in packet["runs"].items()
        if not run.get("boyut_bandi_recall") or not run.get("sinif_karisikligi")
    ]
    assert not eksik, f"Bu kosularda kirilim kaniti yok: {eksik}"


@pytest.mark.skipif(not LLM_INPUT.is_file(), reason="paket yok")
def test_paketteki_kosular_cevap_anahtariyla_ayni():
    packet = json.loads(LLM_INPUT.read_text(encoding="utf-8"))
    key = json.loads(ANSWER_KEY.read_text(encoding="utf-8"))
    assert set(packet["runs"]) == set(key), (
        "llm_input.json ve answer_key.json farkli kosu kumeleri iceriyor; "
        "puanlama eksik veya fazla kosu uzerinden yapilir."
    )

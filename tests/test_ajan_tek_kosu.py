"""`--kosu` (tek kosu) modunun sonucu DISKE yazdigini dogrular.

GERCEK HATA: ilk surumde `--kosu` sonucu yalnizca ekrana basiyor, `--output`
yolunu hic kullanmiyordu. scripts/ajan_kontrol_tekrari.py bu komutu
`capture_output=True` ile subprocess olarak cagirdigi icin iki gercek kosunun
cevabi yakalanip ATILDI: API kotasi harcandi, sonuc kayboldu ve script
"tamam" diye raporladi.

Bu testler API anahtari gerektirmez; teshis_uret cagrisi taklit edilir.
"""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]

from teshis.ajan import ajan  # noqa: E402

SAHTE_TESHIS = {
    "run_id": "kosu_01",
    "diagnosis": "saglikli_referans",
    "evidence": ["mAP50 0.9200", "recall 0.8785"],
    "confidence": "yuksek",
    "limitations": ["UAP (n=15) ve UAI (n=17) azdir."],
    "next_measurement": "-",
}


def _calistir(argv: list[str]) -> None:
    with mock.patch.object(ajan, "teshis_uret", return_value=(dict(SAHTE_TESHIS), [])), \
         mock.patch.object(sys, "argv", ["ajan", *argv]):
        ajan.main()


def test_tek_kosu_ciktisini_diske_yazar(tmp_path):
    cikti = tmp_path / "cevap.json"
    log = tmp_path / "log.json"
    _calistir(["--kosu", "kosu_01", "--output", str(cikti), "--log", str(log)])

    assert cikti.is_file(), (
        "--kosu sonucu diske yazmadi. Sonuc yalnizca ekrana basilirsa, komutu "
        "subprocess ile cagiran script cevabi kaybeder ve API kotasi bosa gider."
    )
    icerik = json.loads(cikti.read_text(encoding="utf-8"))
    cevaplar = icerik if isinstance(icerik, list) else [icerik]
    assert any(c["run_id"] == "kosu_01" for c in cevaplar)


def test_tek_kosu_arac_kaydini_da_yazar(tmp_path):
    cikti, log = tmp_path / "cevap.json", tmp_path / "log.json"
    _calistir(["--kosu", "kosu_01", "--output", str(cikti), "--log", str(log)])
    assert log.is_file(), "--kosu arac kaydini yazmadi"


def test_tek_kosu_devam_ile_onceki_cevaplari_korur(tmp_path):
    """Ikinci bir kosu yazildiginda birincisi silinmemeli."""
    cikti, log = tmp_path / "cevap.json", tmp_path / "log.json"
    _calistir(["--kosu", "kosu_01", "--output", str(cikti), "--log", str(log)])

    ikinci = dict(SAHTE_TESHIS, run_id="kosu_11")
    with mock.patch.object(ajan, "teshis_uret", return_value=(ikinci, [])), \
         mock.patch.object(sys, "argv",
                           ["ajan", "--kosu", "kosu_11", "--output", str(cikti),
                            "--log", str(log), "--devam"]):
        ajan.main()

    icerik = json.loads(cikti.read_text(encoding="utf-8"))
    cevaplar = icerik if isinstance(icerik, list) else [icerik]
    kimlikler = {c["run_id"] for c in cevaplar}
    assert kimlikler == {"kosu_01", "kosu_11"}, (
        f"--devam onceki cevabi korumadi: {kimlikler}"
    )


def test_tekrar_scripti_gecici_hatayi_kotadan_ayirir():
    """503 gecicidir; kullaniciya bir gun beklemesini soylemek yanlistir."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kontrol_tekrari", ROOT / "scripts/ajan_kontrol_tekrari.py"
    )
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)

    assert "GECICI" in modul._hata_turu("ServerError: 503 UNAVAILABLE high demand")
    assert "KOTA" in modul._hata_turu("RESOURCE_EXHAUSTED: quota exceeded")
    assert modul._hata_turu("ValueError: baska bir sey") == "BILINMEYEN HATA"


def test_tekrar_scripti_dosya_yazilmadiysa_basarili_saymaz():
    """Cikis kodu 0 olsa bile dosya yoksa kosu basarisiz sayilmalidir."""
    kaynak = (ROOT / "scripts/ajan_kontrol_tekrari.py").read_text(encoding="utf-8")
    assert "hedef.is_file()" in kaynak, (
        "Tekrar scripti dosyanin gercekten yazildigini dogrulamiyor"
    )

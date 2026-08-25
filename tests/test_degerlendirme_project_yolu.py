"""D3 kosusunda gozlenen bir hatanin regresyon testi.

Bazi Ultralytics surumlerinde model.val()/model.train() cagrisina goreceli
bir ``project=`` yolu verilince cikti sessizce ``runs/detect/<project>``
altina yaziliyor; bizim kodumuz ise dosyalari dogrudan ``<project>``
altinda bekliyor (demo/data_loader.py, report klasorleri). D3 diagnostic
degerlendirmesinde tam olarak bu yuzden goruntuler kayboluyordu (bkz.
Bakim Gunlugu). Duzeltme: ``project=`` her zaman ``.resolve()`` ile mutlak
yol olarak verilmeli.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROJECT_PARAM_ICEREN_DOSYALAR = [
    ROOT / "teshis/degerlendirme/d1_sonuc.py",
    ROOT / "teshis/degerlendirme/model.py",
    ROOT / "teshis/degerlendirme/karsilastir.py",
]


@pytest.mark.parametrize("dosya", PROJECT_PARAM_ICEREN_DOSYALAR, ids=lambda yol: yol.name)
def test_project_parametresi_resolve_kullaniyor(dosya: Path):
    kaynak = dosya.read_text(encoding="utf-8")
    assert "project=str(" in kaynak, f"{dosya.name} icinde beklenen project= cagrisi bulunamadi"
    assert "project=str(output_dir),\n" not in kaynak
    assert "project=str(output_root),\n" not in kaynak

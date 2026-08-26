"""scripts/local_d3.py::build_dataset — sinif karisikligi uretimi testleri.

Sentetik bir mini dataset uzerinde calisir; gercek dataset'e dokunmaz.
D3 (UAP/UAI) ve D3b (tasit/insan) ayni kodu farkli sinif ciftiyle kullandigi
icin, takas mantiginin sinif ciftinden bagimsiz dogru calistigi burada
dogrulanir.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("local_d3", ROOT / "scripts/local_d3.py")
local_d3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(local_d3)


def _mini_dataset(root: Path, satirlar: dict[str, list[str]]) -> None:
    """satirlar: {dosya_adi: [etiket satirlari]} — her etikete bir goruntu esler."""
    for alt in ("images/train", "labels/train", "images/val", "images/test"):
        (root / alt).mkdir(parents=True, exist_ok=True)
    for ad, lines in satirlar.items():
        Image.new("RGB", (640, 640)).save(root / "images/train" / f"{ad}.jpg")
        (root / "labels/train" / f"{ad}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _oku(output: Path) -> dict[str, list[str]]:
    return {
        p.stem: p.read_text(encoding="utf-8").splitlines()
        for p in sorted((output / "labels/train").iterdir())
    }


def test_takas_yalnizca_hedef_sinif_ciftini_etkiler(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _mini_dataset(src, {
        "a": ["0 0.5 0.5 0.1 0.1", "1 0.4 0.4 0.1 0.1"],
        "b": ["2 0.3 0.3 0.1 0.1", "3 0.2 0.2 0.1 0.1"],
    })
    # swap_ratio=1.0 -> hedef ciftteki TUM satirlar takas edilir
    local_d3.build_dataset(src, out, 1.0, 42, class_pair=(2, 3), scenario="D3", version="v_test")
    sonuc = _oku(out)
    # tasit/insan (0/1) dokunulmadan kalmali
    assert sonuc["a"] == ["0 0.5 0.5 0.1 0.1", "1 0.4 0.4 0.1 0.1"]
    # UAP<->UAI takas edilmeli
    assert sonuc["b"][0].split()[0] == "3"
    assert sonuc["b"][1].split()[0] == "2"


def test_d3b_sinif_cifti_tasit_insani_takas_eder(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _mini_dataset(src, {
        "a": ["0 0.5 0.5 0.1 0.1", "1 0.4 0.4 0.1 0.1"],
        "b": ["2 0.3 0.3 0.1 0.1"],
    })
    local_d3.build_dataset(src, out, 1.0, 42, class_pair=(0, 1), scenario="D3b", version="v_test")
    sonuc = _oku(out)
    assert sonuc["a"][0].split()[0] == "1"   # tasit -> insan
    assert sonuc["a"][1].split()[0] == "0"   # insan -> tasit
    assert sonuc["b"] == ["2 0.3 0.3 0.1 0.1"]  # UAP dokunulmadi


def test_kutu_koordinatlari_degismez(tmp_path):
    """Sinif karisikligi yalnizca sinif ID'sini degistirir; geometri korunur."""
    src, out = tmp_path / "src", tmp_path / "out"
    _mini_dataset(src, {"a": ["0 0.123456 0.234567 0.05 0.06"]})
    local_d3.build_dataset(src, out, 1.0, 42, class_pair=(0, 1), scenario="D3b", version="v_test")
    alanlar = _oku(out)["a"][0].split()
    assert alanlar[1:] == ["0.123456", "0.234567", "0.05", "0.06"]


def test_takas_orani_sifir_hicbir_seyi_degistirmez(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    orijinal = {"a": ["0 0.5 0.5 0.1 0.1", "1 0.4 0.4 0.1 0.1"]}
    _mini_dataset(src, orijinal)
    local_d3.build_dataset(src, out, 0.0, 42, class_pair=(0, 1), scenario="D3b", version="v_test")
    assert _oku(out)["a"] == orijinal["a"]


def test_manifest_sinif_cifti_ve_sayimlari_kaydeder(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _mini_dataset(src, {"a": ["0 0.5 0.5 0.1 0.1", "1 0.4 0.4 0.1 0.1"]})
    local_d3.build_dataset(src, out, 1.0, 42, class_pair=(0, 1), scenario="D3b", version="v05_test")
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["scenario"] == "D3b"
    assert m["version"] == "v05_test"
    assert m["parameters"]["class_ids"] == [0, 1]
    assert m["parameters"]["class_names"] == ["tasit", "insan"]
    assert m["counts"]["changed_rows"] == 2
    assert m["counts"]["changed_by_class"] == {"tasit_to_insan": 1, "insan_to_tasit": 1}
    assert m["source_dataset_unchanged"] is True


def test_kaynak_dataset_degistirilmez(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _mini_dataset(src, {"a": ["0 0.5 0.5 0.1 0.1", "1 0.4 0.4 0.1 0.1"]})
    once = (src / "labels/train/a.txt").read_text(encoding="utf-8")
    local_d3.build_dataset(src, out, 1.0, 42, class_pair=(0, 1), scenario="D3b", version="v_test")
    assert (src / "labels/train/a.txt").read_text(encoding="utf-8") == once


def test_ayni_seed_ayni_sonucu_verir(tmp_path):
    kayitlar = {f"f{i}": [f"{i % 2} 0.5 0.5 0.1 0.1" for _ in range(4)] for i in range(10)}
    sonuclar = []
    for ad in ("out1", "out2"):
        src, out = tmp_path / f"src_{ad}", tmp_path / ad
        _mini_dataset(src, kayitlar)
        local_d3.build_dataset(src, out, 0.5, 42, class_pair=(0, 1), scenario="D3b", version="v_test")
        sonuclar.append(_oku(out))
    assert sonuclar[0] == sonuclar[1]


def test_ayni_sinif_cifti_reddedilir(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _mini_dataset(src, {"a": ["0 0.5 0.5 0.1 0.1"]})
    with pytest.raises(ValueError, match="iki farkli sinif"):
        local_d3.build_dataset(src, out, 0.5, 42, class_pair=(1, 1), scenario="X", version="v")


def test_gecersiz_takas_orani_reddedilir(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _mini_dataset(src, {"a": ["0 0.5 0.5 0.1 0.1"]})
    with pytest.raises(ValueError, match="0 ile 1"):
        local_d3.build_dataset(src, out, 1.5, 42, class_pair=(0, 1), scenario="X", version="v")


def test_data_yaml_vali_kaynak_datasetin_val_bolmesidir(tmp_path):
    """val, kaynak dataset'in operasyonel val'i olmali; kilitli tanı seti olmamali.

    Not: dogrudan "val_diagnostic" alt dizesi aranmaz, cunku pytest'in gecici
    klasor adi test adindan turedigi icin yanlis pozitif uretebiliyor. Bunun
    yerine val'in tam olarak kaynak dataset'in val yolunu gosterdigi kontrol
    edilir; bu daha kesin bir sozlesmedir.
    """
    src, out = tmp_path / "src", tmp_path / "out"
    _mini_dataset(src, {"a": ["0 0.5 0.5 0.1 0.1"]})
    local_d3.build_dataset(src, out, 0.5, 42, class_pair=(0, 1), scenario="D3b", version="v_test")
    satirlar = (out / "data.yaml").read_text(encoding="utf-8").splitlines()
    val = next(s for s in satirlar if s.startswith("val:")).split("val:", 1)[1].strip()
    assert Path(val) == (src / "images/val").resolve()

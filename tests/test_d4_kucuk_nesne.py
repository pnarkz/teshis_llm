"""scripts/senaryo_D4_kucuk_nesne.py::build_dataset — kucuk nesne silme testleri.

Sentetik mini dataset uzerinde calisir; gercek dataset'e dokunmaz.
En kritik sozlesme: bozulma esigi ile olcum bandi ayni formulu kullanmali,
aksi halde D4'un hipotezi ("yalnizca kucuk nesne recall duser") olculemez.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("local_d4", ROOT / "scripts/senaryo_D4_kucuk_nesne.py")
local_d4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(local_d4)

from teshis.degerlendirme.metrikler import BANTLAR, etkin_sqrt_alan  # noqa: E402


def _dataset(root: Path, satirlar: dict[str, list[str]], boyut=(640, 640)) -> None:
    for alt in ("images/train", "labels/train", "images/val", "images/test"):
        (root / alt).mkdir(parents=True, exist_ok=True)
    for ad, lines in satirlar.items():
        Image.new("RGB", boyut).save(root / "images/train" / f"{ad}.jpg")
        (root / "labels/train" / f"{ad}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _kur(tmp_path, satirlar, esik=16.0, boyut=(640, 640)):
    src, out = tmp_path / "src", tmp_path / "out"
    _dataset(src, satirlar, boyut)
    local_d4.build_dataset(src, out, esik, "D4", "v_test")
    return src, out


def _oku(out: Path, ad: str) -> list[str]:
    return (out / "labels/train" / f"{ad}.txt").read_text(encoding="utf-8").splitlines()


def test_esigin_altindaki_kutu_silinir(tmp_path):
    # 640x640'ta 0.01 kenar -> 6.4 px etkin boyut, esik 16 -> silinmeli
    _, out = _kur(tmp_path, {"a": ["1 0.5 0.5 0.01 0.01"]})
    assert _oku(out, "a") == []


def test_esigin_ustundeki_kutu_korunur(tmp_path):
    # 0.1 kenar -> 64 px etkin boyut -> korunmali
    satir = "0 0.5 0.5 0.1 0.1"
    _, out = _kur(tmp_path, {"a": [satir]})
    assert _oku(out, "a") == [satir]


def test_esik_tam_sinirda_korunur(tmp_path):
    """boyut < esik silinir; boyut == esik korunur (metrikler.boyut_bandi ile ayni sinir)."""
    kenar = 16.0 / 640  # tam 16 px etkin boyut
    satir = f"0 0.5 0.5 {kenar:.10f} {kenar:.10f}"
    _, out = _kur(tmp_path, {"a": [satir]})
    assert len(_oku(out, "a")) == 1


def test_karisik_dosyada_yalnizca_kucukler_silinir(tmp_path):
    buyuk, kucuk = "0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.01 0.01"
    _, out = _kur(tmp_path, {"a": [buyuk, kucuk, buyuk]})
    assert _oku(out, "a") == [buyuk, buyuk]


def test_cozunurluk_normalize_edilir(tmp_path):
    """Ayni oranli kutu, farkli cozunurlukte ayni karari almali."""
    satir = "0 0.5 0.5 0.02 0.02"   # 640'ta 12.8px -> silinir
    _, out_kucuk = _kur(tmp_path / "a", {"x": [satir]}, boyut=(640, 640))
    # 1024x1024'te ayni oran -> 20.48*0.625 = 12.8 px -> yine silinmeli
    _, out_buyuk = _kur(tmp_path / "b", {"x": [satir]}, boyut=(1024, 1024))
    assert _oku(out_kucuk, "x") == []
    assert _oku(out_buyuk, "x") == []


def test_kaynak_dataset_degistirilmez(tmp_path):
    src, out = _kur(tmp_path, {"a": ["1 0.5 0.5 0.01 0.01"]})
    assert (src / "labels/train/a.txt").read_text(encoding="utf-8").strip() == "1 0.5 0.5 0.01 0.01"


def test_manifest_sayimlari_dogru(tmp_path):
    _, out = _kur(tmp_path, {
        "a": ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.01 0.01"],
        "b": ["1 0.3 0.3 0.005 0.005"],
    })
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    c = m["counts"]
    assert c["kept_bbox_rows"] == 1
    assert c["removed_bbox_rows"] == 2
    assert c["removed_by_class"] == {"insan": 2}
    assert c["train_images"] == 2
    assert m["source_dataset_unchanged"] is True
    assert m["val_test_modified"] is False


def test_bozuk_satirlar_korunur(tmp_path):
    """Gecersiz YOLO satiri silinmemeli; sayilmali ama oldugu gibi kalmali."""
    _, out = _kur(tmp_path, {"a": ["bozuk satir", "0 0.5 0.5 0.1 0.1"]})
    assert "bozuk satir" in _oku(out, "a")
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["counts"]["invalid_rows_preserved"] == 1


def test_esik_config_ile_hizali():
    """D4 config'indeki esik, metrikler.BANTLAR'in ilk siniriyla ayni olmali.

    Aksi halde 'egitimden cikarilan bant' ile 'recall'i olculen bant' ortusmez
    ve senaryonun hipotezi dogrudan test edilemez.
    """
    config = yaml.safe_load(
        (ROOT / "senaryolar/veri/d4_kucuk_nesne_sinyal_kaybi.yaml").read_text(encoding="utf-8")
    )
    assert float(config["parametreler"]["etkin_sqrt_alan_esigi_px"]) == BANTLAR[0][0]


def test_bozulma_ve_olcum_ayni_formulu_kullanir():
    """local_d4, boyut hesabini metrikler modulunden import etmelidir."""
    kaynak = (ROOT / "scripts/senaryo_D4_kucuk_nesne.py").read_text(encoding="utf-8")
    assert "from teshis.degerlendirme.metrikler import" in kaynak
    assert local_d4.etkin_sqrt_alan is etkin_sqrt_alan


def test_gecersiz_esik_reddedilir(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _dataset(src, {"a": ["0 0.5 0.5 0.1 0.1"]})
    with pytest.raises(ValueError, match="pozitif"):
        local_d4.build_dataset(src, out, 0.0, "D4", "v_test")


def test_data_yaml_vali_kaynak_bolmesidir(tmp_path):
    src, out = _kur(tmp_path, {"a": ["0 0.5 0.5 0.1 0.1"]})
    satirlar = (out / "data.yaml").read_text(encoding="utf-8").splitlines()
    val = next(s for s in satirlar if s.startswith("val:")).split("val:", 1)[1].strip()
    assert Path(val) == (src / "images/val").resolve()

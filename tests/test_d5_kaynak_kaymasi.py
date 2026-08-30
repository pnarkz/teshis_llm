"""scripts/senaryo_D5_kaynak_kaymasi.py::build_dataset — kaynak/alan kaymasi testleri.

Sentetik mini dataset uzerinde calisir. En kritik sozlesme: D5 etiketlere
dokunmamali ve goruntu kopyalamamali; yalnizca train listesini daraltmali.
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
_spec = importlib.util.spec_from_file_location("local_d5", ROOT / "scripts/senaryo_D5_kaynak_kaymasi.py")
local_d5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(local_d5)

from teshis.veri.istatistik import kaynak_adi  # noqa: E402

# kaynak_adi'nin tanidigi onekler
KARELER = {
    "aaterm__001": ["0 0.5 0.5 0.1 0.1", "1 0.3 0.3 0.1 0.1"],
    "aaterm__002": ["0 0.4 0.4 0.1 0.1"],
    "hituav__001": ["1 0.5 0.5 0.1 0.1"],
    "termal__001": ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1", "1 0.6 0.6 0.1 0.1"],
}


def _dataset(root: Path) -> None:
    for alt in ("images/train", "labels/train", "images/val", "images/test"):
        (root / alt).mkdir(parents=True, exist_ok=True)
    for ad, lines in KARELER.items():
        Image.new("RGB", (640, 640)).save(root / "images/train" / f"{ad}.jpg")
        (root / "labels/train" / f"{ad}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _kur(tmp_path, izinli=("aaterm",)):
    src, out = tmp_path / "src", tmp_path / "out"
    _dataset(src)
    local_d5.build_dataset(src, out, list(izinli), "D5", "v_test")
    return src, out


def _liste(out: Path) -> list[str]:
    return [Path(s).stem for s in (out / "train_images.txt").read_text(encoding="utf-8").split()]


def test_yalnizca_izinli_kaynak_listeye_girer(tmp_path):
    _, out = _kur(tmp_path, izinli=("aaterm",))
    assert sorted(_liste(out)) == ["aaterm__001", "aaterm__002"]


def test_birden_fazla_izinli_kaynak(tmp_path):
    _, out = _kur(tmp_path, izinli=("aaterm", "hituav"))
    assert sorted(_liste(out)) == ["aaterm__001", "aaterm__002", "hituav__001"]


def test_etiketler_hic_degistirilmez(tmp_path):
    """D5 bir veri SECIMI bozulmasidir; etiket icerigine dokunmamalidir."""
    src, out = _kur(tmp_path)
    for ad, lines in KARELER.items():
        assert (src / "labels/train" / f"{ad}.txt").read_text(encoding="utf-8").splitlines() == lines
    # cikti klasorunde hic etiket/goruntu kopyasi olmamali
    assert not (out / "labels").exists()
    assert not (out / "images").exists()


def test_goruntu_kopyalanmaz_manifest_only(tmp_path):
    _, out = _kur(tmp_path)
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["copy_mode"] == "manifest_only"
    assert m["labels_modified"] is False
    # cikti klasorunde yalnizca metin dosyalari olmali
    uzantilar = {p.suffix for p in out.iterdir() if p.is_file()}
    assert uzantilar <= {".txt", ".json", ".yaml"}


def test_manifest_kaynak_sayimlari(tmp_path):
    _, out = _kur(tmp_path, izinli=("aaterm",))
    c = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["counts"]
    assert c["train_frames_before"] == 4
    assert c["train_frames_after"] == 2
    assert c["train_frames_removed"] == 2
    assert c["frames_by_source_before"] == {"aaterm": 2, "hituav": 1, "termal": 1}
    assert c["frames_by_source_after"] == {"aaterm": 2}
    # aaterm: 2+1=3 bbox tutuldu; hituav 1 + termal 3 = 4 bbox cikarildi
    assert c["kept_bbox"] == 3
    assert c["removed_bbox"] == 4


def test_bilinmeyen_kaynak_hic_kare_birakmazsa_hata(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _dataset(src)
    with pytest.raises(ValueError, match="hic kare bulunamadi"):
        local_d5.build_dataset(src, out, ["olmayan_kaynak"], "D5", "v_test")


def test_bos_izinli_liste_reddedilir(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _dataset(src)
    with pytest.raises(ValueError, match="En az bir izinli kaynak"):
        local_d5.build_dataset(src, out, [], "D5", "v_test")


def test_data_yaml_vali_kaynak_bolmesidir(tmp_path):
    src, out = _kur(tmp_path)
    satirlar = (out / "data.yaml").read_text(encoding="utf-8").splitlines()
    val = next(s for s in satirlar if s.startswith("val:")).split("val:", 1)[1].strip()
    assert Path(val) == (src / "images/val").resolve()


def test_kaynak_fonksiyonu_olcum_ile_ayni():
    """D5 ve metrikler.py ayni kaynak_adi fonksiyonunu kullanmali.

    Aksi halde 'egitimden cikarilan kaynak' ile 'performansi olculen kaynak'
    farkli tanimlara dayanir ve senaryonun hipotezi test edilemez.
    """
    from teshis.degerlendirme import metrikler

    assert local_d5.kaynak_adi is kaynak_adi
    assert metrikler.kaynak_adi is kaynak_adi


def test_config_izinli_kaynak_datasette_mevcut():
    """Config'teki izinli kaynaklar, veri raporundaki kaynak adlariyla eslesmeli."""
    config = yaml.safe_load(
        (ROOT / "senaryolar/veri/d5_kaynak_alani_kaymasi.yaml").read_text(encoding="utf-8")
    )
    izinli = set(config["parametreler"]["izinli_kaynaklar"])
    rapor = ROOT / "reports/veri_raporu.json"
    if not rapor.is_file():
        pytest.skip("veri_raporu.json yok")
    mevcut = set(json.loads(rapor.read_text(encoding="utf-8"))["kaynak_toplam"])
    assert izinli <= mevcut, f"Config'te datasette olmayan kaynak var: {izinli - mevcut}"

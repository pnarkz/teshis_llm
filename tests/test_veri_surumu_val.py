"""Hicbir senaryo veri surumu, egitim val'i olarak val_diagnostic kullanmamalidir.

Bu test gercek bir metodoloji hatasindan dogdu: scripts/local_d3.py uretilen
data.yaml'a `val: val_diagnostic/images` yaziyordu. Ultralytics best.pt'yi val
uzerindeki en iyi skora gore sectigi icin, D3'un best.pt'si sonradan tum
senaryolari karsilastirmak icin kullandigimiz kilitli tanı setinin uzerinde
secilmis oldu. Bu, checkpoint seciminin degerlendirme setine bakmasi demektir
ve o senaryoya digerlerine gore iyimser bir yanlilik kazandirir.

Kilitli tanı seti yalnizca egitim bittikten sonra, teshis/degerlendirme
altindaki komutlarla olculmelidir.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Senaryo veri surumu data.yaml'i ureten scriptler.
URETICI_SCRIPTLER = [
    ROOT / "scripts/local_d2b.py",
    ROOT / "scripts/local_d3.py",
    ROOT / "scripts/kaggle_d2a.py",
    ROOT / "scripts/kaggle_d2b.py",
]

VERI_SURUMU_YAMLLARI = sorted((ROOT / "veri_surumleri").glob("*/data.yaml"))

# Bu hata kesfedilmeden ONCE uretilmis, henuz duzeltilmemis veri surumleri.
# Su an bos: v04_d3'un data.yaml'i duzeltildi (etiketler degismedi, yalnizca
# val/test isaretcileri operasyonel bolmelere cevrildi) ve senaryo yeniden
# kosuluyor. Eski yaml git gecmisinde, eski kosu experiments/run_D3_42_local
# altinda duruyor. Ayrinti: README "Bakim Gunlugu" 2026-08-25 (6).
BILINEN_ETKILENEN: set[str] = set()


@pytest.mark.parametrize("script", URETICI_SCRIPTLER, ids=lambda p: p.name)
def test_script_val_diagnostic_i_egitim_vali_yapmaz(script: Path):
    kaynak = script.read_text(encoding="utf-8")
    # `val:` satirini yazan f-string'lerde val_diagnostic gecmemelidir.
    for satir in kaynak.splitlines():
        if "val:" in satir and "val_diagnostic" in satir:
            pytest.fail(
                f"{script.name} data.yaml'a val olarak val_diagnostic yaziyor: {satir.strip()}"
            )


@pytest.mark.skipif(not VERI_SURUMU_YAMLLARI, reason="uretilmis veri surumu yok")
@pytest.mark.parametrize(
    "yaml_yolu", VERI_SURUMU_YAMLLARI, ids=lambda p: p.parent.name
)
def test_uretilmis_veri_surumu_vali_diagnostic_degil(yaml_yolu: Path):
    """Diskteki veri surumlerinin val satiri kilitli tanı setini gostermemelidir."""
    if yaml_yolu.parent.name in BILINEN_ETKILENEN:
        pytest.xfail(
            f"{yaml_yolu.parent.name}: hata kesfedilmeden once uretildi; "
            "tarihsel kayit olarak korunuyor, senaryo yeniden kosulmali."
        )
    icerik = yaml_yolu.read_text(encoding="utf-8")
    val_satiri = next(
        (s for s in icerik.splitlines() if re.match(r"\s*val\s*:", s)), ""
    )
    assert "val_diagnostic" not in val_satiri, (
        f"{yaml_yolu.parent.name}/data.yaml egitim val'i olarak kilitli tanı setini "
        f"kullaniyor: {val_satiri.strip()}. Bu veri surumuyle egitilen modelin best.pt'si "
        "degerlendirme seti uzerinde secilmis olur; senaryo karsilastirmasi adil olmaz."
    )

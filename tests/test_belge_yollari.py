"""Belgelerde adi gecen dosya ve klasor yollari gercekten var mi?

Bu test gercek bir sorundan dogdu: proje yeniden duzenlenirken rapor
klasorleri ve script adlari degisti, ancak README ve docs/ icindeki
referanslar geride kaldi. Boyle bir kayma sessizdir - belgeyi okuyan kisi
olmayan bir dosyayi arar. Test, referanslarin kod tabaniyla birlikte
guncellenmesini zorunlu kilar.
"""

import functools
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BELGELER = sorted([*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md")])

# Metin icinde yol gibi gorunen ama hicbir zaman dosya olmayan kaliplar.
YOKSAY = re.compile(
    r"^(dataset|termal_teshis|runs|\.\.\.|C:|/kaggle|https?:)"
    # Ultralytics results.csv sutun adlari yol gibi gorunur ama dosya degil:
    # `train/box_loss`, `val/cls_loss`, `metrics/mAP50(B)`, `lr/pg0`.
    r"|^(train|val|metrics|lr)/"
)


@functools.lru_cache(maxsize=1)
def _git_disi_kontrolu():
    """Bir yolun .gitignore kapsaminda olup olmadigini soyleyen fonksiyon.

    GERCEK HATA: bu testin yoksayma listesi bir donem `experiments/`,
    `veri_surumleri/`, `*.pt`, `*.jpg` gibi kaliplari ELLE tasiyordu - yani
    .gitignore'un bir KOPYASIYDI. Taze bir klonda test kirildi, cunku
    `val_diagnostic/manifest.json` listede yoktu ama Git disiydi.

    Ayni kural iki yerde yasarsa biri geride kalir. Tek kaynak .gitignore
    olmali; onu da en dogru okuyan Git'in kendisidir.
    """
    try:
        subprocess.run(["git", "rev-parse"], cwd=ROOT, check=True,
                       capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return lambda _yol: False        # Git yoksa hicbir sey yoksayilmaz

    def disi(yol: str) -> bool:
        sonuc = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", yol],
            cwd=ROOT, capture_output=True,
        )
        return sonuc.returncode == 0

    return disi

# Kod bloklari ve satir ici kod icindeki yol adaylari.
YOL_KALIBI = re.compile(r"`([A-Za-z0-9_./-]+/[A-Za-z0-9_./-]+)`")
MD_LINK = re.compile(r"\]\(([^)#]+\.md)\)")


def _yol_adaylari(metin: str) -> set[str]:
    adaylar = set(YOL_KALIBI.findall(metin))
    adaylar |= set(MD_LINK.findall(metin))
    git_disi = _git_disi_kontrolu()
    return {a for a in adaylar
            if not YOKSAY.search(a) and not git_disi(a)}


@pytest.mark.parametrize("belge", BELGELER, ids=lambda p: p.name)
def test_belgedeki_yollar_mevcut(belge: Path):
    eksik = []
    for aday in sorted(_yol_adaylari(belge.read_text(encoding="utf-8"))):
        hedef = ROOT / aday
        if not (hedef.exists() or hedef.parent.exists()):
            eksik.append(aday)
    assert not eksik, f"{belge.name} icinde bulunamayan yollar: {eksik}"


def test_readme_belge_haritasi_eksiksiz():
    """README'nin belge tablosu docs/ altindaki her belgeyi listelemeli."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for belge in sorted((ROOT / "docs").glob("*.md")):
        if belge.name.startswith("proje-brifingi"):
            continue  # disaridan gelen referans belge, haritada olmasi sart degil
        assert f"docs/{belge.name}" in readme, (
            f"docs/{belge.name} README'deki belge haritasinda yok"
        )


def test_mimari_belgesi_gercek_klasorleri_anlatir():
    """docs/MIMARI.md'de anlatilan ust duzey klasorler gercekten var olmali."""
    mimari = (ROOT / "docs/MIMARI.md").read_text(encoding="utf-8")
    for klasor in ("teshis/", "scripts/", "senaryolar/", "reports/", "docs/", "tests/"):
        assert klasor in mimari, f"{klasor} MIMARI.md'de anlatilmiyor"
        assert (ROOT / klasor.rstrip("/")).is_dir()


def test_rapor_adlandirma_kurali_korunuyor():
    """reports/ altindaki her klasor tanimli oneklerden birini kullanmali.

    Kural docs/MIMARI.md'de tanimli: onek = ne tur, sonek = hangisi.
    """
    izinli = ("senaryo_", "referans_", "yolo26n_", "eski_", "hata_galerisi_",
              "kirilim", "ajan_denemesi", "model_secimi", "kanit",
              "kontrol_")
    hatali = [
        d.name for d in (ROOT / "reports").iterdir()
        if d.is_dir() and not d.name.startswith(izinli)
    ]
    assert not hatali, f"Adlandirma kuralina uymayan rapor klasorleri: {hatali}"


def test_senaryo_klasorlerinde_gorseller_tek_tip():
    """Her senaryo raporunda val ciktisi 'gorseller/' adini tasimali."""
    hatali = []
    for klasor in (ROOT / "reports").iterdir():
        if not klasor.is_dir():
            continue
        for alt in klasor.iterdir():
            if alt.is_dir() and "val_diagnostic" in alt.name:
                hatali.append(f"{klasor.name}/{alt.name}")
    assert not hatali, f"Eski adlandirmada kalan alt dizinler: {hatali}"

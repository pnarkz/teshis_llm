"""Belgelerde adi gecen dosya ve klasor yollari gercekten var mi?

Bu test gercek bir sorundan dogdu: proje yeniden duzenlenirken rapor
klasorleri ve script adlari degisti, ancak README ve docs/ icindeki
referanslar geride kaldi. Boyle bir kayma sessizdir - belgeyi okuyan kisi
olmayan bir dosyayi arar. Test, referanslarin kod tabaniyla birlikte
guncellenmesini zorunlu kilar.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BELGELER = sorted([*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md")])

# Metin icinde yol gibi gorunen ama dosya olmayan kaliplar.
YOKSAY = re.compile(
    r"^(dataset|termal_teshis|runs|\.\.\.|C:|/kaggle|https?:)"
    r"|\.(jpg|png|pt|onnx)$"          # ornek gorseller/agirliklar (gitignore'da)
    r"|^(experiments|veri_surumleri)/"  # uretilen kosu ciktilari (gitignore'da)
)

# Kod bloklari ve satir ici kod icindeki yol adaylari.
YOL_KALIBI = re.compile(r"`([A-Za-z0-9_./-]+/[A-Za-z0-9_./-]+)`")
MD_LINK = re.compile(r"\]\(([^)#]+\.md)\)")


def _yol_adaylari(metin: str) -> set[str]:
    adaylar = set(YOL_KALIBI.findall(metin))
    adaylar |= set(MD_LINK.findall(metin))
    return {a for a in adaylar if not YOKSAY.search(a)}


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
              "kirilim", "ajan_denemesi", "model_secimi", "kanit")
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

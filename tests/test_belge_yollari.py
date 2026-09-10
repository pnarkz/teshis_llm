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
              "kirilim", "ajan_denemesi", "ajan_deneyleri", "model_secimi", "kanit",
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


# --- README'nin tasidigi sayilar --------------------------------------------

def test_readme_ajan_tablosu_deneyle_uyusuyor():
    """README'deki rol bazli tablo gercek deney sonucuyla eslesmeli.

    Bu tablo projenin en cok alintilanacak sayilarini tasiyor. Elle yazili
    oldugu icin deney yeniden kosuldugunda sessizce bayatlar - bu projede
    tam olarak boyle hatalar birikti ("ajan denemesinde tekrar yok" cumlesi
    tekrarli deney kosulduktan sonra bile ekranda duruyordu).
    """
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    from data_loader import ajan_deneyi

    deney = ajan_deneyi()
    if deney is None:
        pytest.skip("henuz calistirilmis bir ajan deneyi yok")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    p = deney["puan"]
    tekrar = p["gozlem"] // max(p["kosu"], 1)
    assert f"{p['kosu']} koşu × {tekrar} tekrar = **{p['gozlem']} gözlem**" in readme, (
        f"README ajan deneyinin buyuklugunu yanlis yaziyor; dogrusu "
        f"{p['kosu']} kosu x {tekrar} tekrar = {p['gozlem']} gozlem"
    )
    for rol, d in (p.get("rol_bazli") or {}).items():
        for alan in ("dogru_teshis", "tespit_farkindalikli"):
            assert f"{d[alan]:.3f}" in readme, (
                f"README'de {rol} rolunun {alan} degeri ({d[alan]:.3f}) yok"
            )


def test_readme_senaryo_kosu_aritmetigi_dogru():
    """"14 senaryo / 26 kosu" ve toplami README'de turetilenle ayni olmali."""
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    import katalog

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    h = katalog.senaryo_kosu_haritasi()
    assert f"{h['senaryo_sayisi']} senaryo · {h['toplam']} koşu" in readme
    assert (f"{h['senaryo_kosusu']} senaryo koşusu + {h['altyapi_kosusu']} "
            f"altyapı koşusu\n= {h['toplam']}") in readme.replace("  ", " ") or (
        f"{h['senaryo_kosusu']} senaryo koşusu + {h['altyapi_kosusu']} altyapı koşusu"
        in readme), "README'deki senaryo/kosu aritmetigi turetilenle uyusmuyor"


def test_readme_test_sayisi_guncel():
    """README kac test oldugunu soyluyorsa dogru soylemeli."""
    import subprocess
    import sys

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*(\d+) test\*\*", readme)
    if not m:
        pytest.skip("README test sayisi vermiyor")
    yazan = int(m.group(1))
    sonuc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True)
    toplanan = re.search(r"(\d+) tests? collected", sonuc.stdout)
    assert toplanan, "test sayisi okunamadi"
    gercek = int(toplanan.group(1))
    assert abs(gercek - yazan) <= 5, (
        f"README '{yazan} test' diyor ama pakette {gercek} test var"
    )

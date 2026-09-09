"""docs/SUNUM.md'deki sayilar veriyle uyusuyor mu?

SUNUM.md sunumda soylenecek cumleleri tasir; icindeki her sayi sozlu olarak
iddia edilecektir. Elle yazilan sayilar bu projede tekrar tekrar bayatladi -
"bes iddia zayifladi" cumlesi buna ornektir: elle sayilmisti ve E1 ile E2
atlanmisti, dogrusu yediydi.

Bu testler ozetteki sayilari kaynagina baglar. Bir kontrol kosusu eklendiginde
ya da karsilastirma kurali degistiginde test kirilir ve metin guncellenir.
"""

import csv
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUNUM = ROOT / "docs/SUNUM.md"

DERECELENDIRILEN = {"guclu", "zayif", "gurultu icinde"}


@pytest.fixture(scope="module")
def metin() -> str:
    return SUNUM.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def seviyeler() -> dict[str, str]:
    from teshis.degerlendirme.senaryo_ozeti import kanit_gucu

    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        return {r["scenario"]: kanit_gucu(r["scenario"])["seviye"]
                for r in csv.DictReader(f)}


def test_derecelendirme_sayilari_dogru(metin, seviyeler):
    toplam = len(seviyeler)
    derece = sum(1 for s in seviyeler.values() if s in DERECELENDIRILEN)
    beklenen = f"{toplam} kosunun **{derece}'si** derecelendiriliyor"
    assert beklenen in metin or f"**{derece}'i**" in metin, (
        f"SUNUM.md derecelendirme sayisini yanlis soyluyor; dogrusu "
        f"{toplam} kosudan {derece}'si (kalan {toplam - derece})"
    )
    assert f"kalan {toplam - derece}" in metin


def test_zayiflayan_iddia_sayisi_turetilenle_ayni(metin):
    """'yedi iddia zayifladi' cumlesi demoda turetilen tabloyla ayni olmali."""
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    from bolumler.karsilastirma import _esik_buyumesi
    from data_loader import load_results

    _, zayiflayan, _ = _esik_buyumesi(load_results())
    yazi = {5: "bes", 6: "alti", 7: "yedi", 8: "sekiz", 9: "dokuz"}
    n = len(zayiflayan)
    assert n in yazi, f"beklenmedik sayi: {n}"
    assert f"**{yazi[n]} iddia zayifladi**" in metin, (
        f"SUNUM.md'de zayiflayan iddia sayisi guncel degil; dogrusu {n} "
        f"({yazi[n]})"
    )


def test_saglikli_kosu_guclu_diye_anilmiyor(metin, seviyeler):
    """Metin, bozulmasiz bir kosuyu bulgu gibi sunmamali."""
    from teshis.degerlendirme.karsilastirilabilirlik import bozulmasiz_mi

    for ad, seviye in seviyeler.items():
        if bozulmasiz_mi(ad):
            assert seviye not in DERECELENDIRILEN, (
                f"{ad} bozulmasiz ama '{seviye}' olarak derecelendirilmis"
            )


def test_metinde_kaynaksiz_kesin_yuzde_yok(metin):
    """Her yuzdenin yaninda neye ait oldugu yazmali - kaba bir okunabilirlik
    kontrolu degil, 'ajan %83 dogru bildi' turu tek basina birakilmis bir
    sayinin sunumda yanlis okunmasini onlemek icin."""
    satirlar = [s for s in metin.splitlines() if re.search(r"%\d", s)]
    ciplak = [s.strip() for s in satirlar if len(s.strip()) < 25]
    assert not ciplak, f"Baglamsiz yuzde iceren satirlar: {ciplak}"


# --- docs/SUNUM_SENARYOSU.md: sayfa sayfa sunum metni ------------------------

SENARYO = ROOT / "docs/SUNUM_SENARYOSU.md"


@pytest.fixture(scope="module")
def senaryo_metni() -> str:
    return SENARYO.read_text(encoding="utf-8")


def test_sunum_senaryosu_var(senaryo_metni):
    assert "## 1. Genel Bakis" in senaryo_metni


def test_kapanis_sayilari_deneyle_uyusuyor(senaryo_metni):
    """Metindeki iki sozlu iddia gercek deney sonucuyla eslesmeli.

    Bu iki cumle sunumda AGIZDAN soylenecek; bayatlarsa izleyiciye yanlis
    sayi verilir. Metnin geri kalani sayilari ekrandan okutur, bu ikisi
    cumlenin kendisi oldugu icin burada baglanir.
    """
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    from data_loader import ajan_deneyi, ajan_deneyi_kosu_bazli

    deney = ajan_deneyi()
    if deney is None:
        pytest.skip("henuz calistirilmis bir ajan deneyi yok")

    yazi = {1: "bir", 2: "iki", 3: "uc", 4: "dort", 5: "bes", 6: "alti",
            7: "yedi", 8: "sekiz", 9: "dokuz", 10: "on", 11: "on bir",
            12: "on iki", 13: "on uc", 14: "on dort"}
    # Diakritikler katlanir ve BOSLUK NORMALLESTIRILIR: aranan ifadeler
    # metinde satir sonunda bolunmus olabilir - "uc" satir sonunda kalip
    # "tekrarinda" alt satirda alinti isaretiyle baslayabilir. Test satir
    # kaydirmasina degil, cumlenin kendisine bakmali.
    sade = senaryo_metni
    for a, b in zip("öüışç", "ouisc"):
        sade = sade.replace(a, b)
    sade = re.sub(r"[\s>*]+", " ", sade).lower()

    kontrol = (deney["puan"].get("rol_bazli") or {}).get("kontrol")
    assert kontrol, "kontrol rolu puan raporunda yok"
    assert kontrol["dogru_teshis"] == 1.0, (
        "metin kontrollerde hic uydurma olmadigini soyluyor; deney aksini "
        f"gosteriyor ({kontrol['dogru_teshis']})"
    )
    n = yazi[kontrol["gozlem"]]
    assert f"{n} gozlemin" in sade, (
        f"Metin kontrol gozlem sayisini '{n} gozlemin' diye yazmali "
        f"(deneyde {kontrol['gozlem']} gozlem)"
    )
    assert f"{n}unda da" in sade, (
        f"Metin '{n}unda da ... uydurmadi' cumlesini tasimali"
    )

    kosular = ajan_deneyi_kosu_bazli()
    tutarli = [k for k in kosular if k["hukum_tutarli"]]
    assert len(tutarli) == len(kosular), (
        f"metin butun kosularin tutarli oldugunu soyluyor; "
        f"{len(kosular) - len(tutarli)} kosu degisken"
    )
    k = yazi[len(kosular)]
    assert f"{k} kosunun" in sade and f"{k}u de" in sade, (
        f"Metin kosu sayisini '{k} kosunun {k}u de' diye yazmali "
        f"(deneyde {len(kosular)} kosu)"
    )
    tekrar = yazi[kosular[0]["tekrar"]]
    assert f"{tekrar} tekrarinda" in sade, (
        f"Metin tekrar sayisini '{tekrar}' olarak yazmali "
        f"(deneyde {kosular[0]['tekrar']})"
    )


def test_senaryo_metni_korlugu_vurguluyor(senaryo_metni):
    """En sik yanlis anlasilan nokta: ajan goruntulere BAKMIYOR."""
    sade = senaryo_metni.replace("ö", "o").replace("ü", "u").replace("ı", "i")
    assert "goruntulere bakmiyor" in sade.lower()
    assert "cevap anahtari" in sade.lower()

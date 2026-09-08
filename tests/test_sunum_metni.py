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

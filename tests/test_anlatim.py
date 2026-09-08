"""senaryolar/anlatim.yaml: senaryo ozetinin elle yazilan TEK alani.

Bu dosya arayuzde "ne olcuyor" kutusunda dogrudan gorunur, dolayisiyla iki
sey garanti edilmelidir: defterdeki her kosunun bir karsiligi olmali ve
metinler duzgun Turkce olmali.

GERCEK HATA: `E3` bir donem dosyada IKI KEZ tanimliydi. YAML sessizce
sonuncuyu alir, ilk tanim hicbir yerde gorunmez ve hicbir hata da vermez.
Bu tur bir sessiz uzerine yazma, defterdeki "iki D2b satiri" olayinin
aynisidir - bu projede tekrarlayan bir oruntu.
"""

import csv
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ANLATIM = ROOT / "senaryolar/anlatim.yaml"
RESULTS = ROOT / "results.csv"

TURKCE = re.compile("[çğıöşüÇĞİÖŞÜ]")


@pytest.fixture(scope="module")
def anlatim() -> dict:
    return yaml.safe_load(ANLATIM.read_text(encoding="utf-8"))


def test_hicbir_anahtar_iki_kez_tanimlanmamis():
    """Tekrarlanan anahtar YAML'de sessizce uzerine yazilir."""
    anahtarlar = re.findall(
        r"^([A-Za-z][\w]*):", ANLATIM.read_text(encoding="utf-8"), re.MULTILINE
    )
    tekrar = sorted({a for a in anahtarlar if anahtarlar.count(a) > 1})
    assert not tekrar, f"Bu anahtarlar birden fazla kez tanimli: {tekrar}"


def test_defterdeki_her_kosunun_anlatimi_var(anlatim):
    """Senaryo sayfasi 'ne olcuyor' kutusunu bu dosyadan alir."""
    with RESULTS.open(encoding="utf-8") as f:
        kodlar = {r["scenario"].split()[0] for r in csv.DictReader(f)}
    eksik = sorted(k for k in kodlar if k not in anlatim)
    assert not eksik, f"anlatim.yaml'de karsiligi olmayan kosular: {eksik}"


def test_metinler_duzgun_turkce(anlatim):
    """Bu metinler ekranda oldugu gibi gorunur."""
    for kod, metin in anlatim.items():
        assert TURKCE.search(metin), f"{kod}: ASCII Turkce kalmis -> {metin[:60]}"


def test_beklenen_kanit_metinleri_turkce():
    """`beklenen_kanit` senaryo sayfasinda "beklenen etki" kutusunda gorunur.

    Bu alan konfig dosyalarinda durdugu icin arayuz dili taramasinin (Python
    dosyalari) disinda kaliyordu ve ekranda "yalnizca kucuk nesne recall
    duser" yaziyordu.
    """
    kotu = []
    for yol in sorted((ROOT / "senaryolar").glob("*/*.yaml")):
        konfig = yaml.safe_load(yol.read_text(encoding="utf-8")) or {}
        metin = konfig.get("beklenen_kanit")
        if isinstance(metin, str) and not TURKCE.search(metin):
            kotu.append(f"{yol.name}: {metin[:60]}")
    assert not kotu, "ASCII Turkce kalmis beklenen_kanit alanlari:\n" + "\n".join(kotu)

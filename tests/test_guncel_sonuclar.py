"""docs/BULGULAR.md'nin "GUNCEL SONUCLAR" ozetini kaynak verilere baglar.

Bu ozet, 1500 satirlik kronolojik belgeyi okumadan durumu anlamak icin var.
Tam da bu yuzden bayatlamasi en tehlikeli yer: okuyucu onu guncel sanip
yaniltici bir tabloya bakar. Testler ozetteki her sayiyi results.csv'ye ve
olcum dosyalarina karsi dogrular.
"""

import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BULGULAR = ROOT / "docs/BULGULAR.md"
BASLIK = "# GUNCEL SONUCLAR"


@pytest.fixture(scope="module")
def ozet() -> str:
    metin = BULGULAR.read_text(encoding="utf-8")
    assert BASLIK in metin, (
        "BULGULAR.md'de 'GUNCEL SONUCLAR' ozeti yok. Bu ozet, kronolojik "
        "belgeyi bastan okumadan durumu anlamanin tek yoludur."
    )
    bas = metin.index(BASLIK)
    son = metin.index("\n---\n", metin.index("Bilinen ve kapatilmamis eksikler"))
    return metin[bas:son]


@pytest.fixture(scope="module")
def satirlar() -> dict[str, dict]:
    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        return {s["scenario"]: s for s in csv.DictReader(f)}


@pytest.fixture(scope="module")
def v00() -> dict:
    return json.loads(
        (ROOT / "reports/referans_v00/d1_metrics.json").read_text(encoding="utf-8")
    )


# Ozet tablosundaki senaryo -> results.csv senaryo adi
TABLO = {
    "D2a": "D2a", "D3": "D3", "D4": "D4", "D5": "D5", "D2b": "D2b",
    "D3b": "D3b", "D1": "D1", "D6b": "D6b",
    "E4 (imgsz 512)": "E4 imgsz512",
    "E1 `last.pt`": "E1 last_pt",
    "E1 `best.pt`": "E1",
    "E2": "E2",
}


def test_ozet_tablosundaki_farklar_results_csv_ile_uyusuyor(ozet, satirlar, v00):
    """Her satirdaki ΔmAP50 ve Δrecall, kaynak verilerden hesaplanani vermeli."""
    hatali = []
    for etiket, senaryo in TABLO.items():
        s = satirlar.get(senaryo)
        if s is None:
            hatali.append(f"{senaryo} results.csv'de yok")
            continue
        for alan, sutun in (("mAP50", "mAP50"), ("recall", "recall")):
            beklenen = f"{float(s[sutun]) - v00[alan]:+.4f}"
            if beklenen not in ozet:
                hatali.append(f"{etiket} Δ{alan}={beklenen} ozette yok")
    assert not hatali, hatali


def test_ozetteki_gurultu_esigi_c2_olcumunden_geliyor(ozet, satirlar, v00):
    """Esik degerleri C2 kontrolunun gercek farklarindan turetilmis olmali."""
    c2 = satirlar.get("C2 seed7")
    if c2 is None:
        pytest.skip("C2 kontrolu henuz results.csv'de yok")
    for alan in ("mAP50", "precision", "recall"):
        esik = abs(float(c2[alan]) - v00[alan])
        assert f"{esik:.4f}" in ozet, (
            f"Ozetteki {alan} esigi C2 olcumuyle ({esik:.4f}) uyusmuyor"
        )


def test_ozet_orneklem_sinirini_acikca_soyluyor(ozet):
    """Skor tek basina verilmemeli; n=1 sinirinin ozette yazili olmasi sart.

    Bir mentorun ilk sorusu "guven araligi ne" olur. Ozet bu soruyu
    okuyucudan once sormali.
    """
    for ifade in ("NOKTA TAHMIN", "tekrar yok"):
        assert ifade.lower() in ozet.lower(), f"Ozette '{ifade}' gecmiyor"
    assert re.search(r"\[0\.000,\s*0\.658\]", ozet), (
        "Yanlis pozitif oranının Wilson araligi ozette yazili olmali"
    )


def test_ozet_gurultu_icinde_kalan_senaryoyu_isaretliyor(ozet, satirlar, v00):
    """Gurultunun icinde kalan bir senaryo 'bulgu' gibi sunulmamali."""
    c2 = satirlar["C2 seed7"]
    esik = {a: abs(float(c2[a]) - v00[a]) for a in ("mAP50", "precision", "recall")}
    d6b = satirlar["D6b"]
    icinde = all(
        abs(float(d6b[a]) - v00[a]) <= esik[a] for a in ("precision", "recall")
    )
    assert icinde, "D6b artik gurultunun icinde degil; ozet guncellenmeli"
    d6b_satiri = next(s for s in ozet.splitlines() if s.startswith("| D6b "))
    assert "hayir" in d6b_satiri.lower(), (
        f"D6b satiri gurultu icinde oldugunu soylemiyor: {d6b_satiri}"
    )


def test_acik_eksikler_listesi_bos_degil(ozet):
    """Kapatilmamis eksikler gorunur kalmali; liste bosalirsa gerekce gerekir."""
    bolum = ozet[ozet.index("Bilinen ve kapatilmamis eksikler"):]
    acik = [s for s in bolum.splitlines() if s.startswith("|") and "acik" in s]
    assert acik, "Eksikler tablosunda hicbir 'acik' madde yok - gercekten oyle mi?"

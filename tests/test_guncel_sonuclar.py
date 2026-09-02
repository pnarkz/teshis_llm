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


def _kontrol_satirlari(satirlar: dict) -> list[dict]:
    """v00 ile ayni protokolde, yalnizca seed'i farkli olan kosular."""
    return [
        s for ad, s in satirlar.items()
        if ad.startswith("C") and ad[1:2].isdigit()
        and s["weights_path"].endswith("best.pt")
    ]


def test_ozetteki_gurultu_esigi_tum_kontrollerden_geliyor(ozet, satirlar, v00):
    """Esik, TEK bir kontrolden degil, kontrollerin tamamindan gelmeli.

    n=1 esigi gurultuyu ciddi bicimde kucuk gosteriyordu: uc kontrole
    cikildiginda recall esigi 3.8 kat, mAP50 esigi 4 kat buyudu ve bes iddia
    zayifladi. Ozet eski esigi tasirsa okuyucu gecersiz bir tabloya bakar.
    """
    kontroller = _kontrol_satirlari(satirlar)
    if not kontroller:
        pytest.skip("kontrol kosusu yok")
    for alan in ("mAP50", "precision", "recall"):
        esik = max(abs(float(s[alan]) - v00[alan]) for s in kontroller)
        assert f"{esik:.4f}" in ozet, (
            f"Ozetteki {alan} esigi ({esik:.4f}) {len(kontroller)} kontrol "
            "kosusundan hesaplanandan farkli"
        )


def test_ozet_kac_kontrolden_hesaplandigini_soyluyor(ozet, satirlar):
    """Okuyucu esigin kac gozleme dayandigini bilmelidir."""
    n = len(_kontrol_satirlari(satirlar))
    if not n:
        pytest.skip("kontrol kosusu yok")
    sayi_adi = {1: "bir", 2: "iki", 3: "uc", 4: "dort"}.get(n, str(n))
    assert sayi_adi in ozet.lower() or f"n={n}" in ozet, (
        f"Ozet, esigin {n} kontrol kosusundan geldigini soylemiyor"
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
    tablo = ozet[:ozet.index("Bilinen ve kapatilmamis eksikler")]
    d6b_satiri = next(
        s for s in tablo.splitlines()
        if s.startswith("|") and s.split("|")[1].strip().strip("*") == "D6b"
    )
    assert ("hayir" in d6b_satiri.lower() or "hicbiri" in d6b_satiri.lower()), (
        f"D6b satiri gurultu icinde oldugunu soylemiyor: {d6b_satiri}"
    )


def test_acik_eksikler_listesi_bos_degil(ozet):
    """Kapatilmamis eksikler gorunur kalmali; liste bosalirsa gerekce gerekir."""
    bolum = ozet[ozet.index("Bilinen ve kapatilmamis eksikler"):]
    acik = [s for s in bolum.splitlines() if s.startswith("|") and "acik" in s]
    assert acik, "Eksikler tablosunda hicbir 'acik' madde yok - gercekten oyle mi?"

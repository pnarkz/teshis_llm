"""Iki kosu birbiriyle karsilastirilabilir mi? TEK kaynak.

Neden var
---------
Butun kosular sessizce `v00_saglikli` ile karsilastiriliyordu. Bu, projenin
kendi metodolojisine aykiriydi ve somut bir yanlis pozitif uretiyordu:

    v00_saglikli last_pt  ->  "guclu bozulma kaniti" (mAP50, precision, recall)

Oysa o kosu HICBIR bozulma icermez. Tek farki checkpoint secimidir
(best.pt yerine last.pt). Ayni sekilde `v00n` - saglikli bir yolo26n
referansi - "guclu bozulma" olarak etiketleniyordu; cunku farkli bir
baslangic modelinin sonucu, main_model referansiyla tartiliyordu.

Ajan araclarinda (teshis/ajan/araclar.py) bu filtre zaten vardi; demo
katmaninda yoktu. Ayni kural iki yerde yasayinca biri geride kaldi - bu
projede tekrarlayan hata oruntusu tam olarak budur.

Kural
-----
Iki kosu, ancak dort KIMLIK alani da ayni ise ayni olcekte durur:

    baslangic modeli | degerlendirme kumesi | cikarim cozunurlugu | checkpoint

Bu alanlardan biri farkliysa fark, bozulmanin degil o alanin etkisini
tasir. Tek istisna EŞLENİK OLCUM'dur: ayni agirlik dosyasi farkli bir
cikarim ayariyla yeniden degerlendirilmisse egitim rastgeleligi hic devreye
girmez, fark tamamen o ayarindir (orn. E4: v00'in agirliklari 512 px'te).

Referans ve gurultu tabani
--------------------------
Her grubun kendi referansi ve kendi kontrol kosulari vardir. Kontrolu
olmayan bir grupta esik hesaplanamaz; bu durumda "esik yok" denir, baska
grubun esigi ODUNC ALINMAZ.
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path
from typing import Any, NamedTuple

KOK = Path(__file__).resolve().parents[2]
RESULTS_CSV = KOK / "results.csv"

METRIKLER = ("mAP50", "mAP50_95", "precision", "recall")


class Kimlik(NamedTuple):
    """Bir kosuyu karsilastirma olceginde konumlandiran dort alan."""

    model: str
    degerlendirme_seti: str
    imgsz_eval: str
    checkpoint: str

    def etiket(self) -> str:
        return (f"{self.model} | {self.degerlendirme_seti} | "
                f"{self.imgsz_eval} px | {self.checkpoint}.pt")


@functools.lru_cache(maxsize=1)
def _defter() -> dict[str, dict[str, str]]:
    with RESULTS_CSV.open(encoding="utf-8") as f:
        return {s["scenario"]: s for s in csv.DictReader(f)}


def _checkpoint(satir: dict[str, str]) -> str:
    return "last" if satir.get("weights_path", "").endswith("last.pt") else "best"


def kimlik(senaryo: str) -> Kimlik | None:
    satir = _defter().get(senaryo)
    if satir is None:
        return None
    return Kimlik(
        model=satir.get("model", "?"),
        degerlendirme_seti=satir.get("evaluation_set", "?"),
        imgsz_eval=satir.get("imgsz_eval", "?"),
        checkpoint=_checkpoint(satir),
    )


def bozulmasiz_mi(senaryo: str) -> bool:
    """Kosu hicbir kasitli bozulma icermiyor mu?

    Adlandirma kurali (docs/MIMARI.md): saglikli referanslar `v00` ile,
    kontrol kosulari `C` + rakam ile baslar. Yeni bir kontrol eklendiginde
    liste kendiliginden buyur.
    """
    return senaryo.startswith("v00") or (
        senaryo.startswith("C") and senaryo[1:2].isdigit()
    )


def referans_senaryo(senaryo: str) -> str | None:
    """Bu kosunun AYNI olcekteki saglikli referansi.

    Her grubun referansi, o gruptaki `v00` ile baslayan kosudur. Boyle bir
    kosu yoksa grubun referansi yoktur ve fark hesaplanmamalidir.
    """
    k = kimlik(senaryo)
    if k is None:
        return None
    adaylar = [
        ad for ad in _defter()
        if ad.startswith("v00") and kimlik(ad) == k
    ]
    return sorted(adaylar)[0] if adaylar else None


def kontrol_kosulari(senaryo: str) -> list[str]:
    """Bu kosunun grubundaki, referans DISINDAKI bozulmasiz kosular.

    Gurultu esigi yalnizca bunlardan hesaplanir. Bos donerse o grupta esik
    yoktur - baska grubunki kullanilamaz.

    Degerlendirilen kosunun KENDISI listeye girmez. Bir kontrol kosusu
    incelenirken kendi sapmasi esigi tanimlarsa, farki her zaman esigin tam
    sinirinda cikar ve olcut anlamini yitirir; gurultu.py'deki
    ``haric_run_id`` ayni nedenle vardir.
    """
    k = kimlik(senaryo)
    ref = referans_senaryo(senaryo)
    if k is None:
        return []
    return sorted(
        ad for ad in _defter()
        if bozulmasiz_mi(ad) and ad not in (ref, senaryo) and kimlik(ad) == k
    )


def _eslenik_mi(senaryo: str, aday_ref: str) -> bool:
    """Ayni agirlik dosyasi, farkli cikarim ayari mi?"""
    a, b = _defter().get(senaryo, {}), _defter().get(aday_ref, {})
    return bool(a.get("weights_path")) and a.get("weights_path") == b.get("weights_path")


def _eslenik_referans(senaryo: str) -> str | None:
    for ad in sorted(_defter()):
        if ad != senaryo and ad.startswith("v00") and _eslenik_mi(senaryo, ad):
            return ad
    return None


# Grubunda saglikli referans olmayan kosular icin: eksik olan olcum nedir?
# Bunu yazmak "karsilastirilamaz" demekten daha faydalidir - sonraki adimi
# gosterir.
def eksik_olcum(senaryo: str) -> str:
    k = kimlik(senaryo)
    if k is None:
        return "Bu koşu defterde yok."
    return (
        f"Bu ölçekte sağlıklı referans yok. Gereken koşu: {k.model} ile "
        f"{k.degerlendirme_seti} üzerinde, {k.imgsz_eval} px'te, "
        f"{k.checkpoint}.pt checkpoint'iyle bozulmasız bir eğitim."
    )


def karsilastirma(senaryo: str) -> dict[str, Any]:
    """Bu kosu neyle, hangi gerekceyle karsilastirilir?

    Donen ``tur`` uc degerden biridir:

    - ``ayni_olcek``: dort kimlik alani da referansla ayni.
    - ``eslenik``: ayni agirliklar, tek degisen cikarim ayari. Fark
      tamamen o ayarindir; egitim rastgeleligi devrede degildir.
    - ``yok``: bu olcekte saglikli referans yok; fark hesaplanmamalidir.
    """
    k = kimlik(senaryo)
    if k is None:
        return {"tur": "yok", "referans": None, "kimlik": None,
                "aciklama": "Bu koşu defterde bulunamadı.", "kontroller": []}

    ref = referans_senaryo(senaryo)
    if ref is not None:
        kontroller = kontrol_kosulari(senaryo)
        if senaryo == ref:
            aciklama = "Bu koşu, kendi ölçeğinin sağlıklı referansıdır."
        else:
            aciklama = (
                f"Referans: {ref}. Dört kimlik alanı da aynı; fark yalnızca "
                "senaryonun değiştirdiği şeyi taşır."
            )
        if not kontroller and senaryo != ref:
            aciklama += (
                " Ancak bu ölçekte kontrol koşusu yok, bu yüzden gürültü "
                "eşiği hesaplanamıyor."
            )
        return {"tur": "ayni_olcek", "referans": ref, "kimlik": k,
                "aciklama": aciklama, "kontroller": kontroller}

    eslenik = _eslenik_referans(senaryo)
    if eslenik is not None:
        return {
            "tur": "eslenik", "referans": eslenik, "kimlik": k,
            "aciklama": (
                f"Eşlenik ölçüm: {eslenik} ile TAM OLARAK aynı ağırlık "
                "dosyası, tek değişen çıkarım ayarı. Eğitim rastgeleliği "
                "devrede olmadığı için fark tamamen bu ayarındır; eğitim "
                "gürültüsü eşiği burada geçerli değildir."
            ),
            "kontroller": [],
        }

    return {"tur": "yok", "referans": None, "kimlik": k,
            "aciklama": eksik_olcum(senaryo), "kontroller": []}


@functools.lru_cache(maxsize=64)
def esikler(senaryo: str) -> dict[str, float | None]:
    """Bu kosunun grubundaki kontrol kosularindan olculen gurultu esikleri.

    Esik, kontrollerin referanstan en cok saptigi mutlak farktir. Kontrol
    yoksa None doner ve "esik yok" denir.
    """
    karsi = karsilastirma(senaryo)
    ref = karsi["referans"]
    kontroller = karsi["kontroller"]
    if ref is None or not kontroller:
        return {m: None for m in METRIKLER}
    r = _defter()[ref]
    return {
        m: max(abs(float(_defter()[c][m]) - float(r[m])) for c in kontroller)
        for m in METRIKLER
    }

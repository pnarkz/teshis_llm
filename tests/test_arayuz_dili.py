"""Arayüzde görünen metinler düzgün Türkçe mi?

Sunumda ekrana yansıyacak her cümle Türkçe karakterle yazılmalıdır. Proje
boyunca ASCII Türkçe ("gurultu", "kosu") kullanıldı çünkü geliştirme
araçlarının konsol kodlaması bozuktu; ancak bu yalnızca DOCSTRING ve
YORUMLAR için kabul edilebilir. Kullanıcıya gösterilen dizeler öyle kalırsa
konsol özensiz görünür.

Bu test iki dosya kümesini tarar: demo/ altındaki her şey ve arayüzde
gösterilen metin üreten analiz modülleri.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Arayüzde metin üreten modüller.
HEDEFLER = sorted(ROOT.glob("demo/**/*.py")) + [
    ROOT / "teshis/degerlendirme/senaryo_ozeti.py",
    ROOT / "teshis/degerlendirme/gurultu.py",
]

# ASCII Türkçe belirtisi olan kelimeler. Bunlardan biri geçen ve hiç Türkçe
# karakter içermeyen bir CÜMLE, çevrilmemiş demektir.
ISARETLER = (
    "gurultu", "olcum", "kosu", "icin", "degil", "yalnizca", "saglikli",
    "kanit", "teshis", "sinirlama", "gorunur", "uretil", "bicimde",
    "olculur", "kirilim", "esigi", "guclu", "zayif",
)
TURKCE = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
DIZE = re.compile(r'["\']([^"\'\\n]{20,})["\']')
# Kod gibi görünen dizeler (tanımlayıcı, yol, anahtar) kapsam dışı.
KOD = re.compile(r"[_\[\]{}]|/|\.py|\.json")


def _cumleler(yol: Path):
    """Dosyadaki, docstring ve yorum DIŞINDAKİ cümle benzeri dizeler."""
    docstring = False
    for i, satir in enumerate(yol.read_text(encoding="utf-8").splitlines(), 1):
        if satir.lstrip().startswith("#"):
            continue
        if satir.count('"""') % 2 == 1:
            docstring = not docstring
            continue
        if docstring:
            continue
        for m in DIZE.finditer(satir):
            s = m.group(1)
            if s.count(" ") >= 3 and not KOD.search(s):
                yield i, s


@pytest.mark.parametrize("yol", HEDEFLER, ids=lambda p: p.name)
def test_arayuz_metinleri_turkce(yol: Path):
    cevrilmemis = [
        f"{yol.name}:{i}: {s[:70]}"
        for i, s in _cumleler(yol)
        if any(k in s.lower() for k in ISARETLER) and not TURKCE.search(s)
    ]
    assert not cevrilmemis, (
        "Arayüzde ASCII Türkçe kalmış:\n" + "\n".join(cevrilmemis)
    )


def test_sozluk_anahtarlari_ascii_kaldi():
    """Çeviri sırasında kod tanımlayıcıları BOZULMAMALI.

    GERÇEK HATA: ilk çeviri denemesi kaba bir kelime değiştirme kullandı ve
    sözlük anahtarlarını da çevirdi — skor['teshis'] -> skor['teşhis'].
    Bu çalışma zamanında KeyError üretecekti ve testler yakalamayabilirdi,
    çünkü o kod yolu yalnızca render sırasında çalışıyor.
    """
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    from bolumler.genel_bakis import _ajan_skorlari, _guc_dagilimi
    from data_loader import ajan_kaydi, load_results

    skor = _ajan_skorlari(ajan_kaydi())
    assert set(skor) == {"teshis", "teshis_tespit", "kanit", "sinir", "rubrik"}, (
        f"Skor sözlüğünün anahtarları değişmiş: {sorted(skor)}"
    )

    # "Baseline" results.csv'de yer almaz (fine-tune edilmemiş model), bu
    # yüzden kanıt gücü hesaplanamaz ve dağılıma girmez. Beklenen sayı bu
    # kadar eksiktir; toplamın SIFIR olmaması ve seviye anahtarlarının
    # eşleşmesi asıl kontroldür.
    dagilim = _guc_dagilimi(load_results())
    defterde = sum(
        1 for a in load_results()["scenario"] if str(a) != "Baseline"
    )
    assert dagilim["koşu"].sum() == defterde, (
        f"Kanıt gücü dağılımı {dagilim['koşu'].sum()} koşu sayıyor, "
        f"defterde {defterde} var; seviye anahtarları eşleşmiyor olabilir"
    )


def test_senaryo_ozeti_metinleri_turkce():
    """Senaryo sayfasının türetilen metinleri de arayüzde görünür."""
    from teshis.degerlendirme.senaryo_ozeti import kanit_gucu, ne_sabit_kaldi

    for parca in ne_sabit_kaldi("D4"):
        if any(k in parca.lower() for k in ISARETLER):
            assert TURKCE.search(parca), f"ASCII kalmış: {parca}"
    assert TURKCE.search(kanit_gucu("D6b")["aciklama"])

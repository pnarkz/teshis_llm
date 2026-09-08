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
HEDEFLER = [p for p in sorted(ROOT.glob("demo/**/*.py"))
            if "__pycache__" not in p.parts] + [
    ROOT / "teshis/degerlendirme/senaryo_ozeti.py",
    ROOT / "teshis/degerlendirme/gurultu.py",
    ROOT / "teshis/degerlendirme/karsilastirilabilirlik.py",
]

# --- Tespit yöntemi ---------------------------------------------------------
#
# İlk sürüm sabit bir işaret kelimesi listesi kullanıyordu ve KISMEN çevrilmiş
# cümleleri kaçırıyordu: "sağlıklı bir referans egitilir" cümlesinde Türkçe
# karakter VAR (sağlıklı), dolayısıyla temiz sayılıyordu — oysa "egitilir"
# çevrilmemişti. Bu yüzden tasarım sayfasında 27 satır gözden kaçtı.
#
# Yeni yöntem sözlüğü kendisi kurar: dosyalarda DOĞRU yazılmış her Türkçe
# kelimenin ASCII karşılığı çıkarılır ("değiştirildi" -> "degistirildi"), sonra
# aynı metinlerde o ASCII biçimleri aranır. Böylece liste elle bakılmaz;
# projede kullanılan kelime dağarcığıyla birlikte büyür.

TURKCE = re.compile("[çğıöşüÇĞİÖŞÜ]")
KATLA = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
DIZE = re.compile(r'"((?:[^"\\]|\\.)*)"')
YER_TUTUCU = re.compile(r"\{[^}]*\}")
# Kod gibi görünen belirteçler (dosya adı, tanımlayıcı, yol) kapsam dışı.
KOD_BELIRTECI = re.compile(r"\S*[_./\\]\S*")
# Tam kelime: iki yanında da harf olmamalı. "aracıdır" içindeki "arac"
# yakalanmamalı.
KELIME = re.compile(r"(?<![A-Za-zçğıöşüÇĞİÖŞÜ])([A-Za-z]{4,})(?![A-Za-zçğıöşüÇĞİÖŞÜ])")

# Ajana gönderilen araç açıklamaları ASCII kalır: onlar arayüz değil, prompt.
MUAF = {("gurultu.py", "BAND_ACIKLAMASI")}


def _ui_dizeleri(yol: Path):
    """Dosyadaki, docstring ve yorum DIŞINDAKİ cümle benzeri dizeler."""
    docstring = False
    muaf_blok = False
    for i, satir in enumerate(yol.read_text(encoding="utf-8").splitlines(), 1):
        t = satir.strip()
        if t.startswith("#"):
            continue
        if t.startswith('"""') and t.endswith('"""') and len(t) > 6:
            continue                                  # tek satırlık docstring
        if satir.count('"""') % 2 == 1:
            docstring = not docstring
            continue
        if docstring:
            continue
        if any(yol.name == d and t.startswith(a) for d, a in MUAF):
            muaf_blok = True
        elif t and not t.startswith('"'):
            muaf_blok = False
        if muaf_blok:
            continue
        for m in DIZE.finditer(satir):
            metin = YER_TUTUCU.sub(" ", m.group(1))
            metin = KOD_BELIRTECI.sub(" ", metin)
            if len(metin) >= 15 and metin.count(" ") >= 2:
                yield i, metin


def _sozluk() -> set[str]:
    """Projede DOĞRU yazılmış Türkçe kelimelerin ASCII karşılıkları."""
    kelimeler = set()
    for yol in HEDEFLER:
        for _, metin in _ui_dizeleri(yol):
            for k in re.findall(r"[A-Za-zçğıöşüÇĞİÖŞÜ]{4,}", metin):
                if TURKCE.search(k):
                    kelimeler.add(k.translate(KATLA).lower())
    return kelimeler


@pytest.mark.parametrize("yol", HEDEFLER, ids=lambda p: p.name)
def test_arayuz_metinleri_turkce(yol: Path):
    sozluk = _sozluk()
    assert len(sozluk) > 100, "sözlük kurulamadı; tespit yöntemi bozulmuş olabilir"
    cevrilmemis = []
    for i, metin in _ui_dizeleri(yol):
        kotu = sorted({k for k in KELIME.findall(metin) if k.lower() in sozluk})
        if kotu:
            cevrilmemis.append(f"{yol.name}:{i}: {','.join(kotu)}  ::  {metin[:60]}")
    assert not cevrilmemis, (
        "Arayüzde ASCII Türkçe kalmış:" + chr(10) + chr(10).join(cevrilmemis)
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

    # Dağılım artık YALNIZCA derecelendirilebilen koşuları sayar: kendi
    # ölçeğinde hem referansı hem gürültü eşiği olanlar. Kontrol koşuları,
    # referanslar ve eşiği olmayanlar dışarıda kalır — girselerdi sağlıklı
    # bir koşu "güçlü bozulma kanıtı" diye sayılırdı.
    import stil
    from teshis.degerlendirme.senaryo_ozeti import kanit_gucu

    sonuclar = load_results()
    dagilim = _guc_dagilimi(sonuclar)
    beklenen = sum(
        1 for a in sonuclar["scenario"]
        if kanit_gucu(str(a))["seviye"] in stil.DERECELENDIRILEN
    )
    assert dagilim["koşu"].sum() == beklenen, (
        f"Kanıt gücü dağılımı {dagilim['koşu'].sum()} koşu sayıyor, "
        f"derecelendirilebilen {beklenen} var; seviye anahtarları "
        "eşleşmiyor olabilir"
    )
    assert dagilim["koşu"].sum() > 0, "hiçbir koşu derecelendirilmiyor"
    assert set(dagilim.index) == {stil.seviye_adi(s) for s in stil.DERECELENDIRILEN}


def test_senaryo_ozeti_metinleri_turkce():
    """Senaryo sayfasının türetilen metinleri de arayüzde görünür."""
    from teshis.degerlendirme.senaryo_ozeti import kanit_gucu, ne_sabit_kaldi

    sozluk = _sozluk()
    for ham in ne_sabit_kaldi("D4"):
        parca = KOD_BELIRTECI.sub(" ", ham)
        kotu = [k for k in KELIME.findall(parca) if k.lower() in sozluk]
        assert not kotu, f"ASCII kalmış: {parca} ({kotu})"
    assert TURKCE.search(kanit_gucu("D6b")["aciklama"])

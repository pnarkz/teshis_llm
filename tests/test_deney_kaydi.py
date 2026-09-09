"""Yeni deney duzeni: korluk, izlenebilirlik, tekillestirme, devam edebilirlik.

Bu testler yalnizca "mevcut ortamda geciyor mu" diye bakmaz; her biri
kapatilan somut bir hatayi YENIDEN URETIR ve duzeltmenin onu gercekten
yakaladigini dogrular. Sahte gozlem dosyalari gecici dizinde uretilir, gercek
deney kayitlarina dokunulmaz.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ajan_deney  # noqa: E402
import ajan_deney_puanla as puanla_modulu  # noqa: E402


# --- Yardimci: sahte deney kurar --------------------------------------------

def _deney_kur(kok: Path, gozlemler: list[dict], anahtar: dict) -> Path:
    dizin = kok / "deney"
    (dizin / "gozlemler").mkdir(parents=True)
    (dizin / "degerlendirme").mkdir()
    (dizin / "deney.json").write_text(json.dumps({"deney_id": "test"}),
                                      encoding="utf-8")
    (dizin / "degerlendirme/cevap_anahtari.json").write_text(
        json.dumps(anahtar, ensure_ascii=False), encoding="utf-8")
    for g in gozlemler:
        yol = dizin / "gozlemler" / g["_ad"]
        yol.write_text(json.dumps({k: v for k, v in g.items() if k != "_ad"},
                                  ensure_ascii=False), encoding="utf-8")
    return dizin


def _gozlem(ad, kosu, teshis="eksik_etiket", ozet="aaa", sira=1, kuru=False):
    return {
        "_ad": ad, "kosu_id": kosu, "gozlem_sirasi": sira,
        "kuru_calistirma": kuru, "uretim_ozeti": ozet,
        "cevap": {"run_id": kosu, "diagnosis": teshis,
                  "evidence": ["x"], "limitations": ["y"],
                  "confidence": "yuksek"},
        "arac_cagrilari": [{"arac": "kosu_metriklerini_getir",
                            "argumanlar": {}, "cevap": {"mAP50": 0.9}}],
    }


ANAHTAR = {"kosu_04": {"expected": "eksik_etiket", "gizli_rol": "bozulma_senaryosu",
                       "gizli_senaryo": "D2b"}}


# --- Korluk -----------------------------------------------------------------

def test_gizli_rol_ajanin_gordugu_hicbir_seye_sizmiyor():
    """Gizli rol ve senaryo YALNIZCA degerlendirme tarafinda durmali.

    Ajanin gordugu uc yuzey var: sistem istemi, arac tanimlari ve arac
    cevaplari. Ucunde de senaryo adi, rol veya beklenen teshis gecmemeli.
    """
    import csv

    from teshis.ajan import ajan as ajan_modulu
    from teshis.ajan import araclar, puanlama, semalar

    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        senaryolar = {r["scenario"] for r in csv.DictReader(f)}
    yasak = {a for a in senaryolar if len(a) > 4}
    yasak |= set(puanlama.SENARYO_BEKLENEN.values())

    yuzeyler = {
        "sistem_istemi": ajan_modulu.SISTEM_TALIMATI,
        "arac_bildirimleri": json.dumps(semalar.ARAC_BILDIRIMLERI,
                                        ensure_ascii=False),
    }
    for arac, arg in (("kosu_metriklerini_getir", {"kosu_id": "kosu_04"}),
                      ("baseline_farkini_getir", {"kosu_id": "kosu_04"}),
                      ("kaynak_bazli_recall_getir", {"kosu_id": "kosu_04"}),
                      ("sinif_karisikligini_getir", {"kosu_id": "kosu_04"})):
        yuzeyler[arac] = json.dumps(getattr(araclar, arac)(**arg),
                                    ensure_ascii=False)

    for ad, metin in yuzeyler.items():
        sizan = sorted(x for x in yasak if x and x in metin)
        assert not sizan, f"{ad} icinde gizli bilgi: {sizan}"


def test_cevap_anahtari_yalnizca_degerlendirme_tarafinda(tmp_path):
    """Anahtar dosyasi gozlem dizinine YAZILMAMALI."""
    dizin = _deney_kur(tmp_path, [_gozlem("kosu_04__g01.json", "kosu_04")],
                       ANAHTAR)
    for yol in (dizin / "gozlemler").glob("*.json"):
        metin = yol.read_text(encoding="utf-8")
        assert "expected" not in metin and "gizli_rol" not in metin, yol.name


# --- Tekillestirme ve tekrarlar ---------------------------------------------

def test_ayni_uretim_iki_dosyadan_iki_kez_sayilmiyor(tmp_path):
    """Ayni uretim kopyalanirsa (arsiv, elle yedek) iki gozlem sanilirdi."""
    ayni = "ozetXYZ"
    dizin = _deney_kur(tmp_path, [
        _gozlem("kosu_04__g01.json", "kosu_04", ozet=ayni),
        _gozlem("kosu_04__g02.json", "kosu_04", ozet=ayni, sira=2),
    ], ANAHTAR)
    gecerli, dislanan = puanla_modulu.gozlemleri_oku(dizin)
    assert len(gecerli) == 1
    assert len(dislanan) == 1 and "zaten sayildi" in dislanan[0]["neden"]


def test_gecerli_tekrarlar_ayri_gozlem_olarak_korunuyor(tmp_path):
    """FARKLI uretimler tekrardir ve birbirinin uzerine yazmamali."""
    dizin = _deney_kur(tmp_path, [
        _gozlem("kosu_04__g01.json", "kosu_04", ozet="a1"),
        _gozlem("kosu_04__g02.json", "kosu_04", ozet="a2", sira=2),
    ], ANAHTAR)
    gecerli, dislanan = puanla_modulu.gozlemleri_oku(dizin)
    assert len(gecerli) == 2, "iki gecerli tekrar da korunmali"
    assert not dislanan
    assert {g["gozlem_sirasi"] for g in gecerli} == {1, 2}

    sonuc = puanla_modulu.puanla(dizin)
    assert len(sonuc["runs"]) == 2
    assert len({s["gozlem"] for s in sonuc["runs"]}) == 2, (
        "iki gozlem ayni dosyaya isaret ediyor"
    )


def test_gecersiz_kayit_silinmiyor_gerekcesiyle_dislaniyor(tmp_path):
    """Yarim kalmis uretim de deneyin gecmisinin parcasidir."""
    yarim = _gozlem("kosu_04__g02.json", "kosu_04", sira=2)
    yarim["cevap"] = {"run_id": "kosu_04"}          # teshis yok
    dizin = _deney_kur(tmp_path, [
        _gozlem("kosu_04__g01.json", "kosu_04"), yarim,
    ], ANAHTAR)
    gecerli, dislanan = puanla_modulu.gozlemleri_oku(dizin)
    assert len(gecerli) == 1
    assert dislanan[0]["dosya"] == "kosu_04__g02.json"
    assert "yarim" in dislanan[0]["neden"]
    assert (dizin / "gozlemler/kosu_04__g02.json").is_file(), "dosya silinmis"


def test_elle_dislama_gerekce_ile_calisiyor(tmp_path):
    dizin = _deney_kur(tmp_path, [_gozlem("kosu_04__g01.json", "kosu_04")],
                       ANAHTAR)
    (dizin / "DISLANANLAR.json").write_text(
        json.dumps({"kosu_04__g01.json": "elle: yanlis model kullanildi"}),
        encoding="utf-8")
    gecerli, dislanan = puanla_modulu.gozlemleri_oku(dizin)
    assert not gecerli
    assert dislanan[0]["neden"].startswith("elle:")


def test_kuru_calistirma_puanlanmiyor(tmp_path):
    dizin = _deney_kur(tmp_path, [
        _gozlem("kosu_04__g01.json", "kosu_04", kuru=True),
    ], ANAHTAR)
    gecerli, dislanan = puanla_modulu.gozlemleri_oku(dizin)
    assert not gecerli and "kuru" in dislanan[0]["neden"]


# --- Raporlama ayrimi -------------------------------------------------------

def test_rol_bazli_oranlar_birlestirilmiyor(tmp_path):
    """Kontrol ve bozulma senaryolari FARKLI soruyu olcer."""
    anahtar = dict(ANAHTAR)
    anahtar["kosu_11"] = {"expected": "anlamli_degisim_yok",
                          "gizli_rol": "kontrol", "gizli_senaryo": "C2 seed7"}
    dizin = _deney_kur(tmp_path, [
        _gozlem("kosu_04__g01.json", "kosu_04", ozet="a"),
        _gozlem("kosu_11__g01.json", "kosu_11",
                teshis="anlamli_degisim_yok", ozet="b"),
    ], anahtar)
    sonuc = puanla_modulu.puanla(dizin)
    assert set(sonuc["rol_bazli"]) == {"bozulma_senaryosu", "kontrol"}
    for rol, d in sonuc["rol_bazli"].items():
        assert d["gozlem"] == 1, rol


def test_yeni_deney_eski_denemeye_karismiyor():
    """Yeni deney AYRI dizinde durmali; eski dosyalar degismemis olmali."""
    assert ajan_deney.DENEYLER != ROOT / "reports/ajan_denemesi"
    assert ajan_deney.DENEYLER.name == "ajan_deneyleri"
    sys.path.insert(0, str(ROOT / "demo"))
    from data_loader import ajan_kaydi

    kayit = ajan_kaydi()
    assert len(kayit["cevaplar"]) == 11, (
        "eski ana deneme 11 kosuluk kalmali; yeni deney ona karismamali"
    )


# --- Izlenebilirlik ---------------------------------------------------------

def test_gozlem_uretim_parametrelerini_tasiyor(tmp_path):
    """Model, arac surumu, Git commit'i ve zaman damgasi kayitta olmali.

    Eski denemede bunlarin hicbiri yoktu ve "hangi cevap hangi kod haliyle
    uretildi" sorusu Git arkeolojisiyle bile kesin cevaplanamadi.
    """
    kayit = ajan_deney.kosuyu_calistir(tmp_path / "d", "kosu_04", 1,
                                       "test-model", kuru=True)
    for alan in ("model", "arac_surumu", "git_commit", "zaman_utc",
                 "uretim_ozeti", "ham_cevap", "cevap", "arac_cagrilari"):
        assert alan in kayit, alan
    assert kayit["arac_cagrilari"], "arac cagri kaydi bos"
    assert all("cevap" in c for c in kayit["arac_cagrilari"]), (
        "her arac cagrisi SNAPSHOT tasimali"
    )


def test_snapshot_araclarin_gercek_ciktisiyla_ayni(tmp_path):
    """Snapshot, deneyde kullanilan arac cevabinin ta kendisi olmali."""
    from teshis.ajan import araclar

    kayit = ajan_deney.kosuyu_calistir(tmp_path / "d", "kosu_04", 1,
                                       "test-model", kuru=True)
    snapshot = {c["arac"]: c["cevap"] for c in kayit["arac_cagrilari"]}
    assert (snapshot["kosu_metriklerini_getir"]
            == araclar.kosu_metriklerini_getir("kosu_04"))
    assert (snapshot["boyut_bazli_recall_getir"]
            == araclar.boyut_bazli_recall_getir("kosu_04"))


def test_ayni_komut_tekrar_calistirilinca_kayit_ezilmiyor(tmp_path):
    dizin = tmp_path / "d"
    ilk = ajan_deney.kosuyu_calistir(dizin, "kosu_04", 1, "m", kuru=True)
    ikinci = ajan_deney.kosuyu_calistir(dizin, "kosu_04", 1, "m", kuru=True)
    assert ilk["zaman_utc"] == ikinci["zaman_utc"], "kayit yeniden yazilmis"


def test_devam_yalnizca_eksik_gozlemleri_kosuyor(tmp_path, monkeypatch):
    """Tamamlanmis gozlemler YENIDEN CAGRILMAMALI - her cagri API maliyeti."""
    monkeypatch.setattr(ajan_deney, "DENEYLER", tmp_path)   # gercek kayitlara dokunma
    dizin = tmp_path / "d"
    ajan_deney.kosuyu_calistir(dizin, "kosu_04", 1, "m", kuru=True)

    cagrilan = []
    gercek = ajan_deney.kosuyu_calistir

    def izle(d, kosu, sira, model, kuru=False):
        yol = ajan_deney._gozlem_yolu(d, kosu, sira)
        if not yol.is_file():
            cagrilan.append((kosu, sira))
        return gercek(d, kosu, sira, model, kuru=kuru)

    monkeypatch.setattr(ajan_deney, "kosuyu_calistir", izle)
    plan = {"kosular": [
        {"kosu_id": "kosu_04", "tekrar": 2, "gizli_rol": "x",
         "gizli_senaryo": "D2b", "beklenen_teshis": "eksik_etiket"},
    ], "toplam_gozlem": 2}
    ajan_deney.deneyi_yurut(dizin.name, plan, "m", kuru=True, devam=True)
    assert cagrilan == [("kosu_04", 2)], (
        f"yalnizca eksik gozlem kosulmaliydi, kosulan: {cagrilan}"
    )


def test_plan_gizli_rolleri_ve_gerekceyi_iceriyor():
    """Deney BASLAMADAN once hangi kosular, hangi rolle, neden."""
    plan = ajan_deney.plan_uret(tekrar=1)
    assert plan["kosu_sayisi"] >= 13
    roller = {k["gizli_rol"] for k in plan["kosular"]}
    assert {"saglikli_referans", "kontrol", "bozulma_senaryosu"} <= roller
    for k in plan["kosular"]:
        assert k["gerekce"] and k["gizli_senaryo"]
    assert plan["tahmini_api_istegi"] > 0


def test_kuru_calistirma_ayri_onek_aliyor():
    """Prova, deney degildir: klasor adindan ayirt edilebilmeli."""
    assert ajan_deney.deney_kimligi(kuru=True).startswith("KURU__")
    assert not ajan_deney.deney_kimligi(kuru=False).startswith("KURU__")


def test_en_yeni_deney_provayi_secmiyor(tmp_path, monkeypatch):
    """En yeni klasor bir prova olabilir; puanlayici onu atlamali."""
    monkeypatch.setattr(puanla_modulu, "DENEYLER", tmp_path)
    for ad in ("20260101T000000Z__aaaa", "KURU__20260909T000000Z__bbbb"):
        (tmp_path / ad).mkdir()
        (tmp_path / ad / "deney.json").write_text("{}", encoding="utf-8")
    secilen = puanla_modulu.en_yeni_deney()
    assert secilen is not None and not secilen.name.startswith("KURU__")


def test_prova_klasorleri_git_disinda():
    """KURU__ klasorleri repoya deney gibi girmemeli."""
    import subprocess

    yol = "reports/ajan_deneyleri/KURU__20260101T000000Z__test/deney.json"
    sonuc = subprocess.run(["git", "check-ignore", "-q", yol],
                           cwd=ROOT, capture_output=True)
    assert sonuc.returncode == 0, f"{yol} .gitignore kapsaminda degil"


def test_dusen_gozlem_sirasi_devamda_dolduruluyor(tmp_path, monkeypatch):
    """g01 duserse g02/g03 uretilse bile o sira BOS kalmamali.

    Gercek kosuda goruldu: kosu_01 g01 gecici bir 503 ile dustu, dosya
    yazilmadi, calisma g02 ile devam etti. Eski mantik "kac gozlem var"
    sayiyordu; --devam sayiya bakip g03'u zaten var goruyor ve kosu KALICI
    olarak 2 tekrarla kaliyordu. Ustelik hicbir yerde hata gorunmuyordu.
    """
    monkeypatch.setattr(ajan_deney, "DENEYLER", tmp_path)
    dizin = tmp_path / "d"
    # g01 dusmus gibi: yalnizca g02 ve g03 diskte.
    ajan_deney.kosuyu_calistir(dizin, "kosu_04", 2, "m", kuru=True)
    ajan_deney.kosuyu_calistir(dizin, "kosu_04", 3, "m", kuru=True)

    kosulan = []
    gercek = ajan_deney.kosuyu_calistir

    def izle(d, kosu, sira, model, kuru=False):
        if not ajan_deney._gozlem_yolu(d, kosu, sira).is_file():
            kosulan.append((kosu, sira))
        return gercek(d, kosu, sira, model, kuru=kuru)

    monkeypatch.setattr(ajan_deney, "kosuyu_calistir", izle)
    plan = {"kosular": [
        {"kosu_id": "kosu_04", "tekrar": 3, "gizli_rol": "x",
         "gizli_senaryo": "D2b", "beklenen_teshis": "eksik_etiket"},
    ], "toplam_gozlem": 3}
    ajan_deney.deneyi_yurut(dizin.name, plan, "m", kuru=True, devam=True)

    assert kosulan == [("kosu_04", 1)], (
        f"eksik sira doldurulmadi; kosulan: {kosulan}")
    assert len(ajan_deney.mevcut_gozlemler(dizin)["kosu_04"]) == 3


def test_uretilemeyen_gozlem_sessizce_gecilmiyor(tmp_path, monkeypatch, capsys):
    """Uretilemeyen gozlem ekrana yazilmali; deney eksik bittigini soylemeli."""
    monkeypatch.setattr(ajan_deney, "DENEYLER", tmp_path)
    monkeypatch.setattr(ajan_deney, "kosuyu_calistir",
                        lambda *a, **k: None)          # her uretim dusuyor
    plan = {"kosular": [
        {"kosu_id": "kosu_04", "tekrar": 2, "gizli_rol": "x",
         "gizli_senaryo": "D2b", "beklenen_teshis": "eksik_etiket"},
    ], "toplam_gozlem": 2}
    ajan_deney.deneyi_yurut("d", plan, "m", kuru=True, devam=False)
    cikti = capsys.readouterr().out
    assert "URETILEMEYEN GOZLEM (2)" in cikti
    assert "kosu_04 g01" in cikti and "kosu_04 g02" in cikti


def test_gecici_sunucu_hatasi_yeniden_deneniyor(monkeypatch):
    """503 'high demand' kota degil, gecici sunucu yogunlugudur.

    Ilk denemede firlatilirsa gozlem kaybolur. Kaybolan gozlem de deneyi
    planlanandan az tekrarla birakir.
    """
    from teshis.ajan import ajan as ajan_modulu

    assert ajan_modulu._gecici_sunucu_hatasi_mi(
        Exception("ServerError: 503 UNAVAILABLE. This model is currently "
                  "experiencing high demand."))
    assert not ajan_modulu._gecici_sunucu_hatasi_mi(
        Exception("429 RESOURCE_EXHAUSTED PerDay quota"))

    monkeypatch.setattr(ajan_modulu.time, "sleep", lambda s: None)
    cagri = {"n": 0}

    class SahteModeller:
        def generate_content(self, **kwargs):
            cagri["n"] += 1
            if cagri["n"] < 3:
                raise Exception("ServerError: 503 UNAVAILABLE high demand")
            return "cevap"

    class SahteClient:
        models = SahteModeller()

    assert ajan_modulu._istek_gonder(SahteClient(), "m", [], None) == "cevap"
    assert cagri["n"] == 3, "gecici hata yeniden denenmedi"


def test_devam_tek_basina_calistir_istemiyor():
    """--devam zaten 'calistir'in bir bicimidir.

    Ayrica --calistir istemek, yarida kalmis bir deneyi surdurmeyi gereksiz
    yere zorlastiriyordu: kullanici --devam veriyor, betik "--calistir verin"
    diyor. Eksik --deney ise mevcut deney kimliklerini listelemeli.
    """
    import subprocess

    sonuc = subprocess.run(
        [sys.executable, "scripts/ajan_deney.py", "--devam", "--tekrar", "3"],
        cwd=ROOT, capture_output=True, text=True)
    hata = sonuc.stderr
    assert "--calistir verin" not in hata, (
        "--devam hala --calistir istiyor")
    assert "--devam icin --deney" in hata
    assert "Mevcut deneyler:" in hata

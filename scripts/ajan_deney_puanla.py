"""Bir deneyi puanlar - uretimden AYRI, gizli cevap anahtarina karsi.

Puanlama neden ayri bir adim
----------------------------
Cevap anahtari ajana hicbir zaman gonderilmez. Uretim ve degerlendirme ayri
proseslerde calisirsa "anahtar kazara isteme sizdi mi?" sorusu kod okunarak
degil, sureç sinirina bakilarak cevaplanir.

Tekillestirme
-------------
Ayni uretim iki dosyada bulunabilir (arsivleme, elle yedek, kopyalama).
Her gozlem `uretim_ozeti` tasir; ayni ozet iki kez sayilmaz. Farkli ozetler
ise GECERLI TEKRARLARDIR ve ayri gozlem olarak korunur.

Raporlama
---------
Ana deney, kontrol kosulari ve tekrarlar AYRI raporlanir. Tek bir "basari
orani" hepsini birlestirmez: kontroller ile bozulma senaryolari farkli
soruyu olcer ve birlestirilmis oran ikisini de yaniltir.

Kullanim:
    python scripts/ajan_deney_puanla.py --deney <id>
    python scripts/ajan_deney_puanla.py            # en yeni deney
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teshis.ajan.puanlama import kosuyu_puanla  # noqa: E402

DENEYLER = ROOT / "reports/ajan_deneyleri"


def en_yeni_deney() -> Path | None:
    """En yeni GERCEK deney; kuru provalar (KURU__) atlanir.

    Prova klasoru en yeni olabilir. Atlanmasaydi "en yeni deneyi puanla"
    komutu sessizce bir provaya bakip "gecerli gozlem yok" derdi.
    """
    adaylar = sorted(d for d in DENEYLER.glob("*")
                     if (d / "deney.json").is_file()
                     and not d.name.startswith("KURU__"))
    return adaylar[-1] if adaylar else None


def gozlemleri_oku(dizin: Path) -> tuple[list[dict], list[dict]]:
    """(gecerli gozlemler, dislanan gozlemler).

    Dislama nedenleri kaydedilir, dosyalar SILINMEZ: yarim kalmis veya
    yerini yenisi almis bir kayit da deneyin gecmisinin parcasidir.
    """
    disla_yolu = dizin / "DISLANANLAR.json"
    elle_dislanan = (json.loads(disla_yolu.read_text(encoding="utf-8"))
                     if disla_yolu.is_file() else {})

    gecerli, dislanan = [], []
    gorulen_ozet: dict[str, str] = {}
    for yol in sorted((dizin / "gozlemler").glob("*.json")):
        k = json.loads(yol.read_text(encoding="utf-8"))
        k["_dosya"] = yol.name
        neden = None
        if yol.name in elle_dislanan:
            neden = elle_dislanan[yol.name]
        elif k.get("kuru_calistirma"):
            neden = "kuru calistirma - model cagrilmadi, teshis yok"
        elif not (k.get("cevap") or {}).get("diagnosis"):
            neden = "cevapta teshis alani yok (yarim kalmis uretim)"
        elif k.get("uretim_ozeti") in gorulen_ozet:
            neden = (f"ayni uretim zaten sayildi "
                     f"({gorulen_ozet[k['uretim_ozeti']]})")
        if neden:
            dislanan.append({"dosya": yol.name, "kosu_id": k.get("kosu_id"),
                             "neden": neden})
            continue
        gorulen_ozet[k["uretim_ozeti"]] = yol.name
        gecerli.append(k)
    return gecerli, dislanan


def puanla(dizin: Path) -> dict[str, Any]:
    anahtar_yolu = dizin / "degerlendirme/cevap_anahtari.json"
    anahtar = json.loads(anahtar_yolu.read_text(encoding="utf-8"))
    gecerli, dislanan = gozlemleri_oku(dizin)

    if not gecerli:
        return {"gozlem": 0, "dislanan": dislanan,
                "not": "puanlanacak gecerli gozlem yok"}

    # Puanlama GOZLEM basinadir, kosu basina degil.
    #
    # Kosu kimligine gore puanlamak (paketi_puanla'nin yaptigi) ayni kosunun
    # birden fazla gecerli tekrarini tek satira indirger: son tekrar
    # oncekilerin uzerine yazilir ve tekrar sayisi raporda kaybolur. Tekrarlar
    # bu deneyin ana olcusudur - her biri ayri satir olmalidir.
    #
    # Cevap anahtari da yalnizca gozlemi olan kosularla sinirlidir; olculmemis
    # kosular "missing" sayilip 0 alsaydi ortalama asagi cekilirdi.
    satirlar = []
    for kayit in gecerli:
        gizli = anahtar.get(kayit["kosu_id"], {})
        satir = kosuyu_puanla(
            kayit["kosu_id"],
            # hidden_role: tespit-farkindalikli puan icin gizli senaryo adi.
            # Yalnizca DEGERLENDIRME tarafinda; ajanin gordugu hicbir yuzeyde
            # bulunmaz.
            {"expected": gizli.get("expected"),
             "hidden_role": gizli.get("gizli_senaryo")},
            dict(kayit["cevap"], run_id=kayit["kosu_id"]),
        )
        satir["gozlem"] = kayit["_dosya"]
        satir["gozlem_sirasi"] = kayit["gozlem_sirasi"]
        satir["uretim_ozeti"] = kayit["uretim_ozeti"]
        satir["gizli_rol"] = gizli.get("gizli_rol")
        satir["gizli_senaryo"] = gizli.get("gizli_senaryo")
        satirlar.append(satir)

    # Deney TEK PARCA mi: butun gozlemler ayni model, ayni arac surumu ve
    # ayni kod haliyle mi uretildi?
    #
    # Bir deney gunlere yayilabilir (kota), farkli bir hesapla surdurulebilir,
    # arada kod degisebilir. API anahtari onemli degil - model, arac katmani
    # ve commit onemli. Bunlar karisirsa gozlemler ayni kosulda uretilmemis
    # olur ve tek bir oran altinda toplanamaz. Sessiz kalmak yerine yazilir.
    uretim = {
        alan: sorted({str(k.get(alan)) for k in gecerli})
        for alan in ("model", "arac_surumu", "git_commit")
    }
    karisik = {a: v for a, v in uretim.items() if len(v) > 1}

    sonuc: dict[str, Any] = {
        "gozlem": len(satirlar),
        "kosu": len({s["run_id"] for s in satirlar}),
        "uretim_kosullari": uretim,
        "tek_parca_mi": not karisik,
        "runs": satirlar,
        "metric_definition": (
            "diagnosis, evidence ve limitations ayri ayri 0..1 puanlanir; "
            "bu bir pilot rubriktir, bilimsel bir kiyaslama olcusu degildir"
        ),
    }

    # Rol bazli AYRI raporlama.
    roller: dict[str, list] = defaultdict(list)
    for satir in satirlar:
        roller[satir["gizli_rol"] or "bilinmiyor"].append(satir)

    ozet = {}
    for rol, rol_satirlari in roller.items():
        n = len(rol_satirlari)
        ozet[rol] = {
            "gozlem": n,
            # Gozlem sayisi tekrarlari icerir; kac AYRI kosu olculdugu ayri
            # bilgidir ve ikisi karistirilmamalidir.
            "kosu": len({s["run_id"] for s in rol_satirlari}),
            "dogru_teshis": round(
                sum(s["diagnosis_score"] for s in rol_satirlari) / n, 3),
            "tespit_farkindalikli": round(
                sum(s["diagnosis_score_tespit"] for s in rol_satirlari) / n, 3),
        }

    sonuc["rol_bazli"] = ozet
    sonuc["dislanan"] = dislanan
    sonuc["_not"] = (
        "Rol bazli oranlar BIRLESTIRILMEZ: kontrol kosulari 'uydurma yapiyor "
        "mu', bozulma senaryolari 'nedeni bulabiliyor mu' sorusunu olcer. "
        "Tek bir ortalama ikisini de yaniltir. Bu deney, eski "
        "reports/ajan_denemesi sonuclarindan da AYRIDIR."
    )
    if karisik:
        sonuc["_uyari"] = (
            "Gozlemler ayni kosulda uretilmemis; su alanlar deney icinde "
            "degisiyor: "
            + "; ".join(f"{a} -> {', '.join(v)}" for a, v in karisik.items())
            + ". Bu gozlemler tek bir oran altinda toplanmadan once "
              "hangisinin hangi kosulda uretildigi belirtilmelidir."
        )
    return sonuc


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--deney", default=None)
    args = a.parse_args()

    dizin = (DENEYLER / args.deney) if args.deney else en_yeni_deney()
    if dizin is None or not dizin.is_dir():
        raise SystemExit("deney bulunamadi")

    sonuc = puanla(dizin)
    (dizin / "degerlendirme/puan.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"deney: {dizin.name}")
    print(f"gecerli gozlem: {len(sonuc.get('runs', []))}  "
          f"({sonuc.get('kosu', 0)} ayri kosu)  "
          f"dislanan: {len(sonuc.get('dislanan', []))}")
    # IKI puan da yazilir. Yalnizca kati puani yazmak yaniltiyordu: D1 gibi
    # bozulmanin kanitta anlamli iz BIRAKMADIGI kosularda ajan "bozulma
    # saptanmadi" dediginde kati puan 0.0 verir, oysa bu, gordugu kanitla
    # tutarli tek okumadir. Hangisinin kullanilacagi okuyucunun karari.
    for rol, d in (sonuc.get("rol_bazli") or {}).items():
        print(f"  {rol:20} gozlem={d['gozlem']:<3} kosu={d['kosu']:<3} "
              f"kati={d['dogru_teshis']:<6} "
              f"tespit-farkindalikli={d['tespit_farkindalikli']}")
    if sonuc.get("_uyari"):
        print(f"\nUYARI: {sonuc['_uyari']}\n")
    for d in sonuc.get("dislanan", [])[:6]:
        print(f"  DISLANDI {d['dosya']}: {d['neden']}")


if __name__ == "__main__":
    main()

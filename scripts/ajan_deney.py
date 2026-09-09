"""Kor teshis deneyi: planla, calistir, tam metaveriyle kaydet.

Neden yeni bir kosucu
---------------------
Mevcut ana deneme (`reports/ajan_denemesi/`) su sorunlari tasiyor ve
GERIYE DONUK duzeltilemez:

- **Arac cevaplari saklanmamis.** Depodaki 102 arac cagrisinin hicbirinde
  `cevap` alani yok; "ajan neyi gordu" sorusu ancak araclar yeniden
  calistirilarak YAKLASIK olarak cevaplanabiliyor. Araclara sonradan
  gurultu bandi alanlari eklendigi icin bu yaklasiklik gercek bir fark.
- **Uretim parametreleri kayitli degil.** Model surumu, arac surumu, Git
  commit'i, zaman damgasi ve anonim kosu haritasi hicbir yerde yazmiyor;
  hangi cevabin hangi kod haliyle uretildigi ancak Git arkeolojisiyle
  cikarilabiliyor - ve bir kisminda kesin sonuca varilamiyor.
- **Ham model cevabi yok.** Yalnizca ayristirilmis sozluk saklaniyor.
  Ayristirma kurali degisirse eski kayittan yeniden uretilemez.

Bu kosucu ayni hatayi tekrarlamamak icin her seyi kaydeder.

Tasarim kararlari
-----------------
**Degismez deney kimligi.** Her calistirma `reports/ajan_deneyleri/<id>/`
altina yazar; `id` = UTC zaman damgasi + Git kisa sha. Eski dosyalarin
uzerine YAZILMAZ.

**Gozlem birimi.** Ayni `kosu_NN` birden fazla kez calistirilabilir; her
uretim ayri bir GOZLEM'dir ve kendi kimligini alir. Ayni uretimin iki kez
sayilmasini onlemek icin her gozlem, uretimin icerik ozetini (`uretim_ozeti`)
tasir; puanlayici bu ozete gore tekilleştirir.

**Devam edebilirlik.** `--devam` yalnizca EKSIK gozlemleri kosar;
tamamlanmis olanlari yeniden cagirmaz. Ayni komut yanlislikla tekrar
calistirilirsa hicbir kaydi ezmez.

**Korluk.** Anonim kosu haritasi ve gercek roller yalnizca
`degerlendirme/` altina yazilir. Ajanin gordugu hicbir sey (istem, arac
tanimlari, arac cevaplari) bu bilgiyi tasimaz; `tests/test_deney_korlugu.py`
bunu dogrular.

Kullanim
--------
    python scripts/ajan_deney.py --plan                 # yalnizca plani yaz
    python scripts/ajan_deney.py --kuru-calistirma      # API'siz tam prova
    python scripts/ajan_deney.py --calistir             # gercek deney
    python scripts/ajan_deney.py --calistir --devam --deney <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DENEYLER = ROOT / "reports/ajan_deneyleri"


# --- Plan -------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "bilinmiyor"


def _git_temiz_mi() -> bool:
    try:
        cikti = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        return not cikti
    except (OSError, subprocess.CalledProcessError):
        return False


def plan_uret(tekrar: int = 1) -> dict[str, Any]:
    """Hangi kosular, hangi gizli rolle, neden dahil edildi?

    Plan deney BASLAMADAN once yazilir. Boylece "hangi kosular secildi ve
    neden" sorusu sonuclara bakilarak degil, onceden verilmis bir karara
    bakilarak cevaplanir.
    """
    from teshis.ajan import araclar, puanlama
    from teshis.degerlendirme.karsilastirilabilirlik import bozulmasiz_mi

    harita = {"kosu_01": araclar.REFERANS_SENARYO}
    harita.update({k: _senaryo_adi(v) for k, v in
                   araclar.anonim_kosu_haritasi().items()})

    kayitlar = []
    for kosu_id, senaryo in sorted(harita.items()):
        bozulmasiz = bozulmasiz_mi(senaryo)
        if senaryo == araclar.REFERANS_SENARYO:
            rol, gerekce = "saglikli_referans", (
                "Karsilastirma tabani. Ajanin 'bozulma yok' diyebilmesi "
                "gerekir; diyemezse yanlis pozitif uretiyor demektir.")
        elif bozulmasiz:
            rol, gerekce = "kontrol", (
                "Yalnizca seed farkli, hicbir bozulma yok. Projenin en zayif "
                "iddiasini ('ajan sorun uydurmuyor') dogrudan sinar.")
        else:
            rol, gerekce = "bozulma_senaryosu", (
                "Kasitli bir bozulma icerir; ajanin nedeni ayirt edip "
                "edemedigi olculur.")
        kayitlar.append({
            "kosu_id": kosu_id,
            "gizli_rol": rol,
            "gizli_senaryo": senaryo,
            "beklenen_teshis": puanlama.SENARYO_BEKLENEN.get(senaryo),
            "gerekce": gerekce,
            "tekrar": tekrar,
        })

    istek = len(kayitlar) * tekrar * 2.5
    return {
        "kosu_sayisi": len(kayitlar),
        "kosu_basina_tekrar": tekrar,
        "toplam_gozlem": len(kayitlar) * tekrar,
        "tahmini_api_istegi": round(istek),
        "tahmin_gerekcesi": (
            "Her gozlem ~2.5 istek: bir tur arac cagrisi + nihai cevap, "
            "artı ara sira ek tur. Ucretsiz katman 20 istek/gun, 5 istek/dk."
        ),
        "kosular": kayitlar,
    }


def _senaryo_adi(run_id: str) -> str:
    import csv

    with (ROOT / "results.csv").open(encoding="utf-8") as f:
        for satir in csv.DictReader(f):
            if satir["run_id"] == run_id:
                return satir["scenario"]
    return run_id


# --- Deney dizini -----------------------------------------------------------

def deney_kimligi(kuru: bool = False) -> str:
    """Degismez deney kimligi: <UTC zaman>__<git sha>.

    Kuru calistirma AYRI bir onek alir. Prova, deney degildir: API'ye hic
    gidilmez, teshis uretilmez. Onek olmasa provanin klasoru gercek bir
    deneyin klasoruyle ayni bicimde gorunur ve sonradan "bu deney kosuldu mu"
    sorusu dosya adina bakarak cevaplanamaz. `KURU__` klasorleri ayrica
    .gitignore'da: repoya deney gibi girmezler.
    """
    zaman = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    onek = "KURU__" if kuru else ""
    return f"{onek}{zaman}__{_git_commit()[:8]}"


def _uretim_ozeti(cevap: dict, arac_kaydi: list) -> str:
    """Bir uretimin icerik parmak izi.

    Ayni uretim iki farkli dosyaya kopyalanirsa (arsivleme, elle yedek)
    puanlayici onu iki ayri gozlem sanabilir. Ozet, tekillestirmenin
    dosya adina degil ICERIGE dayanmasini saglar.
    """
    govde = json.dumps(
        {"cevap": {k: v for k, v in cevap.items() if not k.startswith("_")},
         "arac": [(c.get("arac"), c.get("argumanlar")) for c in arac_kaydi]},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(govde.encode("utf-8")).hexdigest()[:16]


def _gozlem_yolu(dizin: Path, kosu_id: str, sira: int) -> Path:
    return dizin / "gozlemler" / f"{kosu_id}__g{sira:02d}.json"


def mevcut_gozlemler(dizin: Path) -> dict[str, list[dict]]:
    """Diskteki gozlemler; DISLANANLAR haric."""
    kok = dizin / "gozlemler"
    if not kok.is_dir():
        return {}
    disla = _dislananlar(dizin)
    sonuc: dict[str, list[dict]] = {}
    for yol in sorted(kok.glob("*.json")):
        kayit = json.loads(yol.read_text(encoding="utf-8"))
        if yol.name in disla:
            continue
        sonuc.setdefault(kayit["kosu_id"], []).append(kayit)
    return sonuc


def _dislananlar(dizin: Path) -> dict[str, str]:
    yol = dizin / "DISLANANLAR.json"
    if not yol.is_file():
        return {}
    return json.loads(yol.read_text(encoding="utf-8"))


# --- Calistirma -------------------------------------------------------------

def kosuyu_calistir(dizin: Path, kosu_id: str, sira: int, model: str,
                    kuru: bool = False) -> dict[str, Any] | None:
    """Tek gozlem uretir ve diske yazar. Basarisizsa None doner."""
    hedef = _gozlem_yolu(dizin, kosu_id, sira)
    if hedef.is_file():
        print(f"  {hedef.name}: zaten var, atlaniyor")
        return json.loads(hedef.read_text(encoding="utf-8"))

    from teshis.ajan import araclar

    zaman = datetime.now(timezone.utc).isoformat()
    if kuru:
        # KURU CALISTIRMA: API'ye gidilmez. Araclar gercekten calistirilir -
        # boylece kirilim dosyalari, sema ve arac katmani API harcamadan
        # dogrulanir. Cevap yerine isaretli bir yer tutucu yazilir.
        arac_kaydi = []
        for ad in ("baseline_metriklerini_getir", "kosu_metriklerini_getir",
                   "boyut_bazli_recall_getir", "kaynak_bazli_recall_getir",
                   "sinif_karisikligini_getir"):
            fn = getattr(araclar, ad)
            arg = {} if ad in ("baseline_metriklerini_getir",) else {"kosu_id": kosu_id}
            try:
                sonuc = fn(**arg)
                hata = sonuc.get("hata") if isinstance(sonuc, dict) else None
            except Exception as h:  # noqa: BLE001
                sonuc, hata = {"hata": str(h)}, str(h)
            arac_kaydi.append({"tur": 1, "arac": ad, "argumanlar": arg,
                               "cevap": sonuc, "hata": hata})
        cevap = {"run_id": kosu_id, "_kuru_calistirma": True}
        ham = None
    else:
        from teshis.ajan import ajan as ajan_modulu

        try:
            cevap, arac_kaydi = ajan_modulu.teshis_uret(kosu_id, model=model)
        except ajan_modulu.GunlukKotaBitti:
            # Gunluk kota tek bir gozlemin sorunu degil, calismanin sonudur.
            # Yutulursa kalan her gozlem sirayla denenir, hepsi ayni hatayla
            # duser ve ekrana onlarca ayni satir basilir - hicbiri uretilmez.
            raise
        except Exception as hata:  # noqa: BLE001
            print(f"  {kosu_id} g{sira:02d}: BASARISIZ - "
                  f"{type(hata).__name__}: {str(hata)[:160]}")
            return None
        ham = ajan_modulu.son_ham_cevap()

    kayit = {
        "kosu_id": kosu_id,
        "gozlem_sirasi": sira,
        "zaman_utc": zaman,
        "kuru_calistirma": kuru,
        "model": model,
        "arac_surumu": araclar.arac_surumu(),
        "git_commit": _git_commit(),
        "git_temiz": _git_temiz_mi(),
        # Ham ve ayristirilmis cevap AYRI tutulur.
        "ham_cevap": ham,
        "cevap": cevap,
        "arac_cagrilari": arac_kaydi,
        "uretim_ozeti": _uretim_ozeti(cevap, arac_kaydi),
    }
    hedef.parent.mkdir(parents=True, exist_ok=True)
    hedef.write_text(json.dumps(kayit, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"  {hedef.name}: tamam ({len(arac_kaydi)} arac cagrisi)")
    return kayit


def deneyi_yurut(deney_id: str | None, plan: dict, model: str,
                 kuru: bool, devam: bool) -> Path:
    dizin = DENEYLER / (deney_id or deney_kimligi(kuru))
    dizin.mkdir(parents=True, exist_ok=True)

    ustveri_yolu = dizin / "deney.json"
    if ustveri_yolu.is_file() and not devam:
        raise SystemExit(
            f"{dizin.name} zaten var. Ustune yazmak yerine --devam kullanin "
            "veya deney kimligi vermeyin (yenisi olusturulur)."
        )
    if not ustveri_yolu.is_file():
        ustveri_yolu.write_text(json.dumps({
            "deney_id": dizin.name,
            "baslangic_utc": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "kuru_calistirma": kuru,
            "git_commit": _git_commit(),
            "git_temiz": _git_temiz_mi(),
            "uretim_parametreleri": {
                "automatic_function_calling": "kapali",
                "sicaklik": "saglayici varsayilani (acikca ayarlanmadi)",
                "max_tur": 8,
                "sema": "teshis/ajan/semalar.py::TESHIS_SEMASI",
                "sistem_talimati": "teshis/ajan/ajan.py::SISTEM_TALIMATI",
            },
            "plan": plan,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Anonim harita ve gercek roller YALNIZCA degerlendirme tarafinda.
    deg = dizin / "degerlendirme"
    deg.mkdir(exist_ok=True)
    (deg / "cevap_anahtari.json").write_text(json.dumps(
        {k["kosu_id"]: {"expected": k["beklenen_teshis"],
                        "gizli_senaryo": k["gizli_senaryo"],
                        "gizli_rol": k["gizli_rol"]}
         for k in plan["kosular"]},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # EKSIK SIRALAR doldurulur, tamamlananlarin SAYISINDAN devam edilmez.
    #
    # Onceki hali `range(var + 1, tekrar + 1)` idi: yani "kac gozlem var" sayilip
    # oradan devam ediliyordu. g01 gecici bir sunucu hatasiyla dustugunde dosya
    # yazilmaz; sonraki tur g02 ve g03'u uretir, sayim 2 olur ve --devam
    # `range(3, 4)`e bakip g03'u zaten var gorur. Sonuc: kosu KALICI olarak 2
    # tekrarla kalir ve --devam bunu bir daha asla duzeltmez - ustelik hicbir
    # yerde hata gorunmez. Artik her sira ayri ayri kontrol edilir.
    basarisiz: list[str] = []
    durduran: str | None = None
    try:
        for kayit in plan["kosular"]:
            kosu_id = kayit["kosu_id"]
            for sira in range(1, kayit["tekrar"] + 1):
                if _gozlem_yolu(dizin, kosu_id, sira).is_file():
                    continue
                if kosuyu_calistir(dizin, kosu_id, sira, model,
                                   kuru=kuru) is None:
                    basarisiz.append(f"{kosu_id} g{sira:02d}")
    except KeyboardInterrupt:
        durduran = "elle durduruldu (Ctrl+C)"
    except Exception as hata:  # noqa: BLE001 - kota vb. calismayi bitirir
        durduran = f"{type(hata).__name__}: {str(hata).splitlines()[0][:160]}"

    _kalan_ozeti(dizin, plan, basarisiz, durduran)
    return dizin


def eksik_gozlemler(dizin: Path, plan: dict) -> list[str]:
    """Plana gore diskte olmayan gozlemler."""
    return [f"{k['kosu_id']} g{s:02d}"
            for k in plan["kosular"]
            for s in range(1, k["tekrar"] + 1)
            if not _gozlem_yolu(dizin, k["kosu_id"], s).is_file()]


def _kalan_ozeti(dizin: Path, plan: dict, basarisiz: list[str],
                 durduran: str | None) -> None:
    """Calismanin sonunda NE EKSIK kaldigini ve nasil surdurulecegini yazar.

    Yarim kalmis bir calisma sessizce "bitti" gorunmemeli: deneyin
    planlanandan az gozlemle kaldigi, ancak ustveriye elle bakarak
    anlasilirsa fark edilmeyecegi icin burada acikca soylenir.
    """
    eksik = eksik_gozlemler(dizin, plan)
    if durduran:
        print(f"\nCALISMA YARIDA KESILDI - {durduran}")
    if basarisiz:
        print(f"URETILEMEYEN GOZLEM ({len(basarisiz)}): {', '.join(basarisiz)}")
    if eksik:
        print(f"EKSIK GOZLEM: {len(eksik)}/{plan['toplam_gozlem']}  "
              f"-> {', '.join(eksik[:8])}"
              + (" ..." if len(eksik) > 8 else ""))
        print("Surdurmek icin:\n"
              f"  python scripts/ajan_deney.py --devam "
              f"--tekrar {plan['kosular'][0]['tekrar']} --deney {dizin.name}")
        print("Tamamlanan gozlemler tekrar API'ye GITMEZ.")
    else:
        print(f"\nDeney tamam: {plan['toplam_gozlem']} gozlemin hepsi uretildi.")


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--plan", action="store_true", help="yalnizca plani yaz")
    a.add_argument("--durum", action="store_true",
                   help="deneyin neresinde kalindigini yazar; API'ye gitmez")
    a.add_argument("--kuru-calistirma", action="store_true",
                   help="API'siz tam prova: araclar calisir, model cagrilmaz")
    a.add_argument("--calistir", action="store_true", help="gercek deney")
    a.add_argument("--devam", action="store_true",
                   help="var olan deneyde yalnizca eksik gozlemleri kosar")
    a.add_argument("--deney", default=None, help="var olan deney kimligi")
    a.add_argument("--tekrar", type=int, default=1, help="kosu basina gozlem")
    a.add_argument("--model", default=None)
    args = a.parse_args()

    model = args.model
    if model is None:
        sys.path.insert(0, str(ROOT / "demo"))
        from ajan_katmani import VARSAYILAN_MODEL

        model = VARSAYILAN_MODEL

    plan = plan_uret(args.tekrar)
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if args.durum:
        # Kota bittiginde veya calisma yarida kesildiginde "neredeyim"
        # sorusu API'ye hic gitmeden cevaplanabilmeli.
        adaylar = sorted(d for d in DENEYLER.glob("*")
                         if (d / "deney.json").is_file()
                         and not d.name.startswith("KURU__"))
        secili = (DENEYLER / args.deney) if args.deney else (
            adaylar[-1] if adaylar else None)
        if secili is None or not secili.is_dir():
            raise SystemExit("deney bulunamadi")
        mevcut = mevcut_gozlemler(secili)
        uretilen = sum(len(v) for v in mevcut.values())
        print(f"deney: {secili.name}")
        print(f"uretilen gozlem: {uretilen}/{plan['toplam_gozlem']}")
        for kayit in plan["kosular"]:
            var = sorted(g["gozlem_sirasi"]
                         for g in mevcut.get(kayit["kosu_id"], []))
            eksik = [s for s in range(1, kayit["tekrar"] + 1) if s not in var]
            isaret = "TAM " if not eksik else "eksik"
            print(f"  {isaret} {kayit['kosu_id']}: var={var or '-'} "
                  f"eksik={eksik or '-'}")
        _kalan_ozeti(secili, plan, [], None)
        return

    # --devam zaten "calistir"in bir bicimidir; ayrica --calistir istemek
    # yarida kalmis bir deneyi surdurmeyi gereksiz yere zorlastiriyordu.
    if args.devam and not args.kuru_calistirma:
        args.calistir = True
    if not (args.kuru_calistirma or args.calistir):
        a.error("--plan, --kuru-calistirma, --calistir veya --devam verin")
    if args.devam and not args.deney:
        a.error(
            "--devam icin --deney <kimlik> gerekir. Mevcut deneyler:\n  "
            + "\n  ".join(sorted(
                d.name for d in DENEYLER.glob("*")
                if (d / "deney.json").is_file()
                and not d.name.startswith("KURU__")
            ) or ["(yok)"])
        )

    dizin = deneyi_yurut(args.deney, plan, model,
                         kuru=args.kuru_calistirma, devam=args.devam)
    gozlemler = mevcut_gozlemler(dizin)
    print(json.dumps({
        "deney": dizin.name,
        "kosu": len(gozlemler),
        "gozlem": sum(len(v) for v in gozlemler.values()),
        "beklenen_gozlem": plan["toplam_gozlem"],
        "dizin": str(dizin.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

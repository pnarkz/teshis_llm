"""Sartname bolum 9'daki kanit sozlesmesini uretir: experiments/<kosu>/kanit.json.

Neden gerekli
-------------
Sartname her egitim kosusu icin TEK bir kanit dosyasi ister ve kurali acikca
koyar: **"Tek basina mAP hicbir teshis icin yeterli kanit sayilmaz."**

Bu bilgilerin tamami projede zaten uretiliyor, ancak dort ayri yere dagilmis:

    reports/senaryo_*/d1_metrics.json    genel + sinif metrikleri
    reports/kirilim/<run_id>.json        boyut / kaynak kirilimi, karisiklik
    experiments/<kosu>/results.csv       egitim egrisi
    experiments/<kosu>/run_manifest.json seed, model, imgsz, batch
    veri_surumleri/*/manifest.json       veri surumu sayimlari

Dagilmis olmasinin somut bedeli: bir kosunun kanitini okumak icin hangi
dosyanin nerede oldugunu bilmek gerekiyor ve eksik bir parca sessizce
gozden kaciyor.

Eksigi gizlememe kurali
-----------------------
Bu modul, sozlesmenin karsilanmayan maddelerini SILMEZ; `sozlesme_durumu`
altinda hangi maddenin neden eksik oldugunu yazar. Yarim bir kanit
dosyasinin tam gorunmesi, hic olmamasindan daha tehlikelidir.

Ogrenme orani hakkinda
----------------------
`lr0` alani iki deger tasir: protokolde BEYAN EDILEN ve egitimde GECERLI
olan. Ultralytics `optimizer=auto` iken lr0'i yok sayip kendi degerini secer
(bkz. docs/BULGULAR.md "optimizer=auto tuzagi"), bu yuzden yalnizca beyan
edilen degeri yazmak yaniltici olurdu.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .bootstrap import VAL_DIAGNOSTIC_BBOX_N, wilson_araligi, tabakali_goruntu_bootstrap
from .raporlar import rapor_klasoru

KOK = Path(__file__).resolve().parents[2]
RESULTS_CSV = KOK / "results.csv"
SINIFLAR = ["tasit", "insan", "UAP", "UAI"]

# Ultralytics optimizer=auto iken sectigi deger (E1 kosusunun log'undan).
AUTO_OPTIMIZER_LR = 0.00125
AUTO_OPTIMIZER_AD = "AdamW"


def _oku(yol: Path) -> Any:
    return json.loads(yol.read_text(encoding="utf-8")) if yol.is_file() else None


def _kosu_satiri(run_id: str) -> dict[str, str]:
    with RESULTS_CSV.open(encoding="utf-8") as f:
        for satir in csv.DictReader(f):
            if satir["run_id"] == run_id:
                return satir
    raise KeyError(f"{run_id} results.csv'de yok")


def _egitim_egrisi(kosu_dizini: Path) -> dict[str, Any] | None:
    """En iyi epoch, val loss dip epoch'u ve train/val farkini cikarir."""
    yol = kosu_dizini / "results.csv"
    if not yol.is_file():
        return None
    with yol.open(encoding="utf-8") as f:
        satirlar = [{k.strip(): v for k, v in s.items()} for s in csv.DictReader(f)]
    if not satirlar:
        return None

    def sayi(satir: dict[str, str], anahtar: str) -> float | None:
        try:
            return float(satir[anahtar])
        except (KeyError, TypeError, ValueError):
            return None

    map50 = [(int(float(s["epoch"])), sayi(s, "metrics/mAP50(B)")) for s in satirlar]
    map50 = [(e, v) for e, v in map50 if v is not None]
    val_cls = [(int(float(s["epoch"])), sayi(s, "val/cls_loss")) for s in satirlar]
    val_cls = [(e, v) for e, v in val_cls if v is not None]
    train_cls = [(int(float(s["epoch"])), sayi(s, "train/cls_loss")) for s in satirlar]
    train_cls = [(e, v) for e, v in train_cls if v is not None]

    en_iyi = max(map50, key=lambda p: p[1]) if map50 else (None, None)
    dip = min(val_cls, key=lambda p: p[1]) if val_cls else (None, None)

    # train/val farki: overfitting imzasi. Son epoch'ta train dususe devam
    # ederken val yukseliyorsa fark acilmis demektir.
    fark = None
    if train_cls and val_cls:
        son_t, son_v = train_cls[-1][1], val_cls[-1][1]
        ilk_t, ilk_v = train_cls[0][1], val_cls[0][1]
        fark = {
            "ilk_epoch": {"train_cls_loss": ilk_t, "val_cls_loss": ilk_v,
                          "fark": round(ilk_v - ilk_t, 4)},
            "son_epoch": {"train_cls_loss": son_t, "val_cls_loss": son_v,
                          "fark": round(son_v - son_t, 4)},
            "fark_acildi_mi": (son_v - son_t) > (ilk_v - ilk_t),
        }

    return {
        "epoch_sayisi": len(satirlar),
        "en_iyi_epoch": {"epoch": en_iyi[0], "mAP50": en_iyi[1]},
        "son_epoch_mAP50": map50[-1][1] if map50 else None,
        "val_cls_loss_dip_epoch": {"epoch": dip[0], "val_cls_loss": dip[1]},
        "dip_sonrasi_yukselis": (
            None if dip[0] is None or not val_cls
            else round(val_cls[-1][1] - dip[1], 4)
        ),
        "train_val_farki": fark,
    }


def _sinif_metrikleri(
    metrikler: dict[str, Any], kirilim: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Sinif basina P/R/F1/AP + bbox sayisi + guven araligi."""
    satirlar = []
    kayitlar = (kirilim or {}).get("goruntu_kayitlari")
    for i, ad in enumerate(SINIFLAR):
        p = metrikler["class_precision"][i]
        r = metrikler["class_recall"][i]
        n = VAL_DIAGNOSTIC_BBOX_N.get(ad)
        satir: dict[str, Any] = {
            "sinif": ad,
            "bbox_n": n,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
            "ap50": round(metrikler["class_ap50"][i], 4),
            "ap50_95": round(metrikler["class_ap50_95"][i], 4),
        }
        if kayitlar:
            # Sartnameye uygun yontem: tabakali, goruntu birimli bootstrap.
            #
            # DIKKAT: bu aralik `recall` alanina AIT DEGILDIR. Iki farkli olcum
            # var ve karistirilmalari somut bir hataya yol acti:
            #   - `recall`        : d1_sonuc.py (Ultralytics val, kendi conf esigi)
            #   - `recall_kirilim`: metrikler.py (conf=0.25, IoU=0.5 eslestirme)
            # Ilk surumde Ultralytics recall'u metrikler.py'nin araligiyla
            # eslestirilmisti; sonuc, nokta tahminin KENDI ARALIGININ DISINDA
            # kalmasiydi (tasit: 0.8568, aralik [0.8708, 0.9088]). Aralik artik
            # ait oldugu olcumle birlikte veriliyor.
            boot = tabakali_goruntu_bootstrap(kayitlar, grup=ad, tekrar=1000)
            satir["recall_kirilim"] = round(boot["oran"], 4)
            satir["recall_ga"] = [round(boot["alt"], 4), round(boot["ust"], 4)]
            satir["ga_hangi_olcum"] = "recall_kirilim"
            satir["ga_yontemi"] = "tabakali_goruntu_bootstrap (sartnameye uygun)"
        elif n:
            alt, ust = wilson_araligi(round(r * n), n)
            satir["recall_ga"] = [round(alt, 4), round(ust, 4)]
            satir["ga_hangi_olcum"] = "recall"
            satir["ga_yontemi"] = (
                "wilson (kutu birimli) - SARTNAMEYE UYGUN DEGIL; kutulari bagimsiz "
                "sayar ve araligi ~1.5 kat dar gosterir. Goruntu birimli aralik icin "
                "kirilim olcumu goruntu_kayitlari ile yeniden uretilmelidir."
            )
        satirlar.append(satir)
    return satirlar


def kanit_uret(run_id: str) -> dict[str, Any]:
    """Bir kosu icin sartname bolum 9 kanit sozlesmesini toplar."""
    satir = _kosu_satiri(run_id)
    senaryo = satir["scenario"]
    kosu_dizini = KOK / Path(satir["weights_path"]).parent.parent

    rapor = rapor_klasoru(senaryo)
    metrikler = _oku(rapor / "d1_metrics.json") if rapor else None
    kirilim = _oku(KOK / f"reports/kirilim/{run_id}.json")
    manifest = _oku(kosu_dizini / "run_manifest.json")
    egri = _egitim_egrisi(kosu_dizini)
    galeri = _oku(KOK / f"reports/hata_galerisi_{senaryo}/gallery.json")

    if metrikler is None:
        raise FileNotFoundError(
            f"{run_id} icin d1_metrics.json bulunamadi (rapor klasoru: {rapor}). "
            "Kanit sozlesmesi metrik dosyasi olmadan uretilemez."
        )

    eksikler: list[str] = []
    if kirilim is None:
        eksikler.append(
            "boyut/kaynak kirilimi ve karisiklik matrisi: reports/kirilim/"
            f"{run_id}.json yok. `python -m teshis.degerlendirme.metrikler` ile uretilir."
        )
    elif "goruntu_kayitlari" not in kirilim:
        eksikler.append(
            "goruntu birimli guven araligi: kirilim dosyasi goruntu_kayitlari "
            "icermiyor (eski surumle uretilmis). Guven araliklari sartnameye "
            "uygun olmayan Wilson yontemiyle hesaplandi."
        )
    if egri is None:
        eksikler.append(
            f"egitim egrisi: {kosu_dizini}/results.csv yok. Egitim gerektirmeyen "
            "kosularda (orn. yeniden degerlendirme) beklenen bir durumdur."
        )
    if galeri is None:
        eksikler.append(
            f"hata ornekleri: reports/hata_galerisi_{senaryo}/ yok. "
            "`python -m teshis.degerlendirme.hata_galerisi` ile uretilir."
        )

    beyan_lr = float(satir["lr0"])
    kanit: dict[str, Any] = {
        "format": "kanit_sozlesmesi_v1",
        "run_id": run_id,
        "senaryo": senaryo,
        "kural": "Tek basina mAP hicbir teshis icin yeterli kanit sayilmaz.",
        "degerlendirme_seti": satir["evaluation_set"],
        "genel_metrikler": {
            "mAP50": round(metrikler["mAP50"], 4),
            "mAP50_95": round(metrikler["mAP50_95"], 4),
            "precision": round(metrikler["precision"], 4),
            "recall": round(metrikler["recall"], 4),
        },
        "sinif_metrikleri": _sinif_metrikleri(metrikler, kirilim),
        "yapilandirma": {
            "veri_surumu": satir["data_version"],
            "baslangic_modeli": satir["model"],
            "seed": int(satir["seed"]),
            "imgsz_egitim": int(satir["imgsz_train"]),
            "imgsz_degerlendirme": int(satir["imgsz_eval"]),
            "epochs": int(satir["epochs"]),
            "batch": int(satir["batch"]),
            "lr0_beyan_edilen": beyan_lr,
            "lr0_gecerli": AUTO_OPTIMIZER_LR,
            "lr0_notu": (
                f"optimizer=auto oldugu icin beyan edilen lr0 ({beyan_lr}) "
                f"UYGULANMADI; Ultralytics {AUTO_OPTIMIZER_AD}"
                f"(lr={AUTO_OPTIMIZER_LR}) secti. bkz. docs/BULGULAR.md."
            ),
            "protokol_sapmalari": (manifest or {}).get("protokol_sapmalari") or {},
            "e_senaryo": (manifest or {}).get("e_senaryo"),
        },
        "kapsam": {
            "kare_sayisi": (kirilim or {}).get("goruntu"),
            "benzersiz_kaynak_sayisi": (
                len((kirilim or {}).get("kaynak_recall") or {}) or None
            ),
        },
        "sozlesme_durumu": {
            "tam_mi": not eksikler,
            "eksikler": eksikler,
        },
    }

    if kirilim:
        kanit["boyut_bandi_recall"] = kirilim.get("boyut_bandi_recall")
        kanit["bant_tanimi"] = kirilim.get("bant_tanimi")
        kanit["kaynak_recall"] = kirilim.get("kaynak_recall")
        kanit["karisiklik_matrisi"] = kirilim.get("karisiklik_matrisi")
        # Background kaynakli FN: sinif bagimsiz eslestirmede hic eslesmeyenler.
        karisiklik = kirilim.get("karisiklik_matrisi") or {}
        kanit["background_fn"] = {
            ad: sayimlar.get("bulunamadi", 0) for ad, sayimlar in karisiklik.items()
        }
        kanit["toplam_tahmin"] = kirilim.get("toplam_tahmin")
    if egri:
        kanit["egitim_egrisi"] = egri
    if galeri:
        kanit["hata_ornekleri"] = {
            "kaynak": f"reports/hata_galerisi_{senaryo}/gallery.json",
            "ornek_sayisi": len(galeri),
            "en_kotu_bes": galeri[:5],
        }

    return kanit


def _dizin_sahibi(kosu_dizini: Path) -> str | None:
    """Bir kosu dizinini hangi run_id'nin "sahiplendigini" dondurur.

    Her satir kendi agirligini egitmez: D6a ve E4 gibi yeniden degerlendirme
    kosulari, baska bir kosunun (v00) agirliklarini farkli bir kumede veya
    farkli imgsz ile olcer. Uc satir da ayni weights_path'i gosterir.

    Bunu gozden kacirmak somut bir hataya yol acti: E4'un kanit.json'i v00'un
    dizinine yazilip v00'unkini EZDI ve v00'un kanit dosyasi E4 metriklerini
    tasir hale geldi - dosya adi dogru, icerigi baska bir kosunun.

    Sahiplik kurali: dizini gercekten EGITEN satir (duration_min > 0) sahiptir.
    Sartname zaten "her EGITIM kosusu icin" diyor; yeniden degerlendirmeler
    reports/kanit/ altina yazilir.
    """
    with RESULTS_CSV.open(encoding="utf-8") as f:
        paylasanlar = [
            s for s in csv.DictReader(f)
            if (KOK / Path(s["weights_path"]).parent.parent) == kosu_dizini
        ]
    egitenler = [
        s for s in paylasanlar
        if (s["duration_min"] or "0").strip() not in ("", "0")
    ]
    if len(egitenler) == 1:
        return egitenler[0]["run_id"]
    if len(paylasanlar) == 1:
        return paylasanlar[0]["run_id"]
    return None


def kanit_yolu(run_id: str) -> Path:
    """Kanit dosyasinin yazilacagi yer.

    Egitim kosulari sartnamedeki yeri kullanir: experiments/<kosu>/kanit.json.
    Yeniden degerlendirmeler baskasinin dizinini ezemeyecegi icin
    reports/kanit/<run_id>.json altina yazilir.
    """
    satir = _kosu_satiri(run_id)
    kosu_dizini = KOK / Path(satir["weights_path"]).parent.parent
    if _dizin_sahibi(kosu_dizini) == run_id:
        return kosu_dizini / "kanit.json"
    return KOK / "reports/kanit" / f"{run_id}.json"


def yaz(run_id: str) -> Path:
    kanit = kanit_uret(run_id)
    hedef = kanit_yolu(run_id)
    hedef.parent.mkdir(parents=True, exist_ok=True)
    hedef.write_text(json.dumps(kanit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return hedef


def tum_kosular() -> list[str]:
    with RESULTS_CSV.open(encoding="utf-8") as f:
        return [s["run_id"] for s in csv.DictReader(f)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", help="tek bir kosu; verilmezse tumu")
    args = parser.parse_args()

    hedefler = [args.run_id] if args.run_id else tum_kosular()
    tam = eksik = hata = 0
    for run_id in hedefler:
        try:
            yol = yaz(run_id)
        except (FileNotFoundError, KeyError) as sorun:
            print(f"  ATLANDI  {run_id}: {sorun}")
            hata += 1
            continue
        durum = json.loads(yol.read_text(encoding="utf-8"))["sozlesme_durumu"]
        if durum["tam_mi"]:
            print(f"  TAM      {run_id}")
            tam += 1
        else:
            print(f"  EKSIK    {run_id}: {len(durum['eksikler'])} madde")
            for e in durum["eksikler"]:
                print(f"             - {e.split(':')[0]}")
            eksik += 1
    print(f"\ntam: {tam} | eksik: {eksik} | uretilemedi: {hata}")


if __name__ == "__main__":
    main()

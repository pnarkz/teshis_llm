"""Ajani bir TESHIS SISTEMI olarak siniflandirir: hangi hatayi ne siklikta yapiyor?

Ortalama skor (`mean_score`) tek basina yaniltici olabilir: "bozulmayi uydurdu"
ile "bozulmayi kacirdi" ayni sifiri alir, oysa bir teshis sisteminde bu iki
hata tamamen farkli sonuclar dogurur. Uydurma, olmayan bir sorun icin is
yaptirir; kacirma, gercek bir sorunu gozden kacirtir.

Bu script kosulari uce ayirir ve her grupta ajanin ne yaptigini sayar:

A. **Saf kontrol** — hicbir bozulma yok (Baseline, C2 seed7). Buradaki her
   "sorun var" cevabi UYDURMADIR.
B. **Bozulma var, karakteristik imzasi yok** — bozulma uygulanmis ama kanitta
   kendi imzasini birakmamis (bkz. puanlama.TESPIT_EDILEMEYEN). "Degisim yok"
   da dogru sayilir.
C. **Bozulma var ve tespit edilebilir** — asil teshis gorevi.

Sinir: her grupta ornek sayisi kucuk. Wilson araligi bunu acikca gosterir;
"sifir uydurma" iyi bir isarettir ama ust sinir yuksektir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DENEME = ROOT / "reports/ajan_denemesi"

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from teshis.degerlendirme.bootstrap import wilson_araligi  # noqa: E402

# Hicbir bozulma icermeyen kosular. Baseline ve C2, ajanin YANLIS POZITIF
# oranini olcen tek iki vakadir.
SAF_KONTROL_ROLLERI = {"Baseline", "C2 seed7"}


def _oku(ad: str) -> dict:
    return json.loads((DENEME / ad).read_text(encoding="utf-8"))


def siniflandir() -> dict:
    puan = _oku("llm_score.json")
    anahtar = _oku("answer_key.json")
    R = {r["run_id"]: r for r in puan["runs"]}

    gruplar: dict[str, list[dict]] = {"saf_kontrol": [], "imzasiz": [], "tespit_edilebilir": []}
    for kosu, bilgi in anahtar.items():
        r = R.get(kosu)
        if r is None or r["model_diagnosis"] == "missing":
            continue
        rol = bilgi["hidden_role"]
        if rol in SAF_KONTROL_ROLLERI:
            grup = "saf_kontrol"
        elif not r["tespit_edilebilir"]:
            grup = "imzasiz"
        else:
            grup = "tespit_edilebilir"

        teshis = str(r["model_diagnosis"])
        saglikli_dedi = "saglikli" in teshis.lower() or "tespit_edilmedi" in teshis.lower()
        if grup in ("saf_kontrol", "imzasiz"):
            sonuc = "dogru" if r["diagnosis_score_tespit"] == 1.0 else "uydurdu"
        elif r["diagnosis_score"] == 1.0:
            sonuc = "dogru"
        elif r["diagnosis_score"] == 0.5:
            sonuc = "kismi"
        elif saglikli_dedi:
            sonuc = "kacirdi"
        else:
            sonuc = "yanlis_neden"

        gruplar[grup].append({
            "kosu": kosu, "rol": rol, "ajan_teshisi": teshis, "sonuc": sonuc,
        })

    saf = gruplar["saf_kontrol"]
    uydurma = sum(1 for x in saf if x["sonuc"] == "uydurdu")
    alt, ust = wilson_araligi(uydurma, len(saf)) if saf else (0.0, 1.0)

    tespit = gruplar["tespit_edilebilir"]
    sayim = {s: sum(1 for x in tespit if x["sonuc"] == s)
             for s in ("dogru", "kismi", "yanlis_neden", "kacirdi")}

    return {
        "kaynak": "reports/ajan_denemesi/llm_score.json",
        "gruplar": gruplar,
        "saf_kontrol_ozeti": {
            "n": len(saf),
            "uydurma": uydurma,
            "uydurma_orani": round(uydurma / len(saf), 3) if saf else None,
            "wilson_95": [round(alt, 3), round(ust, 3)],
            "yorum": (
                "Sifir uydurma iyi bir isaret, ancak yalnizca iki saf kontrol var; "
                "ust sinir yuksek kaliyor. Kesin konusmak icin daha fazla kontrol "
                "kosusu (farkli seed'lerle v00 tekrarlari) gerekir."
            ),
        },
        "tespit_edilebilir_ozeti": {
            "n": len(tespit),
            **sayim,
            "dogru_neden_puani": sayim["dogru"] + 0.5 * sayim["kismi"],
        },
        "baskin_hata_turu": (
            "yanlis neden atfetmek ve kacirmak; bozulma uydurmak degil"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yazma", action="store_true")
    args = parser.parse_args()

    s = siniflandir()
    basliklar = {
        "saf_kontrol": "A) SAF KONTROL - hicbir bozulma yok",
        "imzasiz": "B) BOZULMA VAR ama karakteristik imzasi yok",
        "tespit_edilebilir": "C) BOZULMA VAR ve tespit edilebilir",
    }
    for anahtar, baslik in basliklar.items():
        print(baslik)
        for x in s["gruplar"][anahtar]:
            print(f"   {x['kosu']:<9} {x['rol']:<16} -> {x['ajan_teshisi'][:32]:<34} {x['sonuc']}")
        print()

    saf = s["saf_kontrol_ozeti"]
    print(f"Saf kontrolde uydurma: {saf['uydurma']}/{saf['n']} = {saf['uydurma_orani']}  "
          f"Wilson %95 GA {saf['wilson_95']}")
    t = s["tespit_edilebilir_ozeti"]
    print(f"Tespit edilebilir bozulmada dogru neden: {t['dogru_neden_puani']}/{t['n']} "
          f"(tam {t['dogru']} | kismi {t['kismi']} | yanlis neden {t['yanlis_neden']} | "
          f"kacirdi {t['kacirdi']})")
    print(f"\nBaskin hata turu: {s['baskin_hata_turu']}")

    if not args.yazma:
        hedef = DENEME / "hata_turleri.json"
        hedef.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nRapor: {hedef.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

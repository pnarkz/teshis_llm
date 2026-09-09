"""Kontrol tekrarlarini puanlar - ana denemeden AYRI tutarak.

Neden ayri
----------
`kosu_12` (C2 seed13) ve `kosu_13` (C2 seed21) ana denemeden SONRA kontrol
kosusu olarak eklendi ve ayri calistirildi. Cevaplari uretildi ama hic
puanlanmadi: `llm_score.json` yalnizca ana denemenin 11 kosusunu tasiyor.
Demo bu yuzden o iki kosuda "teshis puani" gosteremiyordu.

Puanlar ana denemeye KARISTIRILMAZ. Iki ayri deneydir ve birlestirilirse
"ajan 13 kosuda %X" gibi, iki farkli arac surumuyle uretilmis cevaplari tek
orana katan yaniltici bir sayi cikardi. Cikti ayri bir dosyaya yazilir.

Kullanim:
    python scripts/ajan_tekrarlari_puanla.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teshis.ajan.puanlama import paketi_puanla  # noqa: E402

TEKRAR_DIZINI = ROOT / "reports/ajan_denemesi/kontrol_tekrarlari"
ANAHTAR = ROOT / "reports/ajan_denemesi/answer_key.json"
CIKTI = TEKRAR_DIZINI / "llm_score_tekrarlar.json"


def _cevaplari_topla() -> list[dict]:
    """Her tekrar dosyasindan cevabi okur; dosya adini kayit olarak tasir."""
    cevaplar = []
    for yol in sorted(TEKRAR_DIZINI.glob("*_tekrar*.json")):
        if yol.name.endswith("_arac.json") or yol.name.startswith("llm_score"):
            continue
        ham = json.loads(yol.read_text(encoding="utf-8"))
        cevap = ham[0] if isinstance(ham, list) and ham else ham
        if isinstance(cevap, dict) and cevap.get("run_id"):
            cevap = dict(cevap, _dosya=yol.name)
            cevaplar.append(cevap)
    return cevaplar


def main() -> None:
    if not TEKRAR_DIZINI.is_dir():
        print("kontrol tekrari dizini yok")
        return
    cevaplar = _cevaplari_topla()
    if not cevaplar:
        print("puanlanacak tekrar cevabi yok")
        return

    tam_anahtar = json.loads(ANAHTAR.read_text(encoding="utf-8"))
    # Cevap anahtari YALNIZCA tekrari olan kosularla sinirlanir. Tam anahtarla
    # puanlandiginda tekrari olmayan 9 kosu "missing" sayilip 0 aliyor ve
    # ortalama 1.00 yerine 0.31 cikiyordu - olculmemis bir seyi basarisizlik
    # gibi gosteren bir sayi.
    kosular = {c["run_id"] for c in cevaplar}
    anahtar = {k: v for k, v in tam_anahtar.items() if k in kosular}
    sonuc = paketi_puanla(cevaplar, anahtar)

    # Hangi cevabin hangi dosyadan geldigi kaybolmasin: ayni kosu birden
    # fazla kez tekrarlanabilir ve satirlar yalnizca run_id tasir.
    dosyalar = {}
    for c in cevaplar:
        dosyalar.setdefault(c["run_id"], []).append(c["_dosya"])
    for satir in sonuc.get("runs", []):
        satir["dosya"] = (dosyalar.get(satir["run_id"]) or ["?"])[0]

    sonuc["_not"] = (
        "Kontrol tekrarlari. Ana denemeden AYRI tutulur: farkli arac "
        "surumuyle ve farkli tarihte uretildiler; ortalamalar birlestirilirse "
        "iki ayri deneyi tek orana katan yaniltici bir sayi cikar."
    )
    CIKTI.write_text(json.dumps(sonuc, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(json.dumps(
        {"puanlanan": len(sonuc.get("runs", [])),
         "ortalama": sonuc.get("mean_score"),
         "cikti": str(CIKTI.relative_to(ROOT))},
        ensure_ascii=False, indent=2))
    for s in sonuc.get("runs", []):
        print(f"  {s['run_id']}: beklenen={s['expected']} "
              f"teshis={s.get('model_diagnosis')} puan={s['diagnosis_score']}")


if __name__ == "__main__":
    main()

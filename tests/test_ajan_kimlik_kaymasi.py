"""Kor kimlikler (kosu_NN) kaymadan duruyor mu?

Ajana koşular gercek adlariyla degil `kosu_NN` takma adlariyla verilir ve bu
adlar `results.csv` satir sirasindan TURETILIR. Defter append-only oldugu
surece adlar sabit kalir.

Ama defter bir kez yeniden siralanirsa - ya da bir satir aradan silinirse -
butun takma adlar kayar. O zaman demo, tamamlanmis bir denemenin cevabini
YANLIS senaryoyla eslestirip "gercegi goster" ekraninda baska bir senaryo
adi acar. Hicbir hata mesaji cikmaz; sunumda yanlis bir cumle kurulur.

Bu test, kaydedilmis cevap anahtarinin bugunku haritayla hala ayni sey
soyledigini dogrular.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANAHTAR = ROOT / "reports/ajan_denemesi/answer_key.json"

pytestmark = pytest.mark.skipif(
    not ANAHTAR.is_file(), reason="tamamlanmis ajan denemesi yok"
)


@pytest.fixture(scope="module")
def harita() -> dict[str, str]:
    import sys

    sys.path.insert(0, str(ROOT / "demo"))
    from data_loader import ajan_kosu_haritasi

    return ajan_kosu_haritasi()


def test_kaydedilmis_kimlikler_hala_ayni_senaryoyu_gosteriyor(harita):
    from teshis.ajan import puanlama

    anahtar = json.loads(ANAHTAR.read_text(encoding="utf-8"))
    kaymis = []
    for kosu_id, kayit in anahtar.items():
        beklenen = kayit["expected"] if isinstance(kayit, dict) else kayit
        senaryo = harita.get(kosu_id)
        if senaryo is None:
            kaymis.append(f"{kosu_id}: guncel haritada yok")
            continue
        # Cevap anahtari senaryo ADINI degil bozulma TURUNU tasir; ikisi
        # arasindaki eslesme puanlama modulunde tanimlidir.
        bugun = puanlama.SENARYO_BEKLENEN.get(senaryo)
        if bugun is not None and bugun != beklenen:
            kaymis.append(
                f"{kosu_id}: kayitta '{beklenen}', bugun '{senaryo}' -> '{bugun}'"
            )
    assert not kaymis, (
        "Kor kimlikler kaymis - demo yanlis senaryoyu acar:\n" + "\n".join(kaymis)
    )


def test_her_kimlik_tek_bir_kosuya_isaret_ediyor(harita):
    assert len(set(harita)) == len(harita)
    # Ayni gercek senaryo iki takma ad altinda gorunmemeli.
    tersi: dict[str, list[str]] = {}
    for kosu_id, senaryo in harita.items():
        tersi.setdefault(senaryo, []).append(kosu_id)
    cift = {s: k for s, k in tersi.items() if len(k) > 1}
    assert not cift, f"Ayni senaryo birden fazla kimlikte: {cift}"

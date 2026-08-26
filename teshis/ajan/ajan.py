"""Function-calling tabanli teshis ajani.

``scripts/run_gemini_trial.py`` tek atislik bir denemedir: tum kanit
onceden verilir ve yalnizca **yorumlama** olculur. Bu modul daha zor
soruyu sorar: ajan, hangi kanita bakmasi gerektigini **kendisi secebiliyor
mu?** Model veriyi dogrudan gormez; yalnizca ``araclar.py`` fonksiyonlarini
cagirarak okuyabilir.

Bu ayrim projenin merkezi sorusu icin onemlidir, cunku senaryolarin bir
kismi (D3b, D4, D5) toplam metriklerde GORUNMEZ; yalnizca dogru kirilim
araci cagrildiginda ortaya cikar. Bu yuzden her kosu icin **hangi araclarin
cagrildigi da kaydedilir** ve rapora girer.

Cikti formati ``scripts/score_llm_trial.py`` ile ayni oldugu icin iki deneme
ayni rubrikle puanlanabilir; karsilastirilabilmeleri icin ayri dosyalara
yazilirlar (tek atislik: gemini_response.json, ajan: ajan_response.json).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from . import araclar, semalar

ROOT = Path(__file__).resolve().parents[2]
# Tek atislik denemenin ciktisini EZMEMEK icin ayri dosya. Onceki surum
# gemini_response.json'a yaziyordu ve iki deneme birbirini siliyordu.
VARSAYILAN_CIKTI = ROOT / "reports/llm_trial/ajan_response.json"
VARSAYILAN_LOG = ROOT / "reports/llm_trial/ajan_arac_kaydi.json"

# Arac adi -> gercek Python fonksiyonu. semalar.ARAC_BILDIRIMLERI ile birebir eslesmelidir.
ARAC_UYGULAMALARI: dict[str, Callable[..., Any]] = {
    "kosu_listesini_getir": araclar.kosu_listesini_getir,
    "baseline_metriklerini_getir": araclar.baseline_metriklerini_getir,
    "kosu_metriklerini_getir": araclar.kosu_metriklerini_getir,
    "baseline_farkini_getir": araclar.baseline_farkini_getir,
    "bbox_sayilarini_getir": araclar.bbox_sayilarini_getir,
    "boyut_bazli_recall_getir": araclar.boyut_bazli_recall_getir,
    "kaynak_bazli_recall_getir": araclar.kaynak_bazli_recall_getir,
    "sinif_karisikligini_getir": araclar.sinif_karisikligini_getir,
}

SISTEM_TALIMATI = """
Sen termal drone YOLO nesne tespit modelinin teshis ajanisin. Sana veri
dogrudan verilmez; yalnizca sana taninan araclari cagirarak okuyabilirsin.
kosu_01 saglikli referanstir (veri hic bozulmadan, digerleriyle ayni
protokolde egitilmistir).

Yontem:
- Once genel metrikleri ve baseline farkini al.
- ONEMLI: Toplam mAP/precision/recall bazi bozulmalari TAMAMEN GIZLEYEBILIR.
  Genel metrikler az degismis olsa bile, teshise varmadan once su uc kirilimi
  de kontrol et:
    * boyut_bazli_recall_getir  -> nesne boyutuna gore recall
    * kaynak_bazli_recall_getir -> veri kaynagi grubuna gore recall
    * sinif_karisikligini_getir -> gercek sinif hangi sinif olarak tahmin edilmis
  Bunlardan birinde belirgin fark varsa teshisi ona dayandir.
- sinif_karisikligini_getir ciktisinda "bulunamadi", o gercek kutunun hicbir
  tahminle eslesmedigi anlamina gelir; baska bir sinif adi ise yanlis
  siniflandirmadir.

Kurallar:
- kosu_01, kosu_02, ... kimliklerinden hangi senaryoya ait olduklarini TAHMIN
  ETMEYE CALISMA; yalnizca sayilara dayan.
- Her iddiayi en az iki sayisal kanitla destekle.
- Bir grubun ornek sayisi dusukse (orn. 20 bbox altinda) kesin genelleme
  yapma; bunu limitations alaninda GRUP ADI VE SAYISIYLA birlikte belirt.
- Kanit gercekten yetersizse diagnosis alaninda "yetersiz_kanit" yaz. Ancak
  metrikler saglikli gorunuyorsa bunu da soyleyebilirsin; olmayan bir
  bozulmayi uydurma.
- Analizini bitirdiginde SADECE asagidaki alanlara sahip tek bir JSON nesnesi
  dondur; baska aciklama metni veya markdown ekleme:

{sema}
""".strip()


def _arac_tanimlari():
    """google-genai Tool nesnesini olusturur (yalnizca cagrildiginda import edilir)."""
    from google.genai import types

    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(**bildirim) for bildirim in semalar.ARAC_BILDIRIMLERI
        ]
    )


def json_guvenli(deger: Any) -> Any:
    """Infinity/NaN gibi JSON'da gecersiz degerleri metne cevirir.

    RFC 8259 Infinity ve NaN'a izin vermez; Python'un json modulu bunlari
    varsayilan olarak yazar ama Gemini API'si boyle bir govdeyi
    400 INVALID_ARGUMENT ile reddeder. Gercek bir kosuda tum kosular bu yuzden
    basarisiz oldu: boyut bandi tanimindaki ust sinir float("inf") idi.

    Kaynak (metrikler.bant_araliklari) duzeltildi; bu fonksiyon ikinci savunma
    hattidir ve eski ciktilarla da calisilabilmesini saglar.
    """
    if isinstance(deger, float):
        if deger != deger:  # NaN
            return None
        if deger in (float("inf"), float("-inf")):
            return "sonsuz" if deger > 0 else "-sonsuz"
        return deger
    if isinstance(deger, dict):
        return {anahtar: json_guvenli(alt) for anahtar, alt in deger.items()}
    if isinstance(deger, (list, tuple)):
        return [json_guvenli(alt) for alt in deger]
    return deger


def _arac_cagrisini_calistir(ad: str, argumanlar: dict[str, Any]) -> dict[str, Any]:
    """Araci calistirir; hata olursa ajani dusurmek yerine modele hata dondurur."""
    fonksiyon = ARAC_UYGULAMALARI.get(ad)
    if fonksiyon is None:
        return {"hata": f"bilinmeyen_arac:{ad}"}
    try:
        sonuc = fonksiyon(**argumanlar)
    except Exception as hata:  # noqa: BLE001 - arac hatasi modele geri bildirilir
        return {"hata": str(hata)}
    # FunctionResponse.response bir sozluk olmalidir; liste donen araclari sar.
    if not isinstance(sonuc, dict):
        sonuc = {"sonuc": sonuc}
    return json_guvenli(sonuc)


def json_ayikla(metin: str) -> dict[str, Any]:
    """Model ciktisindan JSON nesnesini cikarir.

    Model bazen JSON'u ```json ... ``` blogu icinde veya kisa bir aciklama
    metniyle birlikte dondurur. Onceki surum dogrudan json.loads cagirdigi
    icin bu durumda cokerdi.
    """
    ham = (metin or "").strip()
    if not ham:
        raise ValueError("model bos cevap dondu")
    blok = re.search(r"```(?:json)?\s*(.+?)\s*```", ham, re.DOTALL)
    if blok:
        ham = blok.group(1).strip()
    try:
        return json.loads(ham)
    except json.JSONDecodeError:
        pass
    # Son care: ilk '{' ile son '}' arasini dene.
    bas, son = ham.find("{"), ham.rfind("}")
    if bas >= 0 and son > bas:
        return json.loads(ham[bas : son + 1])
    raise ValueError(f"cevapta gecerli JSON bulunamadi: {ham[:200]}")


def _bekleme_suresi(hata: Exception, varsayilan: float = 20.0) -> float:
    """429 hatasindan sunucunun onerdigi bekleme suresini cikarir."""
    metin = str(hata)
    eslesme = re.search(r"retry in ([\d.]+)s", metin, re.IGNORECASE)
    if eslesme:
        return float(eslesme.group(1)) + 1.0
    eslesme = re.search(r"'retryDelay':\s*'(\d+)s'", metin)
    if eslesme:
        return float(eslesme.group(1)) + 1.0
    return varsayilan


def _kota_hatasi_mi(hata: Exception) -> bool:
    metin = str(hata)
    return "429" in metin or "RESOURCE_EXHAUSTED" in metin


def _istek_gonder(client, model: str, contents, config, deneme: int = 4):
    """generate_content cagrisini yapar; kota hatasinda bekleyip yeniden dener.

    Ucretsiz katmanda dakikada 5 istek siniri var; ajan her kosuda birden fazla
    tur kullandigi icin bu sinire kolayca takiliyor. Sunucunun onerdigi
    retryDelay dikkate alinarak beklenir.
    """
    son_hata: Exception | None = None
    for sira in range(1, deneme + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as hata:  # noqa: BLE001
            if not _kota_hatasi_mi(hata) or sira == deneme:
                raise
            son_hata = hata
            sure = _bekleme_suresi(hata)
            print(f"      kota siniri, {sure:.0f} sn bekleniyor ({sira}/{deneme - 1})...")
            time.sleep(sure)
    raise son_hata  # pragma: no cover - dongu her zaman doner veya firlatir


def teshis_uret(
    kosu_id: str,
    model: str = "gemini-3.6-flash",
    max_tur: int = 8,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tek kosu icin arac erisimli teshis uretir.

    Dondurur: (teshis, arac_kaydi). arac_kaydi, modelin hangi araci hangi
    argumanla ve kacinci turda cagirdigini icerir; "ajan dogru kanita
    yonelebiliyor mu" sorusu bu kayitla olculur.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY tanimli degil. API anahtarini sohbete yazmadan "
            "PowerShell'de ortam degiskeni olarak ayarlayin."
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    sema_metni = json.dumps(semalar.TESHIS_SEMASI["properties"], ensure_ascii=False, indent=2)
    config = types.GenerateContentConfig(
        tools=[_arac_tanimlari()],
        system_instruction=SISTEM_TALIMATI.format(sema=sema_metni),
        # Modelin araclari kendisi secmesini istiyoruz; otomatik cagri kapali
        # olmalidir ki her cagriyi biz calistirip kaydedebilelim.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents: list[Any] = [
        types.Content(
            role="user",
            parts=[types.Part(text=f"Incelenecek kosu: {kosu_id}. Teshisini uret.")],
        )
    ]
    arac_kaydi: list[dict[str, Any]] = []

    for tur in range(1, max_tur + 1):
        response = _istek_gonder(client, model, contents, config)
        if not response.candidates:
            raise ValueError(f"{kosu_id}: model aday cevap dondurmedi")
        aday = response.candidates[0]
        contents.append(aday.content)

        parcalar = list(aday.content.parts or [])
        cagrilar = [p.function_call for p in parcalar if getattr(p, "function_call", None)]

        if not cagrilar:
            cevap = json_ayikla(response.text or "")
            cevap["run_id"] = kosu_id
            return cevap, arac_kaydi

        yanit_parcalari = []
        for cagri in cagrilar:
            argumanlar = dict(cagri.args or {})
            sonuc = _arac_cagrisini_calistir(cagri.name, argumanlar)
            arac_kaydi.append(
                {
                    "tur": tur,
                    "arac": cagri.name,
                    "argumanlar": argumanlar,
                    "hata": sonuc.get("hata") if isinstance(sonuc, dict) else None,
                }
            )
            yanit_parcalari.append(
                types.Part.from_function_response(name=cagri.name, response=sonuc)
            )
        contents.append(types.Content(role="user", parts=yanit_parcalari))

    raise RuntimeError(
        f"{kosu_id}: {max_tur} turda teshis alinamadi ({len(arac_kaydi)} arac cagrisi yapildi)"
    )


def kor_deneme_calistir(
    model: str = "gemini-3.6-flash",
    cikti: Path = VARSAYILAN_CIKTI,
    log_yolu: Path = VARSAYILAN_LOG,
    bekleme_sn: float = 8.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Tum anonim kosular icin teshis uretir; tek kosunun hatasi digerlerini dusurmez."""
    sonuclar: list[dict[str, Any]] = []
    kayitlar: dict[str, Any] = {}

    for kosu_id in araclar.kosu_listesini_getir():
        try:
            teshis, arac_kaydi = teshis_uret(kosu_id, model=model)
            hatalar = semalar.teshis_dogrula(teshis)
            if hatalar:
                teshis["_sema_hatalari"] = hatalar
                print(f"  {kosu_id}: UYARI sema hatalari {hatalar}")
            sonuclar.append(teshis)
            kayitlar[kosu_id] = {"arac_cagrilari": arac_kaydi, "hata": None}
            araclar_ozeti = ", ".join(dict.fromkeys(k["arac"] for k in arac_kaydi)) or "(yok)"
            print(f"  {kosu_id}: {len(arac_kaydi)} arac cagrisi -> {araclar_ozeti}")
        except Exception as hata:  # noqa: BLE001 - bir kosu duserse digerleri devam etsin
            print(f"  {kosu_id}: BASARISIZ ({type(hata).__name__}: {hata})")
            sonuclar.append({"run_id": kosu_id, "diagnosis": "hata", "_hata": str(hata)})
            kayitlar[kosu_id] = {"arac_cagrilari": [], "hata": str(hata)}
        if bekleme_sn:
            time.sleep(bekleme_sn)

    cikti.parent.mkdir(parents=True, exist_ok=True)
    cikti.write_text(json.dumps(sonuclar, indent=2, ensure_ascii=False), encoding="utf-8")
    log_yolu.write_text(json.dumps(kayitlar, indent=2, ensure_ascii=False), encoding="utf-8")
    return sonuclar, kayitlar


def arac_kullanim_ozeti(kayitlar: dict[str, Any]) -> dict[str, Any]:
    """Hangi araclarin ne siklikta cagrildigini ozetler."""
    sayim: dict[str, int] = {}
    kosu_basina: dict[str, int] = {}
    for kosu_id, kayit in kayitlar.items():
        adlar = [c["arac"] for c in kayit.get("arac_cagrilari", [])]
        kosu_basina[kosu_id] = len(adlar)
        for ad in adlar:
            sayim[ad] = sayim.get(ad, 0) + 1
    kirilim_araclari = {
        "boyut_bazli_recall_getir",
        "kaynak_bazli_recall_getir",
        "sinif_karisikligini_getir",
    }
    kirilim_kullanan = [
        kosu_id
        for kosu_id, kayit in kayitlar.items()
        if kirilim_araclari & {c["arac"] for c in kayit.get("arac_cagrilari", [])}
    ]
    return {
        "arac_cagri_sayisi": dict(sorted(sayim.items(), key=lambda x: -x[1])),
        "kosu_basina_cagri": kosu_basina,
        "kirilim_araci_kullanan_kosular": sorted(kirilim_kullanan),
        "kirilim_araci_kullanim_orani": (
            round(len(kirilim_kullanan) / len(kayitlar), 3) if kayitlar else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--output", type=Path, default=VARSAYILAN_CIKTI)
    parser.add_argument("--log", type=Path, default=VARSAYILAN_LOG)
    parser.add_argument("--kosu", default=None, help="Yalnizca tek bir kosu calistir (orn. kosu_02)")
    parser.add_argument("--bekleme", type=float, default=8.0,
                        help="Kosular arasi bekleme (sn). Ucretsiz katman 5 istek/dk sinirlidir.")
    args = parser.parse_args()

    if args.kosu:
        teshis, kayit = teshis_uret(args.kosu, model=args.model)
        print(json.dumps({"teshis": teshis, "arac_cagrilari": kayit}, indent=2, ensure_ascii=False))
        return

    print(f"ajan denemesi basliyor: model={args.model}")
    sonuclar, kayitlar = kor_deneme_calistir(
        model=args.model, cikti=args.output, log_yolu=args.log, bekleme_sn=args.bekleme
    )
    ozet = arac_kullanim_ozeti(kayitlar)
    print(f"\nsaved={args.output.resolve()}")
    print(f"log  ={args.log.resolve()}")
    print("\narac kullanimi:")
    print(json.dumps(ozet, indent=2, ensure_ascii=False))
    basarisiz = [s["run_id"] for s in sonuclar if s.get("_hata")]
    if basarisiz:
        print(f"\nUYARI basarisiz kosular: {basarisiz}")
    print(f"\nsonraki adim: python scripts/score_llm_trial.py --response {args.output}")


if __name__ == "__main__":
    main()

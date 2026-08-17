# Proje Dosya Sozlesmesi

Bu belge, proje icindeki dosya ve klasor adlarinin tek kaynagidir.

## Klasorler

```text
termal_teshis/
  config.yaml                         Ortam ve yol ayarlari
  main_model.pt                       Secilen ana model, buyuk dosya
  proje-brifingi-v2.2.md              Proje kapsam belgesi
  results.csv                         Tum deneylerin tek satirlik kaydi
  teshis/                             Ana Python paketi
    veri/                             Veri raporu ve veri surumleme
    egitim/                           Egitim komutlari ve kayit
    degerlendirme/                    Metrik, hata ve kanit analizi
    ajan/                             Araclar, JSON cikti ve puanlama
    servis/                           Aşama 2 servis ve log analizi
  senaryolar/                         Degistirilemez senaryo tanimlari
    katalog.yaml                      Senaryo kodlarinin tek listesi
    veri/                             D1-D6 veri arizalari
    egitim/                           E1-E4 egitim arizalari
  veri_surumleri/                     Uretilen dataset kopyalari
  experiments/                        Her kosunun kanit ve metrik ciktilari
  reports/                            Insan okunabilir raporlar
  scripts/                            Tekrarlanabilir yardimci komutlar
  tests/                              Birim ve sozlesme testleri
```

## Isim kurallari

- Senaryo kodlari: `D1`, `D2a`, `D2b`, `D3`, `D4`, `D5`, `D6a`, `D6b`, `E1`, `E2`, `E3`, `E4`.
- Veri surumu: `v00_saglikli`, `v01_d1_sinif_yetersizligi` gibi.
- Kosu: `run_YYYYMMDD_HHMMSS_<senaryo>_<seed>`.
- Ajanin gordugu kosu adlari puanlama oncesi anonimlestirilir.
- Test seti gelistirme sirasinda kullanilmaz.

## Veri akisi

```text
dataset/
  -> teshis/veri/surum_uret.py
  -> veri_surumleri/vXX_*/manifest.json
  -> teshis/egitim/kos.py
  -> experiments/run_*/results.csv + kanit.json
  -> teshis/degerlendirme/kanit.py
  -> teshis/ajan/ajan.py
```

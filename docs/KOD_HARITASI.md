# Sunum icin kod haritasi

Bir senaryo soruldugunda once asagidaki **uygulama** dosyasini acin.
YAML deneyin tanimini, Python uygulamayi, `reports/` olcumleri tutar.

## Projenin iskeleti

```text
teshis/
  veri/              Veri tarama, saglikli referans ve senaryo uygulamalari
  egitim/            Ortak egitim kosucusu, protokol ve kosu kaydi
  degerlendirme/     Metrikler, karsilastirma, guven araligi ve kanit
  ajan/              LLM araclari, teshis dongusu, cikti semasi ve puanlama
senaryolar/          Deney tanimlari ve parametreler (YAML)
scripts/            Komut satiri ve Kaggle girisleri
demo/               Streamlit konsolu; bolumler/ icinde ekranlar
tests/              Davranis ve sozlesme testleri
docs/               Yontem, kod haritasi, bulgular ve calistirma
veri_surumleri/      Uretilmis veri varyantlari ve manifestler
experiments/        Egitim kosulari
reports/            Olcumler ve ajan deneyleri
```

## Senaryo -> uygulama -> calistirma -> sonuc

| Senaryo | Uygulama | Calistirma girisi | Sonuc |
|---|---|---|---|
| v00 saglikli referans | [referans_v00_saglikli.py](../teshis/veri/referans_v00_saglikli.py) | `python -m teshis.veri.surum_uret --surum v00 ...` | `reports/referans_v00/` |
| D1 sinif yetersizligi | [senaryo_d1_sinif_yetersizligi.py](../teshis/veri/senaryo_d1_sinif_yetersizligi.py) | `python -m teshis.veri.surum_uret --surum d1 ...` | `reports/senaryo_D1/` |
| D2a lokalizasyon gurultusu | [senaryo_d2a_lokalizasyon_gurultusu.py](../teshis/veri/senaryo_d2a_lokalizasyon_gurultusu.py) | `scripts/kaggle_D2a_lokalizasyon_gurultusu.py` | `reports/senaryo_D2a/` |
| D2b eksik etiket | [senaryo_d2b_eksik_etiket.py](../teshis/veri/senaryo_d2b_eksik_etiket.py) | `scripts/senaryo_D2b_eksik_etiket.py` / `scripts/kaggle_D2b_eksik_etiket.py` | `reports/senaryo_D2b/`, `reports/senaryo_D2b_final_best/` |
| D3 UAP/UAI, D3b tasit/insan karisikligi | [senaryo_d3_d3b_sinif_karisikligi.py](../teshis/veri/senaryo_d3_d3b_sinif_karisikligi.py) | `scripts/senaryo_D3_D3b_sinif_karisikligi.py` | `reports/senaryo_D3/`, `reports/senaryo_D3b/` |
| D4 kucuk nesne sinyal kaybi | [senaryo_d4_kucuk_nesne_sinyal_kaybi.py](../teshis/veri/senaryo_d4_kucuk_nesne_sinyal_kaybi.py) | `scripts/senaryo_D4_kucuk_nesne.py` | `reports/senaryo_D4/` |
| D5 kaynak alani kaymasi | [senaryo_d5_kaynak_alani_kaymasi.py](../teshis/veri/senaryo_d5_kaynak_alani_kaymasi.py) | `scripts/senaryo_D5_kaynak_kaymasi.py` | `reports/senaryo_D5/`, `reports/senaryo_D5_last_pt/` |
| D6a split sizintisi | [senaryo_d6a_split_sizintisi.py](../teshis/veri/senaryo_d6a_split_sizintisi.py) | `scripts/senaryo_D6a_split_sizintisi.py` | `reports/senaryo_D6a/` |
| D6b tekrar agirligi | [senaryo_d6b_tekrar_agirligi.py](../teshis/veri/senaryo_d6b_tekrar_agirligi.py) | `scripts/senaryo_D6b_tekrar_agirligi.py` | `reports/senaryo_D6b/` |
| E1 overfitting | [senaryo_e1_overfitting.py](../teshis/veri/senaryo_e1_overfitting.py) (alt kume), [kos.py](../teshis/egitim/kos.py) (egitim) | `scripts/senaryo_E1_overfitting.py`, sonra `python -m teshis.egitim.kos --e-senaryo E1 ...` | `reports/senaryo_E1/`, `reports/senaryo_E1_last_pt/` |
| E2 underfitting, E3/E3b kararsizlik | [egitim_protokolu.yaml](../senaryolar/egitim_protokolu.yaml) (`e_serisi`), [protokol.py](../teshis/egitim/protokol.py), [kos.py](../teshis/egitim/kos.py) | `python -m teshis.egitim.kos --e-senaryo E2 ...` (E3/E3b icin kodu degistir) | Kosu ve sonuc yollarini `results.csv` ve [BULGULAR.md](BULGULAR.md) uzerinden bulun |
| E4 cozunurluk uyumsuzlugu | [senaryo_e4_cozunurluk_uyumsuzlugu.py](../teshis/degerlendirme/senaryo_e4_cozunurluk_uyumsuzlugu.py) | `scripts/senaryo_E4_cozunurluk_uyumsuzlugu.py` | `reports/senaryo_E4/` |

D serisinin tanimlari [senaryolar/veri/](../senaryolar/veri/), E serisininkiler
[senaryolar/egitim/](../senaryolar/egitim/) altindadir. Tam komutlar icin
[CALISTIRMA.md](CALISTIRMA.md). Yukaridaki `...` ifadeleri komut sablonudur.

## Sunumda sik sorulan sorular

| Soru | Acilacak dosya |
|---|---|
| Veri setini nasil taradik? | [istatistik.py](../teshis/veri/istatistik.py) |
| Tani seti nasil ayrildi? | [val_olustur.py](../teshis/veri/val_olustur.py) |
| Veri degisikligi nasil kaydediliyor? | Ilgili `senaryo_*.py` icindeki manifest olusturma bolumu; [manifest.py](../teshis/veri/manifest.py) yalnizca format sabitini tanimlar |
| Model nasil egitiliyor? | [kos.py](../teshis/egitim/kos.py), [protokol.py](../teshis/egitim/protokol.py) |
| Precision/recall ve kirilimlar nerede? | [metrikler.py](../teshis/degerlendirme/metrikler.py) |
| Karsilastirma adil mi? | [karsilastirilabilirlik.py](../teshis/degerlendirme/karsilastirilabilirlik.py) |
| Rastgelelikten nasil ayiriyoruz? | [gurultu.py](../teshis/degerlendirme/gurultu.py), [bootstrap.py](../teshis/degerlendirme/bootstrap.py) |
| LLM hangi verileri goruyor? | [araclar.py](../teshis/ajan/araclar.py) |
| LLM'e verilen talimat ve dongu nerede? | [ajan.py](../teshis/ajan/ajan.py) |
| Teshis nasil puanlaniyor? | [puanlama.py](../teshis/ajan/puanlama.py) |
| Tekrarli deney nasil kaydediliyor? | [ajan_deney.py](../scripts/ajan_deney.py), `reports/ajan_deneyleri/` |
| Konsoldaki ekranlar nerede? | [demo/bolumler/](../demo/bolumler/); giris [app.py](../demo/app.py) |
| Ajanin konsola baglantisi nerede? | [ajan_katmani.py](../demo/ajan_katmani.py) |

## Isimlendirme ve uyumluluk

- Yeni senaryo modulleri: `senaryo_<kod>_<aciklama>.py` (kucuk harf).
- Saf yardimcilar senaryo adini tasimaz: `dosyalar.py`, `istatistik.py`.
- `surum_uret.py` v00/D1 komut girisidir; uygulamayi ilgili modulden alir.
- `bozulmalar.py` eski D1 importlarini korur; algoritma burada yazilmaz.
- Eski script adlari belgelerdeki ve Kaggle'daki komutlari korumak icin kalir.
  Her birinin basinda uygulamanin yeni yolu yazilidir.
- Tarihsel sonuc/deney klasorleri yeniden adlandirilmaz: raporlarin ve kosu
  kimliklerinin baglantilari bu yollara dayanir.

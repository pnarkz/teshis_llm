# Mimari ve Dosya Sozlesmesi

Bu belge, proje icindeki dosya ve klasor adlarinin **tek kaynagidir**.
Yeni bir dosya eklenirken buradaki adlandirma kurallarina uyulur.

## Adlandirma kurallari

Klasor ve dosya adlari **onek = ne tur, sonek = hangisi** kuralini izler.
Boylece alfabetik siralama isleri kendiliginden gruplar.

| Onek | Anlami | Ornek |
|---|---|---|
| `senaryo_` | Bir bozulma senaryosunun sonucu | `reports/senaryo_D4/` |
| `referans_` | Saglikli karsilastirma tabani | `reports/referans_v00/` |
| `yolo26n_` | Farkli taban modelden gelen kontrol cifti | `reports/yolo26n_senaryo_D1n/` |
| `eski_` | Superseded kosu; tarihsel kayit, kullanilmaz | `reports/eski_D3_val_sizintili/` |
| `hata_galerisi_` | Siralanmis hata ornekleri | `reports/hata_galerisi_D2a/` |
| `kontrol_` | Sartname bolum 8 kontrol kosullari | `reports/kontrol_C2_seed7/` |
| `_last_pt` | Ayni kosunun `last.pt` checkpoint'i | `reports/senaryo_D5_last_pt/` |

Senaryo kodlari degismez: `D1`, `D2a`, `D2b`, `D3`, `D3b`, `D4`, `D5`,
`D6a`, `D6b`, `E1`-`E4`. Veri surumleri `vNN_<senaryo>_<aciklama>` bicimindedir.
Kosu klasorleri `run_YYYYMMDD_HHMMSS_<senaryo>_<seed>` bicimindedir.

## Klasorler

```text
termal_teshis/
  README.md                     Giris kapisi ve belge haritasi
  config.yaml                   Ortam ve yol ayarlari
  results.csv                   Tum kosularin tek satirlik kaydi
  main_model.pt                 Ana model (D serisi bu modelden fine-tune edilir)
  yolo26n.pt                    Genel amacli model (yolo26n kontrol cifti icin)

  docs/                         Belgeler
    MIMARI.md                   Bu dosya: dosya sozlesmesi
    BULGULAR.md                 Tum senaryo sonuclari
    KURALLAR.md                 Degismez kurallar ve sabit yollar
    CALISTIRMA.md               Kurulum ve komutlar
    BAKIM_GUNLUGU.md            Kronolojik degisiklik kaydi
    SUNUM.md                    Teknik olmayan anlatim
    proje-brifingi-v2.1.md      Sartname (disaridan gelen referans belge)

  teshis/                       Ana Python paketi
    veri/
      istatistik.py             Dataset saglik taramasi ve kaynak gruplama
      surum_uret.py             v00 saglikli surum + D1 (manifest-only)
      val_olustur.py            Kilitli tanı seti uretimi
      bozulmalar.py             Ortak bozulma yardimcilari
    egitim/
      kos.py                    Kontrollu egitim kosusu (+ --devam)
      protokol.py               Ortak egitim protokolunu YAML'dan okur
      kayit.py                  Kosu manifesti yazar
    degerlendirme/
      d1_sonuc.py               Tanı setinde degerlendirme (sinif AP/P/R)
      metrikler.py              Sinif / boyut / kaynak kirilimi + karisiklik matrisi
      bootstrap.py              Wilson araligi, iki oran testi, goruntu bootstrap
      karsilastir.py            Iki modeli ayni sette karsilastirir
      hata_galerisi.py          Siralanmis hata ornekleri uretir
      kanit.py                  Sartname bolum 9 kanit sozlesmesi (kanit.json)
    ajan/
      araclar.py                Function-calling araclari (anonim metrik okuma)
      semalar.py                Arac bildirimleri + teshis cikti semasi
      ajan.py                   Gemini function-calling orkestrasyonu
      puanlama.py               Pilot puanlama rubrigi (tek kaynak)
    servis/                     Asama 2 (henuz bos iskelet)

  senaryolar/                   Degistirilemez senaryo tanimlari
    katalog.yaml                Senaryo kodlarinin tek listesi
    egitim_protokolu.yaml       Sabit egitim protokolu (tek kaynak) +
                                e_serisi: E kosularinin BEYAN EDILMIS sapmalari
    veri/                       D1-D6b bozulma parametreleri
    egitim/                     E1-E4 egitim arizasi tanimlari

  scripts/                      Adindan ne yaptigi anlasilan calistiricilar
    senaryo_D2b_eksik_etiket.py
    senaryo_D3_D3b_sinif_karisikligi.py    (sinif cifti parametreyle secilir)
    senaryo_D4_kucuk_nesne.py
    senaryo_D5_kaynak_kaymasi.py
    senaryo_D6a_split_sizintisi.py
    senaryo_D6b_tekrar_agirligi.py
    senaryo_E4_cozunurluk_uyumsuzlugu.py   (egitim gerektirmez; imgsz taramasi)
    kaggle_D2a_lokalizasyon_gurultusu.py
    kaggle_D2b_eksik_etiket.py
    ajan_paket_hazirla.py                  LLM deneme paketi + cevap anahtari
    ajan_tek_atislik_calistir.py           Tum kanit onceden verilir
    ajan_puanla.py                         Rubrikle puanlar

  veri_surumleri/               Uretilen dataset surumleri (manifest + data.yaml)
  experiments/                  Her egitim kosusunun ciktisi ve agirliklari
  reports/                      Degerlendirme ciktilari (asagida)
  val_diagnostic/               KILITLI tanı seti; degistirilmez
  demo/                         Streamlit sunum konsolu
  tests/                        Birim ve sozlesme testleri
```

## reports/ duzeni

```text
reports/
  veri_raporu.json              Dataset saglik taramasi

  referans_v00/                 Saglikli referans (otoriter karsilastirma tabani)
  referans_v00_last_pt/

  senaryo_D1/ D2a/ D2b/ D2b_final_best/
  senaryo_D3/ D3b/ D4/ D5/ D6a/ D6b/          Guncel senaryo sonuclari
  senaryo_D4_last_pt/ D5_last_pt/ D6b_last_pt/  last.pt varyantlari

  senaryo_E4/                   E4 tarama ozeti + sinif anlamlilik testi
  senaryo_E4_imgsz512/ ... _imgsz1280/          E4 ham degerlendirmeleri

  yolo26n_referans_v00n/        Farkli taban modelli kontrol cifti
  yolo26n_senaryo_D1n/

  eski_D1_protokol_sapmali/     SUPERSEDED - kullanilmaz, tarihsel kayit
  eski_D3_val_sizintili/
  eski_D1_last_pt/

  kirilim/                      metrikler.py ciktilari (run_id basina bir dosya)
  kanit/                        Egitim YAPMAYAN kosularin kanit sozlesmesi.
                                Egitim kosularininki experiments/<kosu>/kanit.json
                                altindadir; D6a ve E4 gibi yeniden
                                degerlendirmeler baskasinin dizinini ezmesin diye
                                buraya yazilir.
  ajan_denemesi/                LLM paketi, cevap, puan
  ajan_denemesi_arsiv_4kosu/    Kirilim araclari eklenmeden onceki deneme
  model_secimi/                 Adil model karsilastirmasi (kullanilan)
  model_secimi_ilk_adil_degil/  Ilk karsilastirma; imgsz farkli oldugu icin adil degil
  hata_galerisi_D2a/
```

Her senaryo klasoru ayni yapiyi tasir:

```text
reports/senaryo_D4/
  d1_metrics.json     Metrikler (dosya adi tarihsel; degerlendirme/d1_sonuc.py uretir)
  gorseller/          confusion matrix, PR/F1 egrileri, val ornekleri
```

## Veri akisi

```text
dataset/
  -> teshis/veri/istatistik.py          saglik raporu
  -> teshis/veri/surum_uret.py          veri_surumleri/vNN_*/manifest.json
     veya scripts/senaryo_*.py
  -> teshis/egitim/kos.py               experiments/run_*/weights/best.pt
  -> teshis/degerlendirme/d1_sonuc.py   reports/senaryo_*/d1_metrics.json
  -> teshis/degerlendirme/metrikler.py  reports/kirilim/<run_id>.json
  -> results.csv                        tek satirlik kayit
  -> teshis/ajan/araclar.py             ajanin gordugu anonim metrikler
```

## Ajana neyin verilmedigi

`teshis/ajan/araclar.py` iki filtre uygular:

- `AJANA_VERILMEYEN`: saglikli referans (kosu_01 olarak sunulur) ve farkli
  taban modelden gelen kosular (yolo26n cifti). Farkli taban modelde fark,
  bozulmadan degil model kapasitesinden gelir.
- `KILITLI_DEGERLENDIRME_SETI`: yalnizca `val_diagnostic` uzerinde olculmus
  kosular. D6a sizintili bir kumede olculdugu icin disaridadir.

Her iki filtre de testlidir; kayitlar `results.csv`'de durur, yalnizca ajana
sunulmaz.

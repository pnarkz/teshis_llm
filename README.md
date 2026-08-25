# Termal Teshis Ajani

Bu belge projenin ana takip, onboarding ve teslim dokumanidir. Bir mentor veya
baska bir AI ajan bu dosyayi okuyarak projenin amacini, dosya yapisini, mevcut
durumunu, degismez kurallarini ve sonraki adimini anlayabilmelidir.

## 1. Projenin Amaci

Bu proje termal drone goruntulerinde YOLO nesne tespit modelinin kontrollu veri
ve egitim arizalari altinda nasil bozuldugunu olcer. Ajan katmani bu olcumlerden
kanita dayali teshis uretir.

Sinif sozlesmesi degismez:

| ID | Sinif |
|---:|---|
| 0 | tasit |
| 1 | insan |
| 2 | UAP |
| 3 | UAI |

Arastirma sorusu: Termal nesne tespit sistemi hangi veri, etiket, dagilim,
egitim ve alan kaymasi kosullarinda bozulur; bir LLM/ajan bu bozulmayi yeterli
kanitla teshis edebilir mi?

## Bakım Günlüğü

Bu bölüm, projede yapılan düzenleme/temizlik/eksik-giderme işlerinin
kronolojik kaydıdır. Her mühendislik değişikliğinden sonra buraya yeni bir
madde eklenir; böylece hangi sorunun ne zaman ve nasıl giderildiği README
üzerinden takip edilebilir. En yeni kayıt en üstte durur.

### 2026-08-25 (2) — Ajan katmanı gerçek koda kavuştu, ilk birim testleri eklendi

Sorun: `teshis/ajan/*.py` ve `teshis/servis/*.py` tamamen tek satırlık
docstring'den ibaretti. Projenin adı "Termal Teşhis Ajanı" olmasına rağmen
gerçekte çalışan tek şey `scripts/run_gemini_trial.py` içindeki, hiç araç
(tool) kullanmayan tek seferlik bir Gemini metin çağrısıydı. Ayrıca
`tests/` klasöründe hiç test dosyası yoktu.

Yapılanlar:

- [teshis/ajan/araclar.py](teshis/ajan/araclar.py): gerçek function-calling
  araçları (`kosu_listesini_getir`, `kosu_metriklerini_getir`,
  `baseline_farkini_getir`, `bbox_sayilarini_getir`). `katalog.yaml`'daki
  `manifest_ajana_verilmez: true` kuralını kod seviyesinde uyguluyor —
  hiçbir fonksiyon manifest veya senaryo adı okumuyor, sadece
  `results.csv`/`reports/` metriklerini anonim `kosu_NN` kimlikleriyle
  sunuyor.
- [teshis/ajan/semalar.py](teshis/ajan/semalar.py): ajan çıktısı için
  `TESHIS_SEMASI` + `teshis_dogrula` (programatik doğrulama — eskiden
  Gemini'nin JSON'u hiç doğrulanmadan diske yazılıyordu) ve Gemini
  function-calling için `ARAC_BILDIRIMLERI`.
- [teshis/ajan/puanlama.py](teshis/ajan/puanlama.py): pilot puanlama
  mantığı `scripts/score_llm_trial.py`'den buraya taşındı, tek kaynak
  oldu; script artık ince bir CLI sarmalayıcısı. Taşıma sonrası çıktı
  gerçek `reports/llm_trial/gemini_response.json` ile bit-bit karşılaştırıldı,
  fark yok.
- [teshis/ajan/ajan.py](teshis/ajan/ajan.py): gerçek Gemini function-calling
  döngüsü (`teshis_uret`, `kor_deneme_calistir`). **Not:** bu kısım canlı
  bir `GEMINI_API_KEY` ile bu ortamda uçtan uca test edilmedi (internet
  erişimi yok); kullanmadan önce tek bir `kosu_id` ile duman testi
  yapılması önerilir. Araç/şema katmanı tamamen yerel test edildi.
- `tests/`: `conftest.py` + 4 yeni test dosyası, toplam 40 test, hepsi
  gerçek proje verisiyle (results.csv, reports/) çalışıyor ve geçiyor.
  `test_egitim_protokolu.py` özellikle önceki maddedeki lr0/warmup_epochs
  hatasının bir daha sessizce geri gelmemesini garanti eden bir sözleşme
  testi içeriyor.
- `requirements-dev.txt` eklendi (`pytest`); `pyproject.toml`'a
  `[tool.pytest.ini_options]` eklendi.
- **Açık iş:** `teshis/servis/*` (API, anomali, arıza, loglama) hâlâ boş
  iskelet; roadmap'te "Aşama 2" olarak işaretli, bu turda kapsanmadı.

### 2026-08-25 (1) — Eğitim protokolü merkezileştirildi, Git kanıt takibi tutarlı hale getirildi

Sorun: `teshis/egitim/kos.py` (D1'de kullanıldı) `lr0=0.0005`,
`warmup_epochs=2` kullanıyordu; `scripts/kaggle_d2a.py`, `kaggle_d2b.py`,
`local_d2b.py`, `local_d3.py` ise `lr0=0.001`, `warmup_epochs=3` kullanıyordu.
Bu, D1 ile D2a/D2b/D3 arasındaki metrik farkının bir kısmının veri
bozulmasından değil, farklı optimizasyon ayarından kaynaklanabileceği
anlamına geliyordu — projenin "senaryolar arasında tek değişken değişir"
temel ilkesini zayıflatan gerçek bir metodoloji açığıydı.

Yapılanlar:

- [senaryolar/egitim_protokolu.yaml](senaryolar/egitim_protokolu.yaml):
  tüm D serisi koşularda paylaşılan sabit optimizasyon/augmentasyon
  değerlerinin tek kaynağı. `bilinen_sapmalar` alanında D1'in tamamlanmış
  koşusunun (`run_20260817_222323_D1_42`) bu protokolden önce, farklı
  `lr0`/`warmup_epochs` ile üretildiği açıkça kayıt altına alındı.
- [teshis/egitim/protokol.py](teshis/egitim/protokol.py): bu YAML'ı okuyan
  `egitim_kwargs()` yardımcı fonksiyonu.
- [teshis/egitim/kos.py](teshis/egitim/kos.py), [scripts/local_d2b.py](scripts/local_d2b.py),
  [scripts/local_d3.py](scripts/local_d3.py), [scripts/kaggle_d2a.py](scripts/kaggle_d2a.py),
  [scripts/kaggle_d2b.py](scripts/kaggle_d2b.py): hepsi artık hiperparametreleri
  kendi içlerinde tekrar tanımlamak yerine `egitim_kwargs()` üzerinden alıyor.
  Kaggle scriptleri repo kökü Kaggle'a yüklendiği için `teshis` paketini
  `sys.path` üzerinden import edebiliyor.
- **Açık iş / kullanıcı kararı gerekiyor:** D1'in mevcut sonucu (README'deki
  D1 tablosu, `results.csv` satırı `d1_20260817_222323`) düzeltilen protokolle
  yeniden üretilmedi. D1'i D2a/D2b/D3 ile kesin biçimde karşılaştırmadan önce
  D1'in bu ortak protokolle yeniden koşulması önerilir (~90 dakika GPU
  süresi). Bu yeniden koşu kullanıcı onayı olmadan başlatılmadı.
- `.gitignore`: `reports/`, `experiments/`, `veri_surumleri/` için tutarsız
  bir politika vardı — bazı kanıt dosyaları (`d2b_sonuc`, `llm_trial`) elle
  force-add edilmişti, diğerleri (`d1_sonuc`, `d2a_sonuc`,
  `model_karsilastirma*`, çoğu veri sürümü manifesti) hiç Git'e girmemişti.
  Artık kural tutarlı: her senaryonun `manifest.json`, `data.yaml`,
  `run_manifest.json`, `args.yaml`, `results.csv` ve rapor JSON/CSV'leri
  otomatik olarak izleniyor; ağır PNG/JPG/HTML çıktılar ve kopyalanan
  dataset görüntü/etiket klasörleri kasıtlı olarak Git dışı kalmaya devam
  ediyor (repo boyutu ve orijinal dataset'e dokunmama kuralı korunuyor).

## 2. Mevcut Durum

Tamamlananlar:

- [x] Proje klasor iskeleti ve PROJECT_STRUCTURE.md dosya sozlesmesi.
- [x] Dataset taramasi: 21.846 goruntu, 154.141 bbox.
- [x] Sinif toplamlar: tasit 95.877, insan 57.802, UAP 268, UAI 194.
- [x] Yetim goruntu/etiket ve gecersiz YOLO satiri kontrolu.
- [x] Kilitli val_diagnostic: 1.056 goruntu, 4.014 bbox.
- [x] val_diagnostic bbox: tasit 1.264, insan 2.718, UAP 15, UAI 17.
- [x] main_model.pt secilen ana model olarak belirlendi.
- [x] D1 veri surumu uretildi.
- [x] GitHub private repository'ye ilk commit push edildi.

Aktif:

- [x] D1 egitimi tamamlandi.
- [x] D1 best.pt val_diagnostic ile degerlendirildi.
- [x] D1 ile saglikli model farki raporlandi.
- [x] D2a lokalizasyon gurultusu kosusu ve diagnostic raporu tamamlandi.
- [x] D2b eksik etiket kosusu iki baslangic modeliyle tamamlandi.
- [x] Gemini 3.6 Flash ile anonim LLM pilotu yapildi.
- [x] Gemini pilotu gizli cevap anahtariyla otomatik puanlandi.
- [x] Streamlit ara sunum demosu kuruldu ve localhost'ta dogrulandi.
- [~] D3 UAP/UAI sinif karisikligi veri surumu ve yerel GPU kosusu baslatildi.

Sonraki isler:

- [ ] D1 hata galerisi.
- [x] D2a veri senaryosu.
- [x] D2b veri senaryosu.
- [ ] D3-D6b veri senaryolari.
- [ ] E1-E4 egitim senaryolari.
- [ ] Bootstrap guven araliklari ve ajan karar akisi.
- [ ] Final test: sadece bir kez.

## 3. Degismez Kurallar

1. Orijinal dataset: C:/Users/ASUS/Desktop/HYZ/dataset.
2. Orijinal dataset silinmez, tasinmaz ve etiketleri degistirilmez.
3. Test final asamasina kadar kullanilmaz.
4. val_diagnostic gelistirme sirasinda degistirilmez.
5. Her senaryo tek bir hedef bozulma uygular.
6. Her veri degisikligi manifest.json ile kaydedilir.
7. Her egitim ayri run_YYYYMMDD_HHMMSS_SENARYO_SEED klasorundedir.
8. Sonuclar genel mAP ile birlikte sinif AP, recall, bbox n ve belirsizlik notu
   icermelidir.
9. UAP/UAI icin bbox n kucuk oldugundan kesin genelleme iddiasi kurulmaz.
10. Aktif kosu varken ikinci kosu baslatilmaz.
11. Dosya yapisi degisecekse once README ve PROJECT_STRUCTURE guncellenir.

## 4. Sabit Yollar

Proje: C:/Users/ASUS/Desktop/termal_teshis

Dataset: C:/Users/ASUS/Desktop/HYZ/dataset

Modeller:

- C:/Users/ASUS/Desktop/termal_teshis/main_model.pt
- C:/Users/ASUS/Desktop/termal_teshis/final_best.pt

Tanı seti:

- C:/Users/ASUS/Desktop/termal_teshis/val_diagnostic/data.yaml
- C:/Users/ASUS/Desktop/termal_teshis/val_diagnostic/manifest.json

GitHub: https://github.com/pnarkz/teshis_llm

## 5. Dosya Yapisi

Ana Python paketi:

- teshis/veri: istatistik, manifest, veri surumu ve val uretimi.
- teshis/egitim: egitim kosucusu ve kosu manifestleri.
- teshis/degerlendirme: model, metrik, hata ve kanit analizi.
- teshis/ajan: ajan araclari, semalar ve puanlama.
- teshis/servis: API, log, ariza ve anomali katmani.

Senaryolar:

- senaryolar/katalog.yaml: tek resmi senaryo listesi.
- senaryolar/veri: D1, D2a, D2b, D3, D4, D5, D6a, D6b.
- senaryolar/egitim: E1, E2, E3, E4.

Ciktilar:

- veri_surumleri: manifest ve data.yaml tabanli veri varyantlari.
- experiments: egitim kosulari; Git disi.
- reports: agir metrik ve gorsel raporlar; Git disi.
- val_diagnostic: kilitli tanı val; Git disi.

Dosya adlarinin ayrintili sozlesmesi PROJECT_STRUCTURE.md dosyasindadir.

## Ek. Uyumlu Baslangic Komutlari

## Baslangic

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m teshis.veri.istatistik --config config.yaml
python -m teshis.veri.val_olustur --config config.yaml
```

Model degerlendirmesi (Ultralytics kurulu bir ortamda):

```powershell
python -m teshis.degerlendirme.model --model C:\path\to\best.pt --data val_diagnostic\data.yaml
```

Iki modeli karsilastirma:

```powershell
python -m teshis.degerlendirme.karsilastir `
  --referans C:\Users\ASUS\Desktop\termal_teshis\final_best.pt `
  --aday C:\Users\ASUS\Desktop\termal_teshis\main_model.pt `
  --val-root C:\Users\ASUS\Desktop\termal_teshis\val_diagnostic
```

Kontrollu fine-tune kosusu (Kaggle GPU):

```powershell
python -m teshis.egitim.v3_iyilestirme `
  --dataset C:\path\to\dataset `
  --model C:\path\to\final_best.pt `
  --val-root C:\path\to\val_diagnostic
```

Ana referans: `proje-brifingi-v2.2.md`

D1 egitimi bittikten sonra tanisal val degerlendirmesi:

```powershell
python -m teshis.degerlendirme.d1_sonuc `
  --model experiments\run_YYYYMMDD_HHMMSS_D1_42\weights\best.pt `
  --data val_diagnostic\data.yaml `
  --output reports\d1_sonuc `
  --imgsz 768
```

## Proje duzeni

```text
teshis/                 Python paketleri
  veri/                 veri istatistikleri ve surumleme
  degerlendirme/        metrik ve hata analizleri
  egitim/               egitim ve deney kaydi
  ajan/                 araclar ve teshis ajani
  servis/               calisma zamani servisi
senaryolar/             kontrollu ariza konfigurasyonlari
veri_surumleri/         uretilen veri surumleri
experiments/            kosu ciktilari
reports/                raporlar
```

---

## Proje Takip Plani

Bu bolum yapilacaklari adim adim takip etmek icindir. Tamamlanan madde [x]
olarak isaretlenir ve kanit dosyasi ayni commit'e eklenir.

### A. Temel Kurulum

- [x] Proje kok klasoru ve Python paket yapisi.
- [x] Lokal/Kaggle yollarinin config.yaml icinde ayrilmasi.
- [x] Git repository ve private GitHub remote kurulumu.
- [x] Model, dataset ve ciktilarin .gitignore ile dislanmasi.
- [x] Ilk Git commit: 41efd5b.

### B. Veri Sagligi

- [x] Dataset goruntu, etiket ve bbox sayilari.
- [x] Sinif ID dagilimi.
- [x] Yetim goruntu/etiket kontrolu.
- [x] Gecersiz YOLO satiri kontrolu.
- [x] reports/veri_raporu.json.
- [x] Kilitli val_diagnostic.
- [ ] Kaynak grup ve background sizintisi raporu.
- [ ] Bootstrap icin goruntu-birimli manifest.

### C. Model Baseline

- [x] final_best.pt eski referans olcumu.
- [x] main_model.pt aday olcumu.
- [x] 640 ve 768 inference ayrimi.
- [x] reports/model_karsilastirma_fair baseline raporu.
- [ ] Baseline hata galerisi.
- [ ] Baseline sinif guven araliklari.

### D. D1 Sinif Yetersizligi

- [x] Config ve manifest-only veri surumu.
- [x] Hedef class: insan, ID 1.
- [x] Insan frame: 9.289 -> 929.
- [x] Train frame: 17.515 -> 9.155.
- [x] Kaynak dataset degistirilmedi.
- [x] Kosu: experiments/run_20260817_222323_D1_42.
- [x] Egitim tamamen bitti.
- [x] weights/best.pt bulundu.
- [x] D1 diagnostic val degerlendirmesi.
- [x] D1-baseline fark raporu.
- [ ] D1 hata galerisi.

D1 sonucu:

| Olcum | main_model baseline | D1 best.pt | Fark |
|---|---:|---:|---:|
| mAP50 | 0.9331 | 0.9061 | -0.0270 |
| mAP50-95 | 0.6988 | 0.6703 | -0.0286 |
| Precision | 0.8975 | 0.8870 | -0.0104 |
| Recall | 0.8786 | 0.8558 | -0.0228 |
| Insan AP50 | 0.8499 | 0.8160 | -0.0340 |
| Insan recall | 0.8157 | 0.7200 | -0.0957 |

D1 modelinin insan recall'i baseline'a gore belirgin dustu. Bu, insan egitim
karelerinin yuzde 90'i cikarilinca modelin daha fazla insan kacirdigini
gosterir. UAP/UAI sonuclari 15 ve 17 bbox'a dayandigi icin bu sinifler hakkinda
guclu genelleme iddiasi kurulamaz.

Son model: experiments/run_20260817_222323_D1_42/weights/best.pt
Son rapor: reports/d1_sonuc/d1_metrics.json

### E. Senaryolar

- [x] D2a lokalizasyon gurultusu.
- [ ] D2b eksik etiket.
- [ ] D3 UAP/UAI class 2-3 karisikligi.
- [ ] D4 kucuk nesne sinyal kaybi.
- [ ] D5 kaynak/alani kaymasi.
- [ ] D6a split sizintisi.
- [ ] D6b tekrar agirligi ve efektif n.
- [ ] E1 overfitting.
- [ ] E2 underfitting.
- [ ] E3 seed bazli kararsizlik.
- [ ] E4 640/768 uyumsuzlugu.

### D2a Sonucu

- [x] Kaggle kosusu: experiments/run_D2a_42.
- [x] Model: experiments/run_D2a_42/weights/best.pt.
- [x] Diagnostic degerlendirme: reports/d2a_sonuc/d2a_metrics.json.
- [x] Hata galerisi: reports/d2a_hata_galerisi/index.html.
- [x] Test seti kullanilmadi.

| Olcum | main_model baseline | D2a | Fark |
|---|---:|---:|---:|
| mAP50 | 0.9331 | 0.8877 | -0.0454 |
| mAP50-95 | 0.6988 | 0.6162 | -0.0826 |
| Precision | 0.8975 | 0.8854 | -0.0121 |
| Recall | 0.8786 | 0.8317 | -0.0469 |

D2a hipotezi desteklendi: bbox merkezleri kaydirilinca tam ortusmeyi olcen
mAP50-95, gevsek ortusmeyi olcen mAP50'den daha fazla dustu. Bu, lokalizasyon
etiket gurultusunun tespit performansini bozduguna dair orta guclukte kanittir.
UAP/UAI bbox sayilari 15 ve 17 oldugu icin bu sinifler icin guclu genelleme
iddiasi kurulmaz.

D2a hata galerisi 1.056 diagnostic goruntuyu taradi ve en yuksek hata skoruna
sahip 50 goruntuyu kaydetti. Yesil kutular gercek etiketleri, kirmizi kutular
model tahminlerini gosterir. Galeri su dosyalardan incelenir:

- reports/d2a_hata_galerisi/index.html
- reports/d2a_hata_galerisi/gallery.json
- reports/d2a_hata_galerisi/images/

### D2b Ilk Sonucu

- [x] Yerel GPU kosusu: experiments/run_D2b_42_local.
- [x] Model: experiments/run_D2b_42_local/weights/best.pt.
- [x] Diagnostic degerlendirme: reports/d2b_sonuc/d1_metrics.json.
- [x] Orijinal dataset degistirilmedi; train etiket satirlarinin yuzde 25'i
  kontrollu olarak eksiltildi.
- [x] Test seti kullanilmadi.

| Olcum | main_model baseline | D2b | Fark |
|---|---:|---:|---:|
| mAP50 | 0.9331 | 0.9070 | -0.0261 |
| mAP50-95 | 0.6988 | 0.6550 | -0.0438 |
| Precision | 0.8975 | 0.8088 | -0.0887 |
| Recall | 0.8786 | 0.9023 | +0.0236 |

D2b ilk sonucu, eksik etiketlerin precision'i belirgin dusurdugunu ve modelin
daha fazla yanlis pozitif urettigini gosteriyor. Recall'in artmasi, eksik
etiketli train verisinin modeli daha genis tahmin yapmaya itmis olabilecegiyle
uyumludur.

- reports/d2b_sonuc/d1_metrics.json
- reports/d2b_sonuc/d2b_val_diagnostic/confusion_matrix.png

### D2b Final_best Karsilastirmasi

- [x] Ayni D2b bozuk veri protokolu final_best.pt ile tekrarlandi.
- [x] Model: experiments/run_D2b_42_final_best_local/weights/best.pt.
- [x] Diagnostic degerlendirme: reports/d2b_final_best_sonuc/d1_metrics.json.

| Olcum | main_model D2b | final_best D2b | Fark |
|---|---:|---:|---:|
| mAP50 | 0.9070 | 0.8992 | -0.0077 |
| mAP50-95 | 0.6550 | 0.6491 | -0.0059 |
| Precision | 0.8088 | 0.8324 | +0.0236 |
| Recall | 0.9023 | 0.8724 | -0.0299 |

Bu iki kosu, ayni eksik etiket senaryosunda baslangic modelinin sonucu
etkiledigini gosteriyor. final_best precision'da daha iyi, ancak recall ve
toplam mAP'te main_model D2b kosusunun gerisinde. UAP/UAI siniflarinda bbox
sayisi 15 ve 17 oldugu icin bu sinifler hakkinda guclu genelleme iddiasi
kurulmamalidir.

- reports/d2b_final_best_sonuc/d1_metrics.json
- reports/d2b_final_best_sonuc/d2b_final_best_val_diagnostic/confusion_matrix.png

### Ara Sunum Demosu

Streamlit arayuzu tamamlanan deneyleri egitim calistirmadan gosterir. Baslatma:

```powershell
python -m pip install -r requirements-demo.txt
streamlit run demo/app.py
```

Arayuzde su bolumler bulunur:

- Genel Bakis: kosu tablosu, mAP/precision/recall trendleri ve sinif performans haritasi.
- Proje ve Senaryolar: her senaryonun nasil ve neden olusturuldugu.
- Senaryo Incele: baseline farki, precision-recall dengesi, confusion matrix ve ornek tahminler.
- LLM Ajan: Gemini'nin anonim metrikler uzerinden uretdigi pilot teshisler.

Demo kaynak kodu `demo/`, bagimliliklari `requirements-demo.txt` altindadir.

LLM pilot puanlamasi `scripts/score_llm_trial.py` ile yapilir. Script, Gemini
cevabini yerelde tutulan gizli `answer_key.json` ile karsilastirir. Puanlama
diagnosis, sayisal evidence ve limitations alanlarina esit agirlik verir. Bu
ilk puan bir benchmark degil, ajan gelistirme icin pilot rubriktir. Ilk Gemini
sonucu ortalama `0.833` olmustur; D1 nedenini acikca adlandiramamasi sonraki
prompt ve kanit tasariminin gelistirme hedefidir. Bunun icin ikinci paket
surumunde her anonim kosuya baseline'a gore toplam ve sinif bazli delta alanlari
eklendi. Gemini v2 bu paketle tekrar calistirildi ve skor yine `0.833` oldu.
Model degisimleri kanit olarak kullandi, ancak D1 ve D2b'nin nedensel adini
dogrudan koyamadi. Bu, LLM'nin metrik okuma becerisi ile nedensel veri teshisi
arasinda ayrim oldugunu gosteren kayitli bir sinirliliktir.

### F. Ajan

- [x] Gemini ile anonim metrik yorumlama pilotu.
- [x] LLM icin anonim girdi ve JSON cikti semasi olusturuldu.
- [x] Izinli araclari tanimla: [teshis/ajan/araclar.py](teshis/ajan/araclar.py)
  (kosu_listesini_getir, kosu_metriklerini_getir, baseline_farkini_getir,
  bbox_sayilarini_getir).
- [x] Pilot JSON girdi/cikti semasi olusturuldu.
- [x] JSON girdi/cikti semalarini production semasi olarak sabitle:
  [teshis/ajan/semalar.py](teshis/ajan/semalar.py)::TESHIS_SEMASI +
  ARAC_BILDIRIMLERI (Gemini function-calling formatinda).
- [x] Manifesti ajandan gizle: `araclar.py` hicbir fonksiyonda
  veri surumu manifestini okumaz; yalnizca results.csv ve reports/ metrik
  JSON'larini, anonim kosu_NN kimlikleriyle sunar.
- [x] Kanit, guven ve sinir alanlarini zorunlu kil:
  `semalar.teshis_dogrula` eksik alan, gecersiz confidence degeri, tek
  kanitli veya sayisal olmayan evidence, liste olmayan limitations
  durumlarini programatik olarak reddeder (bkz. tests/test_ajan_semalar.py).
- [~] yetersiz_kanit kararini destekle: sema ve prompt bunu bir secenek
  olarak taniyor; modelin bunu ne zaman secmesi gerektigi canli bir LLM
  kosusuyla henuz gozlemlenmedi.
- [x] Anonim senaryo adlarini ve puanlama cetvelini test et:
  tests/test_ajan_araclar.py, tests/test_ajan_puanlama.py.
- [ ] `teshis/ajan/ajan.py::teshis_uret` (gercek Gemini function-calling
  dongusu) canli bir `GEMINI_API_KEY` ile uctan uca dogrulanmadi — bu kod
  tabaninda internet erisimi yok. Calistirmadan once kucuk bir kosu ile
  (tek `kosu_id`) duman testi yapilmasi onerilir.

### G. Final

- [ ] Model ve parametreleri dondur.
- [ ] Testi sadece bir kez calistir.
- [ ] Test sonucunu commit/model/veri manifesti ile sakla.
- [ ] Mentor raporunu tamamla.
- [x] GitHub main branch'ini guncelle.

## Deney Degismezleri

Her deney senaryo kodu, config, veri surumu, manifest, model, Git commit'i,
seed, imgsz, batch, epoch, device, split sayilari, bbox sayilari, precision,
recall, mAP50, mAP50-95, sinif AP/recall, UAP/UAI bbox n, belirsizlik,
confusion matrix ve hata goruntulerini kaydetmelidir.

Test final asamasina kadar split=test kullanilamaz.

## Baska AI Ajana Okuma Sirasi

1. README.md.
2. PROJECT_STRUCTURE.md.
3. config.yaml.
4. senaryolar/katalog.yaml.
5. reports/veri_raporu.json.
6. reports/model_karsilastirma_fair/model_karsilastirma.json.
7. val_diagnostic/manifest.json.
8. Aktif deneyin run_manifest.json dosyasi.

Ajan bu sirayi okumadan yeni senaryo veya veri bolmesi baslatmamalidir. Aktif
kosu varsa once surec ve log kontrol edilir. Mevcut dosyalar silinmez; yeni
deney ayri klasore yazilir. Her degisiklik sonrasi syntax, Git status ve tekrar
calistirma komutu raporlanir.

## 13. Teknik Olmayan Okuyucu Icin Proje Ozeti

### Bu sistem ne yapiyor?

Sistem bir termal drone fotografina bakiyor ve fotograf icindeki nesneleri
bulmaya calisiyor. Sadece fotografi "arac var" diye siniflandirmiyor; her
nesnenin fotograf icindeki yerini bir dikdortgen kutu ile isaretliyor ve bu
kutunun sinifini yaziyor.

Ornek:

- Bir arac bulunursa: sinif tasit, kutu koordinatlari.
- Bir insan bulunursa: sinif insan, kutu koordinatlari.
- Bir hedef bulunursa: sinif UAP veya UAI, kutu koordinatlari.

Bu nedenle bu proje basit bir goruntu siniflandirma projesi degildir. Modelin
iki isi vardir:

1. Nesnenin nerede oldugunu bulmak.
2. Buldugu nesnenin hangi sinifa ait oldugunu soylemek.

### Bbox nedir?

Bbox, nesneyi cevreleyen bounding box yani dikdortgen kutudur. Dataset'teki
her etiket satiri bir bbox'i temsil eder. "4.014 bbox" dedigimizde 4.014
fotograf degil, fotograflardaki toplam 4.014 etiketli nesne kastedilir.

Bir fotograf birden fazla bbox icerebilir. Bu nedenle goruntu sayisi ile bbox
sayisi her zaman ayni degildir.

### Dataset neden train, val ve test diye ayrilir?

- Train: Modelin ogrendigi fotograflar.
- Val: Egitim sirasinda model ayarlarini ve en iyi agirliklari secmek icin
  kullanilan kontrol grubu.
- Test: Tum secimler bittikten sonra gercek performansi olcmek icin saklanan
  son sinav grubu.

Val model gelistirme sirasinda kullanilabilir. Test kullanilirsa ekip modelin
ayarlarini test sonucuna gore farkinda olmadan degistirebilir. Bu durumda test
artik bagimsiz bir sinav olmaktan cikar. Bu nedenle bu projede test final
asamasina kadar yasaktir.

### val_diagnostic neden var?

Mevcut val klasorunde ayni kaynaktan gelen benzer veya augment edilmis
goruntuler bulunabilir. val_diagnostic bu durumu azaltmak, model gelistirme
sirasinda daha tutarli bir kontrol noktasi olusturmak icin hazirlandi.

Bu set:

- Model gelistirme icin kullanilir.
- Test seti degildir.
- Olusturulduktan sonra degistirilmez.
- Her model ayni set uzerinde karsilastirilir.

### Metrikler nasil okunur?

Precision, modelin "bu nesne var" dedigi tahminlerin ne kadarinin dogru
oldugunu anlatir. Yuksek precision, modelin gereksiz kutu cizme ihtimalinin
daha dusuk oldugunu gosterir.

Recall, gercekte var olan nesnelerin ne kadarinin yakalandigini anlatir.
Yuksek recall, modelin nesne kacirma ihtimalinin daha dusuk oldugunu gosterir.

mAP, hem sinif dogrulugunu hem de kutunun gercek nesneyle ne kadar ortustugunu
birlikte olcen ozet metriktir. mAP tek basina yeterli degildir; bu projede
her zaman sinif bazli AP, recall ve bbox sayisi ile birlikte raporlanir.

Ornegin UAP mAP50 degeri 0.995 olsa bile bu deger sadece 15 bbox'a dayaniyorsa
sonucun belirsizligi yuksektir. Bu nedenle "UAP cok iyi" demeden once kac
ornekle olculdugune bakilir.

## 14. Bu Proje Neden Senaryolar Kullaniyor?

Modelin neden basarili veya basarisiz oldugunu normal bir egitim sonucundan
anlamak zordur. D1-D6 ve E1-E4 senaryolari kontrollu deneylerdir. Her deneyde
tek bir problem kasitli olarak olusturulur.

Ornekler:

- D1'de insan iceren egitim fotograflarinin cogu cikartilir. Amac, insan
  sinifi az olursa modelin ne kadar bozuldugunu olcmektir.
- D2b'de bazi etiketler eksik birakilir. Amac, etiketleme hatasinin etkisini
  olcmektir.
- D3'te UAP ve UAI etiketleri karistirilir. Amac, sinif ID hatasinin etkisini
  olcmektir.
- D5'te farkli kaynaklardan gelen goruntuler arasindaki alan farki incelenir.

Her senaryoda saglikli referans ile bozuk veri kosusu karsilastirilir. Boylece
"model kotu oldu" yerine "bu spesifik veri problemi recall'i su kadar
dusurdu" gibi olculebilir bir sonuc elde edilir.

## 15. D1'i Basit Dille Anlatim

D1 su soruya cevap arar:

> Egitim dataset'inde insan fotograflarinin buyuk bolumu olmasaydi model
> insanlarin yerini ve varligini yine ogrenebilir miydi?

D1'de:

- Orijinal dataset kopyalanmaz ve degistirilmez.
- Insan iceren train karelerinin yuzde 90'i egitim listesinden cikartilir.
- Val ve test etiketlerine dokunulmaz.
- Ayni main_model.pt ile egitim baslatilir.
- D1 modeli val_diagnostic uzerinde tekrar olculur.
- Insan recall ve insan AP degerlerinin baseline'a gore nasil degistigi
  incelenir.

Bu deney modelin ne kadar iyi oldugunu degil, belirli bir veri eksikligine ne
kadar dayanikli oldugunu gosterir.

## 16. Ajanin Rolu

Ajan modelin yerine gecmez ve kendi basina yeni gercekler uretmemelidir.
Ajanin gorevi:

1. Deney manifestini ve izin verilen raporlari okumak.
2. Saglikli ve bozuk kosunun metriklerini karsilastirmak.
3. Hangi sinifin etkilendigini belirlemek.
4. Hatanin veri mi, etiket mi, egitim mi yoksa alan farki mi olabilecegini
   siniflandirmak.
5. Kararini dosya yolu ve sayisal kanit ile aciklamak.
6. Veri yetersizse yetersiz_kanit demek.

Ornek ajan sonucu:

~~~text
Karar: sinif_yetersizligi
Kanit: D1 manifesti, insan bbox sayisi ve insan recall farki
Guven: orta
Sinir: UAP/UAI bbox sayisi dusuk; bu senaryo onlar hakkinda kanit saglamiyor
Sonraki olcum: D2b eksik etiket senaryosu
~~~

## 17. Mentor Sunumu Icin Kisa Anlatim

Bu proje, termal drone nesne tespit modelinin kontrollu kosullarda neden
bozuldugunu arastiran bir teshis altyapisidir. Once dataset sagligi ve model
baseline'i olculur. Sonra tek bir veri veya egitim problemi kasitli olarak
olusturulur. Ayni model bu bozuk kosulda tekrar egitilir ve kilitli bir
diagnostic validation setinde olculur. Sonuclar sinif bazli metrikler, bbox
sayilari, confusion matrix ve hata goruntuleriyle raporlanir. Son asamada bir
LLM/ajan bu kanitlari kullanarak arizanin nedenini ve guven seviyesini aciklar.

Bu yaklasimin ana farki, sadece daha yuksek mAP aramak yerine modelin hangi
kosullarda guvenilirligini kaybettigini olcmeye calismasidir.

## Sonraki Tek Adim

1. D1 hata galerisi uret ve insan kacirma orneklerini incele.
2. Hata galerisi sonucunu rapora ekle.
3. D1 checklist kutularini tamamla.
4. D2a'ya gec.

## Kaggle D2a Calistirma

Kaggle Notebook ayarlarinda GPU acilmali ve iki input eklenmelidir:

1. Dataset: images/train, labels/train ve diger splitleri iceren dataset.
2. Model: main_model.pt.

Sonra notebook'ta asagidaki komutla script calistirilir:

~~~text
!python /kaggle/working/teshis_llm/scripts/kaggle_d2a.py
~~~

Private GitHub reposu Kaggle'da otomatik okunamiyorsa script dosyasi notebook'a
yuklenebilir veya repository public olmayan durumda Kaggle Dataset olarak
eklenebilir. Script input altinda main_model.pt ve dataset klasorlerini kendisi
arar.

D2a ciktilari:

~~~text
/kaggle/working/v02_d2a_lokalizasyon_gurultusu/data.yaml
/kaggle/working/v02_d2a_lokalizasyon_gurultusu/manifest.json
/kaggle/working/experiments/run_D2a_42/weights/best.pt
~~~

D2a'da her train bbox merkez koordinati, seed=42 ve center_shift_ratio=0.15
ile kontrollu olarak kaydirilir. Sinif ID'si, bbox genisligi ve yuksekligi
degistirilmez. Manifest ortalama kaymayi, degisen satir sayisini ve kaynak
etiket hash'ini kaydeder. Kaggle sonucu indirildikten sonra model yerelde
val_diagnostic ile degerlendirilmelidir.

## Kaggle D2b Calistirma

D2b, train etiket satirlarinin yuzde 25'ini silerek eksik etiket problemini
simule eder. D2a'daki gibi kaynak dataset salt okunur; val ve test etiketleri
degistirilmez.

Notebook'a scripts/kaggle_d2a.py ve scripts/kaggle_d2b.py dosyalarini birlikte
yukleyin. D2b scripti D2a icindeki ortak input bulma ve dosya linkleme
fonksiyonlarini kullanir.

~~~text
!python /kaggle/working/teshis_llm/scripts/kaggle_d2b.py
~~~

D2b ciktilari:

~~~text
/kaggle/working/v03_d2b_eksik_etiket/manifest.json
/kaggle/working/v03_d2b_eksik_etiket/data.yaml
/kaggle/working/experiments/run_D2b_42/weights/best.pt
~~~

Beklenen kanit: Eksik etiketler nedeniyle modelin recall'i ve ozellikle false
negative sayisi artabilir. Config beklentisi precision dususudur; bu beklenti
ayrica diagnostic val metrikleri ve hata galerisi ile kontrol edilmelidir.

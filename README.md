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

- [ ] D1 egitimi tamamlanacak.
- [ ] D1 best.pt val_diagnostic ile degerlendirilecek.
- [ ] D1 ile saglikli model farki raporlanacak.

Sonraki isler:

- [ ] D1 hata galerisi.
- [ ] D2a-D6b veri senaryolari.
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
- [x] Kosu: experiments/run_20260817_222003_D1_42.
- [ ] Egitim tamamen bitti.
- [ ] weights/best.pt bulundu.
- [ ] D1 diagnostic val degerlendirmesi.
- [ ] D1-baseline fark raporu.
- [ ] D1 hata galerisi.

### E. Senaryolar

- [ ] D2a lokalizasyon gurultusu.
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

### F. Ajan

- [ ] Izinli araclari tanimla.
- [ ] JSON girdi/cikti semalarini sabitle.
- [ ] Manifesti ajandan gizle.
- [ ] Kanit, karsilastirma, guven ve sinir alanlarini zorunlu kil.
- [ ] yetersiz_kanit kararini destekle.
- [ ] Anonim senaryo adlarini ve puanlama cetvelini test et.

### G. Final

- [ ] Model ve parametreleri dondur.
- [ ] Testi sadece bir kez calistir.
- [ ] Test sonucunu commit/model/veri manifesti ile sakla.
- [ ] Mentor raporunu tamamla.
- [ ] GitHub main branch'ini guncelle.

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

## Sonraki Tek Adim

1. experiments/run_20260817_222003_D1_42/weights/best.pt dosyasini kontrol et.
2. teshis.degerlendirme.d1_sonuc komutunu calistir.
3. reports/d1_sonuc/d1_metrics.json dosyasini kontrol et.
4. main_model.pt baseline'i ile D1 farkini raporla.
5. D1 checklist kutularini guncelle.
6. D2a'ya gec.

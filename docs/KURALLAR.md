# Degismez Kurallar ve Sabit Yollar

Bu kurallar proje boyunca degistirilmez. Bir kural degisecekse once burada
degistirilir ve Bakim Gunlugu'ne gerekcesi yazilir.

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
11. Dosya yapisi degisecekse once docs/MIMARI.md guncellenir; adlandirma
    kurallari orada tanimlidir.

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
- (teshis/servis: Asama 2 icin planlandi, henuz baslanmadi; bos iskelet
  tutulmuyor.)

Senaryolar:

- senaryolar/katalog.yaml: tek resmi senaryo listesi.
- senaryolar/veri: D1, D2a, D2b, D3, D4, D5, D6a, D6b.
- senaryolar/egitim: E1, E2, E3, E4.
  E kosulari ortak protokolden KASITLI olarak sapar (bozulan sey veri degil,
  egitim protokoludur). Sapmalar `egitim_protokolu.yaml -> e_serisi` altinda
  BEYAN EDILIR; CLI bayragi veya script icine gomulu sayi olarak degil.
  `kos.py --e-senaryo <kod>` bunlari okur ve kosu manifestine
  `protokol_sapmalari` alani olarak yazar. E kosulari D kosulariyla ayni
  tabloda karsilastirilmaz.

Ciktilar:

- veri_surumleri: manifest ve data.yaml tabanli veri varyantlari.
- experiments: egitim kosulari; Git disi.
- reports: agir metrik ve gorsel raporlar; Git disi.
- val_diagnostic: kilitli tanı val; Git disi.

Dosya adlarinin ayrintili sozlesmesi docs/MIMARI.md dosyasindadir.


## Deney Degismezleri

Her deney senaryo kodu, config, veri surumu, manifest, model, Git commit'i,
seed, imgsz, batch, epoch, device, split sayilari, bbox sayilari, precision,
recall, mAP50, mAP50-95, sinif AP/recall, UAP/UAI bbox n, belirsizlik,
confusion matrix ve hata goruntulerini kaydetmelidir.

Test final asamasina kadar split=test kullanilamaz.

## Baska AI Ajana Okuma Sirasi

1. README.md.
2. docs/MIMARI.md.
3. config.yaml.
4. senaryolar/katalog.yaml.
5. reports/veri_raporu.json.
6. reports/model_secimi/model_karsilastirma.json.
7. val_diagnostic/manifest.json.
8. Aktif deneyin run_manifest.json dosyasi.

Ajan bu sirayi okumadan yeni senaryo veya veri bolmesi baslatmamalidir. Aktif
kosu varsa once surec ve log kontrol edilir. Mevcut dosyalar silinmez; yeni
deney ayri klasore yazilir. Her degisiklik sonrasi syntax, Git status ve tekrar
calistirma komutu raporlanir.

## Kayit defteri (results.csv)

- Satir eklemenin tek yolu `teshis/degerlendirme/kayit_defteri.py`'dir.
  Metrikler ELLE verilmez, olcum dosyasindan okunur; boylece defterdeki sayi
  ile raporun sayisi ayrisamaz.
- `run_id` ve `scenario` BENZERSIZ olmalidir. Iki kosu ayni senaryo adini
  tasidiginda demo ve kanit ureticisi hangisinin hangisi oldugunu ayirt
  edemez (d2b_main / d2b_final bu yuzden karismisti).
- Yeni satirlar **sona** eklenir. Ajanin `kosu_NN` numaralari satir sirasina
  dayanir; ortaya eklenen bir satir tamamlanmis denemeleri gecersiz kilar.
- Uretilmis her olcumun defterde bir satiri olmalidir. Bilincli istisnalar
  `kayit_defteri.DEFTER_DISI` altinda **gerekcesiyle** yazilir.
- Ajanin hangi kosulari gorecegi `araclar.ajana_uygun_mu` yukleminde tek
  kaynakta tanimlidir. Testler bu yuklemi kullanir, kopyalamaz.

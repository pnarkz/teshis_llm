# Termal YOLO Teshis Ajani

Bu proje, termal nesne tespit sisteminin veri, egitim, degerlendirme ve calisma zamani sorunlarini olculebilir kanitlarla teshis eder.

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
  --aday C:\Users\ASUS\Desktop\termal_teshis\best(2).pt `
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

# Calistirma Komutlari

Ortam kurulumu, yerel komutlar ve Kaggle kosulari.

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

Ana referans: `docs/proje-brifingi-v2.1.md` (projenin sartname belgesi)

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


## Kaggle D2a Calistirma

Kaggle Notebook ayarlarinda GPU acilmali ve iki input eklenmelidir:

1. Dataset: images/train, labels/train ve diger splitleri iceren dataset.
2. Model: main_model.pt.

Sonra notebook'ta asagidaki komutla script calistirilir:

~~~text
!python /kaggle/working/teshis_llm/scripts/kaggle_D2a_lokalizasyon_gurultusu.py
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

Notebook'a scripts/kaggle_D2a_lokalizasyon_gurultusu.py ve scripts/kaggle_D2b_eksik_etiket.py dosyalarini birlikte
yukleyin. D2b scripti D2a icindeki ortak input bulma ve dosya linkleme
fonksiyonlarini kullanir.

~~~text
!python /kaggle/working/teshis_llm/scripts/kaggle_D2b_eksik_etiket.py
~~~

D2b ciktilari:

~~~text
/kaggle/working/v03_d2b_eksik_etiket/manifest.json
/kaggle/working/v03_d2b_eksik_etiket/data.yaml
/kaggle/working/experiments/run_D2b_42/weights/best.pt
~~~

Beklenen kanit: Eksik etiketler nedeniyle modelin recall'i ve ozellikle false
negative sayisi artabilir. Config beklentisi precision dususudur; bu beklenti

## Ajan kontrol tekrarlari (kota-bilincli)

Projenin en zayif kanitli iddiasi "ajan bozulma yokken sorun uydurmuyor".
Elde yalnizca iki saf kontrol var; Wilson %95 araligi [0.000, 0.658].

Tam denemeyi tekrarlamak pahali: 11 kosu x ~2.5 istek = bir gunluk ucretsiz
kotanin tamami (20 istek/gun). Bunun yerine yalnizca iki kontrol kosusu
tekrarlanir - ayni kotayla kontrol gozlemi 2'den 8'e cikar:

```bash
python scripts/ajan_kontrol_tekrari.py --tekrar 3
```

Kota biterse script bunu ayirt edip temiz durur; ertesi gun:

```bash
python scripts/ajan_kontrol_tekrari.py --tekrar 3 --devam
```

Ozet (kosmadan): `python scripts/ajan_kontrol_tekrari.py --ozet`

`GEMINI_API_KEY` ortam degiskeni tanimli olmalidir; anahtar hicbir dosyaya
yazilmaz, yalnizca `os.environ` uzerinden okunur.

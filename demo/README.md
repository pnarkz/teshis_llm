# Ara Sunum Konsolu

Bu Streamlit uygulaması tamamlanmış deney raporlarını görselleştirir. Eğitim
veya test çalıştırmaz; bu nedenle ara sunumda hızlı ve tekrarlanabilirdir.

Tasarım yönü **ölçüm aleti / mühendislik konsolu**: monospace tipografi,
keskin köşeler, ince ızgara çizgileri; gradyan, gölge ve yuvarlak köşe yok.
Yalnızca işletim sisteminde hazır bulunan monospace yazı tipleri kullanılır,
bu yüzden sunum sırasında internet olmasa da görünüm bozulmaz.

## Kurulum

Proje klasöründe:

```powershell
python -m pip install -r requirements-demo.txt
streamlit run demo/app.py
```

Tarayıcıda varsayılan adres: `http://localhost:8501`

## Bölümler

| Bölüm | İçerik |
|---|---|
| Genel Bakış | Tüm koşuların readout kartları, epoch sparkline'ları, karşılaştırma tablosu, metrik ve sınıf profili |
| Senaryo İncele | Seçilen koşu için baseline farkı, epoch bazlı eğitim eğrileri, confusion matrix, eşik eğrileri, etiket-vs-tahmin görselleri, bozulmuş eğitim verisi önizlemesi |
| Hata Galerisi | 50 görüntülük D2a hata galerisi; FN/FP/IoU dağılımı ve sıralanabilir kareler |
| Proje ve Senaryolar | Deney tasarımının amacı ve senaryo kataloğu |
| LLM Ajan | Anonim metriklerden üretilen pilot teşhisler ve rubrik puanları |

## Gösterilecek akış

1. Genel Bakış'ta baseline ve tüm koşular tek ekranda gösterilir; sparkline'lar
   her koşunun eğitim boyunca nasıl ilerlediğini özetler.
2. Senaryo İncele'de D1, D2a, D2b ve D3 sırayla seçilerek metrik farkları,
   eğitim eğrisi ve görsel kanıt incelenir.
3. Hata Galerisi'nde modelin gerçekte nerede hata yaptığı somut karelerle
   gösterilir.
4. LLM Ajan bölümünde pilot teşhisler ve puanlama sunulur.
5. Sonuç, "hangi veri problemi hangi metriği etkiledi" sorusuyla açıklanır.

## Veri kaynakları

Konsol hiçbir metrik değerini kendi içinde saklamaz; hepsi şu dosyalardan
okunur:

- `results.csv` — koşu başına özet metrikler.
- `reports/<senaryo>_sonuc/` — diagnostic değerlendirme JSON'u ve görselleri.
- `experiments/<run>/results.csv` — epoch bazlı eğitim eğrileri
  (koşu klasörü `results.csv`'deki `weights_path` sütunundan türetilir, ayrıca
  eşleme tablosu tutulmaz).
- `reports/*_hata_galerisi/` — hata galerileri (otomatik keşfedilir).
- `reports/llm_trial/` — LLM pilot çıktısı ve puanı.

Yeni bir senaryo `results.csv`'ye eklendiğinde konsol onu otomatik tanır;
yalnızca `demo/app.py` içindeki `scenario_info` sözlüğüne açıklama metni
eklemek gerekir.

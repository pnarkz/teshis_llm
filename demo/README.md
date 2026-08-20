# Ara Sunum Demosu

Bu Streamlit uygulaması tamamlanmış deney raporlarını görselleştirir. Eğitim
veya test çalıştırmaz; bu nedenle ara sunumda hızlı ve tekrarlanabilirdir.

## Kurulum

Proje klasöründe:

```powershell
python -m pip install -r requirements-demo.txt
streamlit run demo/app.py
```

Tarayıcıda varsayılan adres: `http://localhost:8501`

## Gösterilecek akış

1. Baseline seçilir ve sağlıklı referans açıklanır.
2. D1, D2a ve D2b seçilerek metrik farkları gösterilir.
3. Sınıf bazlı AP50 ve confusion matrix incelenir.
4. Gemini pilot ajan çıktısı gösterilir.
5. Sonuç, "hangi veri problemi hangi metriği etkiledi" sorusuyla açıklanır.

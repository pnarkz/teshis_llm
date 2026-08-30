# LLM Denemesi — Arsiv (4 kosu, Gemini v2)

Bu klasor, ajanin arac katmani genisletilmeden ONCE yapilan ilk kor denemenin
kaydidir. Paket yalnizca dort kosu (baseline, D1, D2a, D2b) ve yalnizca toplam
metrik + sinif AP50 iceriyordu; boyut/kaynak/sinif-karisikligi kirilimlari
ajana verilmiyordu.

- `llm_input.json`   : LLM'ye verilen paket
- `answer_key.json`  : gizli cevap anahtari
- `gemini_response.json` : modelin cevabi
- `llm_score.json`   : pilot rubrik skoru (mean_score = 0.833)

Bu sonuc tarihsel kayittir ve guncel paketle karsilastirilamaz: guncel paket
dokuz kosu iceriyor ve kirilim kanitlarini da sunuyor. Karsilastirma yapilacaksa
ayni paketle yeniden kosulmasi gerekir.

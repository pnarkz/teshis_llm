# Tests

Modul, manifest ve kör test sozlesmeleri burada test edilir.

Calistirma (proje kokunden):

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

Kapsam:

- `test_ajan_semalar.py`: ajan cikti JSON semasinin dogrulanmasi.
- `test_ajan_araclar.py`: ajanin arac katmaninin (anonim kosu haritasi,
  metrik okuma) katalog.yaml kuralina (manifest/senaryo adi ajana verilmez)
  uydugu.
- `test_ajan_puanlama.py`: pilot puanlama rubriginin alt puanlari ve
  puanlamanin kosu_NN numarasindan bagimsiz oldugu.
- `test_llm_paketi.py`: LLM deneme paketi ile ajan araclarinin ayni kosu
  kumesini gordugu, cevap anahtarinin guncel anonim eslemeyle uyumlu kaldigi
  ve pakette senaryo adi sizmadigi.
- `test_egitim_protokolu.py`: tum D serisi egitim scriptlerinin
  `senaryolar/egitim_protokolu.yaml` tek kaynagini kullandigi, D1'de
  yasanan sessiz lr0/warmup_epochs sapmasinin geri gelmedigi.
- `test_degerlendirme_project_yolu.py`: Ultralytics `project=` yolunun mutlak
  verildigi (goreceli yol ciktilari sessizce `runs/detect/` altina yaziyor).
- `test_veri_istatistik.py`: dataset saglik taramasinin (yetim
  goruntu/etiket, gecersiz sinif, cok kucuk kutu) sentetik bir mini
  dataset uzerinde dogru sayim yaptigi. Gercek dataset'e dokunmaz.
- `test_demo_render.py`: demo konsolunun her sayfa, her senaryo ve her galeri
  siralamasinda istisnasiz render edildigi (Streamlit AppTest ile headless).
  Demo bagimliliklari kurulu degilse atlanir.

Gercek Gemini API cagrisi yapan `teshis/ajan/ajan.py::teshis_uret` ve
`kor_deneme_calistir` bu test paketine dahil degildir; bunlar `GEMINI_API_KEY`
gerektirir ve canli olarak elle dogrulanmalidir.

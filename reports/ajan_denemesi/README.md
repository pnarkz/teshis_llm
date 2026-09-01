# LLM Deneme Paketi

LLM'ye sadece `llm_input.json` verilir. `answer_key.json` gizli tutulur ve
LLM cevabini sonradan puanlamak icin kullanilir. Beklenen cikti,
`required_output` semasina uygun JSON olmalidir.

Bu dosyalar `scripts/ajan_paket_hazirla.py` tarafindan `results.csv` ve
`teshis/ajan/araclar.py` uzerinden uretilir; elle duzenlenmemelidir.

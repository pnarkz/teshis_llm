# Termal Teshis Ajani

Termal drone goruntulerinde YOLO nesne tespit modelinin **kontrollu veri
arizalari** altinda nasil bozuldugunu olcer; sonra bir LLM ajanina bu
olcumleri vererek bozulmanin nedenini **kanita dayali** teshis edip
edemedigini sinar.

**Arastirma sorusu:** Termal nesne tespit sistemi hangi veri, etiket ve
dagilim kosullarinda bozulur; bir LLM bu bozulmayi yeterli kanitla teshis
edebilir mi?

Sinif sozlesmesi degismez: `0 tasit`, `1 insan`, `2 UAP`, `3 UAI`.

---

## Belgeler

| Belge | Icerik |
|---|---|
| [docs/BULGULAR.md](docs/BULGULAR.md) | **Tum senaryo sonuclari.** Otoriter karsilastirma tablosu ve her senaryonun ayrintisi. |
| [docs/MIMARI.md](docs/MIMARI.md) | Dosya/klasor sozlesmesi: neyin nerede oldugu ve adlandirma kurallari. |
| [docs/KURALLAR.md](docs/KURALLAR.md) | Degismez kurallar, sabit yollar, deney degismezleri. |
| [docs/CALISTIRMA.md](docs/CALISTIRMA.md) | Kurulum ve komutlar (yerel + Kaggle). |
| [docs/BAKIM_GUNLUGU.md](docs/BAKIM_GUNLUGU.md) | Kronolojik degisiklik kaydi; her duzeltmenin gerekcesi. |
| [docs/SUNUM.md](docs/SUNUM.md) | Teknik olmayan anlatim ve mentor sunumu. |

---

## Yontem ozeti

1. **Saglikli referans (v00):** veri hic bozulmadan, senaryolarla ayni
   protokolde egitilir.
2. **Karsilastirilabilirlik:** bir fark ancak aday ile referans **dort kimlik
   alaninda da** ayni ise bozulmaya atfedilebilir — baslangic modeli,
   degerlendirme kumesi, cikarim cozunurlugu, checkpoint. Her olcegin kendi
   referansi ve kendi gurultu esigi vardir; baska olcegin esigi odunc
   alinmaz (`teshis/degerlendirme/karsilastirilabilirlik.py`).
3. **Tek degisken:** her senaryoda yalnizca bir veri arizasi uygulanir;
   egitim protokolu (`senaryolar/egitim_protokolu.yaml`) sabittir.
4. **Kilitli olcum seti:** kosular `val_diagnostic` setinde olculur
   (1.056 goruntu, 4.014 bbox). Tek istisna D6a'dir: sizintinin olcumu ne
   kadar iyimser yaptigini gostermek icin **kasitli olarak** sizintili kume
   uzerinde degerlendirilir ve bu yuzden digerleriyle ayni tabloda okunmaz.
   Test seti final asamaya kadar kullanilmaz.
5. **Gurultu tabani:** hicbir sey bozulmadan, yalnizca seed degistirilerek
   egitilen kontrol kosulari arasindaki yayilim olculur. Bu bandin altinda
   kalan bir fark, buyuklugu ne olursa olsun rastgelelikten ayirt edilemez.
6. **Kirilimli okuma:** toplam mAP bazi bozulmalari tamamen gizler; sinif,
   nesne boyutu ve veri kaynagi kirilimlariyla birlikte okunur.
7. **Kor teshis:** ajan senaryo adlarini gormez, yalnizca anonim `kosu_NN`
   metriklerini arac cagirarak okur.

## Tamamlanan senaryolar

| Kod | Bozulma | Bulgunun ozeti |
|---|---|---|
| D1 | Sinif yetersizligi (insan karelerinin %90'i) | main_model kurgusunda **etkisiz** (z=-1,22); yolo26n kurgusunda **guclu** (z=-18,58) |
| D2a | Lokalizasyon etiket gurultusu | mAP50-95 en cok duser (-0.0544) |
| D2b | Eksik etiket (%25) | precision -0.1087, recall artar: model fazladan kutu uretir |
| D3 | UAP/UAI sinif karisikligi | precision -0.2047; nadir sinif cokuyor ama n=15/17 |
| D3b | tasit/insan sinif karisikligi | bol veride bozulma **sogurulur** (capraz hata 2 -> 4 kutu) |
| D4 | Kucuk nesne sinyal kaybi | yalnizca <16px bandi coker (-0.4524, z=-21,9); diger bantlar degismez |
| D5 | Kaynak/alan kaymasi | `best.pt`'de gorunmez; `last.pt`'de egitilmeyen kaynaklar coker |
| D6a | Split sizintisi | sizintili val mAP50-95'i +0.0287 sisirir — bircok gercek bozulmadan buyuk |
| D6b | Tekrar agirligi | temsil payi ile performans arasinda monotonik iliski; temsil edilmeyen sinif coker |

Ayrintilar ve sayilar icin: **[docs/BULGULAR.md](docs/BULGULAR.md)**

## Projenin uc ana dersi

1. **Karsilastirma tabani yanlissa tum sonuclar yanlistir.** Fine-tune
   edilmemis bir modele gore olcum yapmak, bozulma etkisi ile fine-tune
   etkisini birbirine karistirir (Bakim Gunlugu 2026-08-26). Ayni hata daha
   sinsi bir bicimde tekrarladi: farkli checkpoint veya farkli baslangic
   modeliyle uretilmis **saglikli** kosular tek bir referansla tartilinca
   "guclu bozulma kaniti" gorundu (2026-09-07).
2. **Toplam mAP yalan soyleyebilir.** D3b, D4 ve D5'in tamami toplam
   metriklerde gorunmez; yalnizca dogru kirilimla ortaya cikar.
3. **Olcum setinin temizligi ve cesitliligi metodolojinin merkezindedir.**
   D5 kaynak cesitliligini, D6a ise sizintiyi gosterir.

## Hizli baslangic

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

Demo konsolu:

```bash
streamlit run demo/app.py
```

Diger komutlar: **[docs/CALISTIRMA.md](docs/CALISTIRMA.md)**

## Durum

- **Tamamlandi:** D serisinin tamami (D1, D2a, D2b, D3, D3b, D4, D5, D6a,
  D6b), v00 saglikli referans, yolo26n kontrol cifti, ajan arac katmani,
  tek atislik LLM denemesi (9 kosu), E4 cozunurluk uyumsuzlugu,
  E2 (negatif sonuc: yakinsamis modelde epoch kesmek underfitting uretmiyor),
  E1 (asiri uyum gerceklesti; best.pt onu tamamen gizliyor),
  E3 (negatif sonuc: 100 kat lr kararsizlik degil tam iraksama uretti),
  E3b (kararsizlik olculdu: seed'e gore mAP50'de 29 kat oynaklik),
  **C2 negatif kontrolu** (seed 7, 13, 21 — gurultu tabani n=3), ajan
  denemesi (tek atislik vs fonksiyon cagirma karsilastirmasi).
- **Devam ediyor:** yok.
- **Tamamlandi:** ajan denemesi 11/11 puanlandi (`mean_score` 0.833). Ajan
  saf kontrolde (Baseline, C2) **sorun uydurmadi** (2/2); baskin hata turu
  yanlis neden atfetmek ve kacirmak. Onceki bulgu gecerli:
  fonksiyon cagirma ile tek atislik arasinda **olculebilir fark yok**
  (teshis 0.500 vs 0.444; dokuz kosuda yarim kosu). Arac kullanim orani 1.0.
- **Yapilmadi:** Asama 2 (calisma zamani servisi), final test kosusu.

- **Sartname boslugu:** Gurultu tabani uc kontrol kosusuna cikarildi ve
  **yedi iddia zayifladi**; D6b artik hicbir metrikte gurultuyu asmiyor.
  (Bu sayi artik elle yazilmiyor: demonun karsilastirma sayfasi defterden
  turetir. Onceden "bes" yaziyordu ve E1 ile E2 atlanmisti.)
  Kanit sozlesmesi (`kanit.json`) 26/26 kosuda TAM. Yayimlanmis guven
  araliklari sartnamedeki tabakali bootstrap yerine Wilson ile hesaplandi
  (~1.5 kat dar). Ayrinti: [docs/BULGULAR.md](docs/BULGULAR.md)
  'Sartnameye Uyum Denetimi'.

- **Karsilastirilabilirlik (2026-09-07):** Demo butun kosulari tek bir
  referansla (`v00_saglikli`) karsilastiriyordu ve bu, hicbir bozulma
  icermeyen kosulari "guclu bozulma kaniti" olarak etiketliyordu
  (`v00_saglikli last_pt`, `v00n`). Artik her kosu yalnizca KENDI
  olcegindeki referansla karsilastirilir - ayni baslangic modeli,
  degerlendirme kumesi, cozunurluk ve checkpoint
  (`teshis/degerlendirme/karsilastirilabilirlik.py`). Ayni filtre ajan
  araclarinda zaten vardi; kural iki yerde yasayinca biri geride kaldi.

Guncel ayrinti: [docs/BAKIM_GUNLUGU.md](docs/BAKIM_GUNLUGU.md)

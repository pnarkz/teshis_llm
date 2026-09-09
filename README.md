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

Sunumda kod bulmak icin: **[Kod haritasi — senaryo, uygulama ve sonuc](docs/KOD_HARITASI.md)**.

| Belge | Icerik |
|---|---|
| [docs/BULGULAR.md](docs/BULGULAR.md) | **Tum senaryo sonuclari.** Otoriter karsilastirma tablosu ve her senaryonun ayrintisi. |
| [docs/MIMARI.md](docs/MIMARI.md) | Dosya/klasor sozlesmesi: neyin nerede oldugu ve adlandirma kurallari. |
| [docs/KURALLAR.md](docs/KURALLAR.md) | Degismez kurallar, sabit yollar, deney degismezleri. |
| [docs/CALISTIRMA.md](docs/CALISTIRMA.md) | Kurulum ve komutlar (yerel + Kaggle). |
| [docs/BAKIM_GUNLUGU.md](docs/BAKIM_GUNLUGU.md) | Kronolojik degisiklik kaydi; her duzeltmenin gerekcesi. |
| [docs/SUNUM.md](docs/SUNUM.md) | Teknik olmayan anlatim ve mentor sunumu. |
| [docs/SUNUM_SENARYOSU.md](docs/SUNUM_SENARYOSU.md) | Konsolu sayfa sayfa gezerken ne gosterilecek ve ne soylenecek. |

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

### Sunum konsolu (baska bir makinede de calisir)

```bash
git clone https://github.com/pnarkz/teshis_llm.git
cd teshis_llm
python -m pip install -r requirements-demo.txt
python -m streamlit run demo/app.py
```

Konsol **yalnizca depoyla gelen olcum ciktilarini okur**; egitim veya test
calistirmaz. Model agirliklari (`*.pt`), kilitli tani seti
(`val_diagnostic/`) ve tam gorsel arsivi (`reports/` altinda 233 MB) Git
disidir - taze bir klonda bunlar bulunmaz ve BULUNMAK ZORUNDA DEGILDIR:

| Eksik olan | Konsol ne yapar |
|---|---|
| `val_diagnostic/` | Etiketli ornek galerisi `demo/assets/ornekler` altindaki tasinabilir alt kumeyi kullanir ve bunu ekranda yazar |
| `reports/**/images` | Hata galerisi `demo/assets/sunum_gorselleri` altindaki kucultulmus seti kullanir ve bunu ekranda yazar |
| `val_batch*.jpg` | "Ornek tahminler" bolumu kendini atlar |
| `*.pt` agirliklari | Hicbir sey; konsol model calistirmaz |

Kenar cubugundaki **Sistem durumu** paneli hangi ciktinin bulundugunu tek
bakista soyler.

### Canli ajan (istege bagli)

```bash
python -m pip install "google-genai>=1.0"
```

`GEMINI_API_KEY` bir **ortam degiskeni** olarak tanimlanir; hicbir zaman
depoya yazilmaz. PowerShell'de:

```powershell
$env:GEMINI_API_KEY = "..."
```

Ajan bolumu anahtarsiz da calisir - kayitli kosu modu varsayilandir ve API
harcamaz. Canli mod secildiginde once bir on kontrol calisir ve hangi
kosulun eksik oldugunu yazar.

### Egitim ve olcum (GPU gerekir)

```bash
python -m pip install -e ".[egitim]"
cp config.example.yaml config.local.yaml   # kendi veri yollarinizi yazin
```

`config.local.yaml` Git disidir; yollar goreli yazilabilir ve proje kokune
gore cozulur.

### Testler

```bash
python -m pip install -r requirements-dev.txt -r requirements-demo.txt
python -m pytest -q
```

Taze bir klonda da gecer: kilitli tani setine baglı testler otomatik olarak
atlanir.

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

- **Esdegerlik metni tek kaynaktan (2026-09-09):** E4 ve D6a "eslenik
  olcum"dur: aday ile referans AYNI agirlik dosyasini kullanir, yalnizca
  cikarim ayari degisir. Egitim rastgeleligi devrede olmadigi icin bu
  kosularda gurultu esigi uygulanmaz. Bu istisna uc ekranda ayri ayri
  yazilmisti ve E4 sayfasinda cozunurluk "sabit" gorunuyordu - oysa DEGISEN
  buydu. Metin artik tek yerden uretilir
  (`karsilastirilabilirlik.gurultu_esigi_gecerli_mi` ve
  `esik_yoklugu_aciklamasi`).

- **Hipotez hukumlerinin kapsami (2026-09-09):** Tablo "hipotez
  desteklendi" diyordu; hesaplanan sey ise "fark gurultu bandini asti mi"
  idi. Etiketler artik yalnizca hesaplanani soyler ("Esigi asan dusus",
  "Beklenmedik yonde etki", "Gurultuyu asan etki yok", "Eslenik olcum",
  "Olculemedi"). D1'de mAP50_95'teki YUKSELIS "kismen desteklendi" olarak
  okunuyordu.

- **Senaryo ile kosu ayrimi (2026-09-09):** Katalog bir deney defteri degil;
  senaryo bir hipotez, kosu ise onun bir kaydidir. Ayrim artik Senaryolar,
  Karsilastirma ve Sonuclar sayfalarinda ayni sekilde uygulanir
  (`demo/katalog.py`); Karsilastirma once senaryo, sonra kosu sorar.

- **Yeni deney duzeni (2026-09-09):** Eski ana denemede ustveri yoktu: hangi
  cevabin hangi kod haliyle uretildigi kesin bilinmiyordu ve **102 arac
  cagrisinin hicbirinin cevabi saklanmamisti**. Yeni kosucu
  (`scripts/ajan_deney.py`) her gozlemi degismez bir deney kimligi
  (`<UTC>__<git sha>`) altinda; model, arac surumu, Git commit'i, calisma
  parametreleri, HAM ve ayristirilmis cevap ve **her arac cevabinin
  snapshot'i** ile kaydeder. Puanlama ayri bir adimdir
  (`scripts/ajan_deney_puanla.py`): cevap anahtari uretim tarafina hic
  girmez, ayni uretim iki kez sayilmaz, gecerli tekrarlar korunur ve sonuclar
  **rol bazli** raporlanir - kontrol kosulari ile bozulma senaryolari tek bir
  basari oraninda birlestirilmez. Yeni deneyler `reports/ajan_deneyleri/`
  altindadir ve eski `reports/ajan_denemesi/` sonuclariyla karistirilmaz.

- **Ilk tekrarli ajan deneyi (2026-09-09):** 13 kosu x 3 tekrar = 39
  gozlem, deney kimligi `20260909T120445Z__a15487c9`. Rol bazli, birlestirilmeden:
  saglikli referans 1.000 (n=3), kontrol 1.000 (n=9), bozulma senaryosu
  kati 0.389 / tespit-farkindalikli 0.722 (n=27).

  Iki bulgu one cikiyor. Birincisi: **13 kosunun 13'u de uc tekrarinda ayni
  hukmu verdi** - sozel ifade degisiyor, hukum degismiyor. Ikincisi:
  kontrol kosularinda 9/9, Wilson %95 [0.701, 1.000]. Eski denemede dort
  kontrolun birinde uydurmustu (kosu_12). Bu bir "model gelisti" bulgusu
  degil: araclar artik her alt grup farkina gurultu bandini ekliyor, yani
  yanlis pozitifi onleyen sey ona **gurultu tabanini gostermek**. Bu, ana
  tezin ajan tarafindaki karsiligi.

  Deney eski `reports/ajan_denemesi` sonuclariyla BIRLESTIRILMEZ; ikisi
  farkli araclarla olculdu. Konsolun ajan sayfasi ikisini ayri gosterir.

Guncel ayrinti: [docs/BAKIM_GUNLUGU.md](docs/BAKIM_GUNLUGU.md)

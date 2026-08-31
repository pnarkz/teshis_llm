# Senaryo Bulgulari

Her senaryonun olcum sonuclari ve yorumu. Otoriter karsilastirma tablosu
"v00 Saglikli Referans" bolumundedir; tek tek senaryo bolumleri o tablonun
ayrintisini verir.

Istatistikler `teshis/degerlendirme/bootstrap.py` ile yeniden uretilebilir.

### v00 Saglikli Referans ve Senaryo Karsilastirmasi (OTORITER TABLO)

Bu bolum projenin ana sonuc tablosudur. Yukaridaki senaryo bolumleri her
senaryonun kendi hikayesini anlatir, ancak **karsilastirma icin bu tablo
kullanilmalidir.**

**Neden v00 gerekliydi.** Onceki tum senaryo tablolari, fine-tune edilmemis
`main_model.pt`'ye gore hesaplanmisti. Bu, her farki
`(bozulma etkisi) + (fine-tune etkisi)` toplami yapiyordu. v00, veriyi hic
bozmadan senaryolarla **birebir ayni protokolde** egitilmis referanstir; ikisi
arasindaki fark artik yalnizca bozulmanin kendisidir.

Fine-tune'un tek basina etkisi kucuk degildi:

| Olcum | main_model (fine-tune yok) | v00 (temiz fine-tune) | fark |
|---|---:|---:|---:|
| mAP50 | 0.9331 | 0.9200 | -0.0132 |
| mAP50-95 | 0.6988 | 0.6707 | -0.0282 |
| Precision | 0.8975 | 0.9175 | +0.0201 |
| Recall (genel) | 0.8786 | 0.8785 | -0.0001 |
| **insan recall** | **0.8157** | **0.7391** | **-0.0765** |

Genel recall'in degismemesine karsilik insan recall'inin 7,7 puan dusmesi
dikkat cekicidir: fine-tune, recall'i siniflar arasinda yeniden dagitmis
(insan duserken UAI +0.1126 yukselmis), toplamda ise sabit birakmistir.

Yani temiz fine-tune bile insan recall'ini 7,7 puan dusuruyor (precision'i
yukseltip recall'i dusuren bir calisma noktasi kaymasi). Bu deger, D1'in
manset iddiasinin neredeyse tamaminin aslinda fine-tune etkisi oldugunu
gosterir.

**Senaryolarin izole edilmis etkisi (v00'a gore):**

| Senaryo | mAP50 | mAP50-95 | Precision | Recall | Hipotez |
|---|---:|---:|---:|---:|---|
| D1 sinif yetersizligi | +0.0050 | +0.0240 | -0.0015 | -0.0166 | **desteklenmedi** |
| D2a lokalizasyon gurultusu | -0.0323 | **-0.0544** | -0.0321 | -0.0468 | desteklendi |
| D2b eksik etiket | -0.0130 | -0.0157 | **-0.1087** | +0.0237 | desteklendi |
| D3 sinif karisikligi (nadir) | -0.0281 | -0.0215 | **-0.2047** | -0.0347 | guclu destek |
| D3b sinif karisikligi (bol) | -0.0262 | +0.0042 | +0.0097 | -0.0495 | **bozulma soguruldu** |
| D4 kucuk nesne sinyal kaybi | -0.0224 | -0.0006 | -0.0642 | -0.0349 | **yalnizca boyut kiriliminda** |
| D5 kaynak kaymasi (`best.pt`) | -0.0107 | +0.0044 | -0.0363 | -0.0485 | best.pt'de gorunmez |
| D5 kaynak kaymasi (`last.pt`) | **-0.5848** | -0.4799 | -0.5308 | -0.5776 | **kaynak kiriliminda net** |

> **Bu tablo tek basina okunmamalidir.** D4'te toplam degerler mutevazidir,
> ancak boyut kirilimi 16 px altindaki bandin 45 puan coktugunu gosterir
> (z=-21,87); diger uc bantta degisim yoktur. D5'te `best.pt` neredeyse temiz
> gorunur, ancak `last.pt` kaynak kiriliminda catastrofik bir alan kaymasi
> ortaya cikar. D3b'de ise metriklerdeki durgunluk **gercek**: bozulma
> gercekten sogurulmustur (capraz hata 2 -> 4 kutu). Uc senaryonun dogru
> yorumu da yalnizca ilgili kirilim bolumu okunarak yapilabilir.

**Sinif bazli recall farki (v00'a gore, * = p<0.05 iki oran testi):**

| Senaryo | tasit (n=1264) | insan (n=2718) | UAP (n=15) | UAI (n=17) |
|---|---:|---:|---:|---:|
| D1 | +0.0071 | -0.0147 | 0.0000 | -0.0588 |
| D2a | -0.0111 | +0.0364 * | 0.0000 | -0.2127 |
| D2b | -0.0111 | +0.1016 * | 0.0000 | +0.0043 |
| D3 | +0.0823 * | +0.0728 * | 0.0000 | **-0.2941 *** |

**Sinif bazli precision farki (v00'a gore):**

| Senaryo | tasit | insan | UAP | UAI |
|---|---:|---:|---:|---:|
| D1 | +0.0017 | +0.0091 | -0.0090 | -0.0078 |
| D2a | -0.0650 | -0.0401 | -0.0335 | +0.0101 |
| D2b | -0.1184 | -0.1663 | -0.1017 | -0.0485 |
| D3 | -0.1035 | -0.1349 | **-0.4887** | -0.0916 |

#### Her senaryonun ayirt edici imzasi

Uc senaryo birbirinden **farkli metrik imzalari** uretiyor; ajanin teshis
gorevini anlamli kilan sey budur:

- **D2a (lokalizasyon gurultusu):** en cok mAP50-95 duser (-0.0544), mAP50'den
  (-0.0323) belirgin sekilde daha fazla. Kutu konumu bozuldugu icin siki IoU
  esikleri daha cok cezalandirir. Precision tum siniflarda orta duzeyde duser.
- **D2b (eksik etiket):** precision cokerken (-0.1087) recall **artar**
  (+0.0237); insan recall'i +0.1016 ile carpici sekilde yukselir. Model,
  egitimde "burada nesne yok" diye ogrendigi yerlerde fazladan kutu uretmeye
  itilmis; yani daha cok tahmin ediyor, daha cok yaniliyor.
- **D3 (sinif karisikligi):** precision en sert duser (-0.2047) ve dusus
  **karistirilan sinifta yogunlasir**: UAP precision -0.4887. Kutu yerleri
  dogru, sinif etiketleri yanlis. UAI recall -0.2941 (n=17 olmasina ragmen
  p<0.05).

Bu ayrisma, "bir LLM metrik imzasindan bozulmanin nedenini teshis edebilir mi"
sorusunun olumlu cevaplanabilmesi icin gereken temeldir.

#### D1 hipotezi neden desteklenmedi

D1 tek basarisiz senaryodur ve bunun nedeni ogreticidir. Hicbir sinif recall
farki istatistiksel anlamliliga ulasmiyor (insan recall -0.0147, z=-1.22).

Sebep, senaryonun kendisinde degil **deney kurgusunda**: tum senaryolar
`main_model.pt`'den fine-tune ile basliyor, ancak bu model zaten **ayni
dataset'in tamami uzerinde** egitilmis durumda. Insan iceren karelerin %90'i
egitimden cikarilsa bile, model insanlari zaten biliyor ve 11 epoch'luk kisa
bir fine-tune bunu unutturmaya yetmiyor.

Bu, iki bozulma turu arasindaki farki ortaya cikariyor:

- **Aktif bozulmalar** (D2a kaydirilmis kutu, D2b eksik etiket, D3 yanlis
  sinif) modele *yanlis bilgi ogretir* ve mevcut bilgisiyle celisir; bu yuzden
  az epoch'ta bile guclu etki uretirler.
- **Pasif bozulma** (D1 daha az ornek) yalnizca sinyali azaltir, yanlis bir
  sey ogretmez. Zaten egitilmis bir model bunu kisa fine-tune'da fark etmez.

Ek bir kanit: v00, D1 ve D2a kosularinda `best.pt` **epoch 1'den** secildi
(model hicbir epoch'ta baslangictan daha iyi olamadi). D2b, D2b final_best ve
D3'te ise sirasiyla epoch 3, 11 ve 14 secildi. Yani D1'in secilen agirliklari
bozulmaya neredeyse hic maruz kalmamis. Esit maruziyeti test etmek icin her
iki kosunun `last.pt`'si (epoch 11) de olculdu; sonuc degismedi:

| karsilastirma | insan AP50 | insan recall |
|---|---:|---:|
| D1 - v00, `best.pt` (epoch 1) | +0.0260 | -0.0147 (z=-1.22) |
| D1 - v00, `last.pt` (epoch 11) | -0.0541 | -0.0150 (z=-1.27) |

**D1'i gecerli hale getirmek icin oneri:** senaryo, dataset'in tamamini gormus
`main_model.pt`'den degil, genel amacli bir baslangic modelinden (repoda
mevcut `yolo26n.pt`) egitilmelidir. O zaman "sinif yetersizligi" modelin
ogrenmesini gercekten kisitlar. Bu, v00'in da ayni baslangictan yeniden
kosulmasini gerektirir.


## Proje Takip Plani

Bu bolum yapilacaklari adim adim takip etmek icindir. Tamamlanan madde [x]
olarak isaretlenir ve kanit dosyasi ayni commit'e eklenir.

### A. Temel Kurulum

- [x] Proje kok klasoru ve Python paket yapisi.
- [x] Lokal/Kaggle yollarinin config.yaml icinde ayrilmasi.
- [x] Git repository ve private GitHub remote kurulumu.
- [x] Model, dataset ve ciktilarin .gitignore ile dislanmasi.
- [x] Ilk Git commit: 41efd5b.

### B. Veri Sagligi

- [x] Dataset goruntu, etiket ve bbox sayilari.
- [x] Sinif ID dagilimi.
- [x] Yetim goruntu/etiket kontrolu.
- [x] Gecersiz YOLO satiri kontrolu.
- [x] reports/veri_raporu.json.
- [x] Kilitli val_diagnostic.
- [ ] Kaynak grup ve background sizintisi raporu.
- [ ] Bootstrap icin goruntu-birimli manifest.

### C. Model Baseline

- [x] final_best.pt eski referans olcumu.
- [x] main_model.pt aday olcumu.
- [x] 640 ve 768 inference ayrimi.
- [x] reports/model_secimi baseline raporu.
- [ ] Baseline hata galerisi.
- [ ] Baseline sinif guven araliklari.

### D. D1 Sinif Yetersizligi

- [x] Config ve manifest-only veri surumu.
- [x] Hedef class: insan, ID 1.
- [x] Insan frame: 9.289 -> 929.
- [x] Train frame: 17.515 -> 9.155.
- [x] Kaynak dataset degistirilmedi.
- [x] Kosu: experiments/run_20260825_223435_D1_42 (ortak protokolle yeniden
  kosuldu; ilk kosu run_20260817_222323_D1_42 tarihsel kayit).
- [x] Egitim tamamen bitti.
- [x] weights/best.pt bulundu.
- [x] D1 diagnostic val degerlendirmesi.
- [x] D1-baseline fark raporu.
- [ ] D1 hata galerisi.

D1 sonucu (2026-08-25 yeniden kosusu, ortak protokol):

> **UYARI — asagidaki tablo fine-tune-edilmemis main_model'e goredir ve
> bozulma etkisini oldugundan buyuk gosterir.** v00 saglikli referansi
> uretildikten sonra goruldu ki buradaki "insan recall -0.0912"nin
> -0.0765'i temiz fine-tune'un kendi etkisi; D1'e ozgu kisim yalnizca
> **-0.0147** ve istatistiksel olarak anlamli degil (z=-1.22).
> **D1 hipotezi desteklenmemistir.** Gerekce ve dogru sayilar icin
> "v00 Saglikli Referans ve Senaryo Karsilastirmasi" bolumune bakin.
> Asagidaki tablo tarihsel kayit olarak korunuyor.

Bu tablo, `senaryolar/egitim_protokolu.yaml` ortak protokoluyle (lr0=0.001,
warmup_epochs=3) yapilan yeniden kosuya aittir. Ilk D1 kosusu farkli
optimizasyon ayariyla (lr0=0.0005, warmup_epochs=2) egitilmisti; bu yuzden
D2a/D2b/D3 ile karsilastirilabilir degildi. Yeniden kosu 11 epoch surdu
(88 dk), ilk kosu da 11 epoch/89,5 dk surmustu.

| Olcum | main_model baseline | D1 (ortak protokol) | Fark |
|---|---:|---:|---:|
| mAP50 | 0.9331 | 0.9250 | -0.0082 |
| mAP50-95 | 0.6988 | 0.6947 | -0.0041 |
| Precision | 0.8975 | 0.9160 | +0.0186 |
| Recall | 0.8786 | 0.8619 | -0.0167 |
| Insan AP50 | 0.8499 | 0.8330 | -0.0170 |
| **Insan recall** | **0.8157** | **0.7244** | **-0.0912** |

Sinif bazli recall (bbox n ile birlikte, rule 8):

| Sinif | baseline recall | D1 recall | Fark | bbox n |
|---|---:|---:|---:|---:|
| tasit | 0.8703 | 0.8410 | -0.0293 | 1.264 |
| insan | 0.8157 | 0.7244 | -0.0912 | 2.718 |
| UAP | 1.0000 | 1.0000 | +0.0000 | 15 |
| UAI | 0.8286 | 0.8824 | +0.0538 | 17 |

**D1 hipotezi desteklendi ve etkisi artik izole edilmis durumda.** Insan
egitim karelerinin yuzde 90'i cikarilinca insan recall'i 9,1 puan dusuyor;
yani model belirgin sekilde daha fazla insan kaciriyor. Buna karsilik
mAP50 yalnizca 0,8 puan, mAP50-95 ise 0,4 puan dusuyor ve precision
baseline'in uzerine cikiyor.

Bu ayrisma, ilk kosunun verdigi resimden onemli olcude farklidir. Ilk kosu
mAP50'de -0,0270 ve mAP50-95'te -0,0286 gosteriyordu; yani bozulma genel bir
performans kaybi gibi gorunuyordu. **O genel kaybin buyuk kismi veri
bozulmasindan degil, dusuk ogrenme oranindan kaynaklaniyormus.** Ortak
protokolle kosuldugunda geriye kalan sey, tam olarak beklenen imza oluyor:
hedeflenen sinifin recall'inde belirgin dusus, genel tespit kalitesinde ise
neredeyse degisim yok. Insan recall dususu iki kosuda da benzer (-0,0957 ve
-0,0912), yani asil bulgu protokol degisikligine dayanikli.

Tasit recall'indeki -0,0293'luk dusus yan etkidir: insan iceren 8.360 kare
egitimden cikarilirken o karelerdeki tasitlar da cikmistir. UAP/UAI
sonuclari 15 ve 17 bbox'a dayandigi icin bu sinifler hakkinda guclu
genelleme iddiasi kurulamaz; UAI'deki +0,0538 tek bir kutunun yakalanmasina
karsilik gelir.

Son model: experiments/run_20260825_223435_D1_42/weights/best.pt
Son rapor: reports/senaryo_D1/d1_metrics.json

Ilk kosu (protokol sapmali, tarihsel kayit olarak korunuyor):
experiments/run_20260817_222323_D1_42 · reports/eski_D1_protokol_sapmali/d1_metrics.json

### E. Senaryolar

- [x] D2a lokalizasyon gurultusu.
- [x] D2b eksik etiket.
- [x] D3 UAP/UAI class 2-3 karisikligi.
- [x] D4 kucuk nesne sinyal kaybi.
- [x] D5 kaynak/alani kaymasi.
- [x] D6a split sizintisi.
- [ ] D6b tekrar agirligi ve efektif n.
- [ ] E1 overfitting.
- [ ] E2 underfitting.
- [ ] E3 seed bazli kararsizlik.
- [ ] E4 640/768 uyumsuzlugu.

### D2a Sonucu

- [x] Kaggle kosusu: experiments/run_D2a_42.
- [x] Model: experiments/run_D2a_42/weights/best.pt.
- [x] Diagnostic degerlendirme: reports/senaryo_D2a/d2a_metrics.json.
- [x] Hata galerisi: reports/hata_galerisi_D2a/index.html.
- [x] Test seti kullanilmadi.

| Olcum | main_model baseline | D2a | Fark |
|---|---:|---:|---:|
| mAP50 | 0.9331 | 0.8877 | -0.0454 |
| mAP50-95 | 0.6988 | 0.6162 | -0.0826 |
| Precision | 0.8975 | 0.8854 | -0.0121 |
| Recall | 0.8786 | 0.8317 | -0.0469 |

D2a hipotezi desteklendi: bbox merkezleri kaydirilinca tam ortusmeyi olcen
mAP50-95, gevsek ortusmeyi olcen mAP50'den daha fazla dustu. Bu, lokalizasyon
etiket gurultusunun tespit performansini bozduguna dair orta guclukte kanittir.
UAP/UAI bbox sayilari 15 ve 17 oldugu icin bu sinifler icin guclu genelleme
iddiasi kurulmaz.

D2a hata galerisi 1.056 diagnostic goruntuyu taradi ve en yuksek hata skoruna
sahip 50 goruntuyu kaydetti. Yesil kutular gercek etiketleri, kirmizi kutular
model tahminlerini gosterir. Galeri su dosyalardan incelenir:

- reports/hata_galerisi_D2a/index.html
- reports/hata_galerisi_D2a/gallery.json
- reports/hata_galerisi_D2a/images/

### D2b Ilk Sonucu

- [x] Yerel GPU kosusu: experiments/run_D2b_42_local.
- [x] Model: experiments/run_D2b_42_local/weights/best.pt.
- [x] Diagnostic degerlendirme: reports/senaryo_D2b/d1_metrics.json.
- [x] Orijinal dataset degistirilmedi; train etiket satirlarinin yuzde 25'i
  kontrollu olarak eksiltildi.
- [x] Test seti kullanilmadi.

| Olcum | main_model baseline | D2b | Fark |
|---|---:|---:|---:|
| mAP50 | 0.9331 | 0.9070 | -0.0261 |
| mAP50-95 | 0.6988 | 0.6550 | -0.0438 |
| Precision | 0.8975 | 0.8088 | -0.0887 |
| Recall | 0.8786 | 0.9023 | +0.0236 |

D2b ilk sonucu, eksik etiketlerin precision'i belirgin dusurdugunu ve modelin
daha fazla yanlis pozitif urettigini gosteriyor. Recall'in artmasi, eksik
etiketli train verisinin modeli daha genis tahmin yapmaya itmis olabilecegiyle
uyumludur.

- reports/senaryo_D2b/d1_metrics.json
- reports/senaryo_D2b/d2b_val_diagnostic/confusion_matrix.png

### D2b Final_best Karsilastirmasi

- [x] Ayni D2b bozuk veri protokolu final_best.pt ile tekrarlandi.
- [x] Model: experiments/run_D2b_42_final_best_local/weights/best.pt.
- [x] Diagnostic degerlendirme: reports/senaryo_D2b_final_best/d1_metrics.json.

| Olcum | main_model D2b | final_best D2b | Fark |
|---|---:|---:|---:|
| mAP50 | 0.9070 | 0.8992 | -0.0077 |
| mAP50-95 | 0.6550 | 0.6491 | -0.0059 |
| Precision | 0.8088 | 0.8324 | +0.0236 |
| Recall | 0.9023 | 0.8724 | -0.0299 |

Bu iki kosu, ayni eksik etiket senaryosunda baslangic modelinin sonucu
etkiledigini gosteriyor. final_best precision'da daha iyi, ancak recall ve
toplam mAP'te main_model D2b kosusunun gerisinde. UAP/UAI siniflarinda bbox
sayisi 15 ve 17 oldugu icin bu sinifler hakkinda guclu genelleme iddiasi
kurulmamalidir.

- reports/senaryo_D2b_final_best/d1_metrics.json
- reports/senaryo_D2b_final_best/d2b_final_best_val_diagnostic/confusion_matrix.png

### D3 Sonucu

- [x] Yerel GPU kosusu: experiments/run_20260826_001456_D3_42 (24 epoch,
  5,6 saat, patience=10 ile erken durdu).
- [x] Model: experiments/run_20260826_001456_D3_42/weights/best.pt.
- [x] Diagnostic degerlendirme: reports/senaryo_D3/d1_metrics.json.
- [x] Orijinal dataset degistirilmedi; train etiketlerinde UAP (2) ve UAI (3)
  satirlarinin %30'u (117/391 satir; 68 UAP->UAI, 49 UAI->UAP) kontrollu
  olarak yer degistirildi (seed=42).
- [x] Egitim val'i operasyonel `dataset/images/val`; kilitli tanı seti
  egitimde kullanilmadi.
- [x] Test seti kullanilmadi.

| Olcum | main_model baseline | D3 | Fark |
|---|---:|---:|---:|
| mAP50 | 0.9331 | 0.8919 | -0.0413 |
| mAP50-95 | 0.6988 | 0.6492 | -0.0497 |
| **Precision** | **0.8975** | **0.7129** | **-0.1846** |
| Recall | 0.8786 | 0.8438 | -0.0348 |
| UAP AP50 | 0.9950 | 0.9950 | 0.0000 |
| UAI AP50 | 0.9950 | 0.8591 | -0.1359 |

Sinif bazli precision/recall (bbox n ile birlikte, rule 8):

| Sinif | baseline P | D3 P | baseline R | D3 R | bbox n |
|---|---:|---:|---:|---:|---:|
| tasit | 0.8203 | 0.7538 | 0.8703 | 0.9161 | 1.264 |
| insan | 0.8279 | 0.7265 | 0.8157 | 0.8120 | 2.718 |
| **UAP** | **0.9417** | **0.4730** | 1.0000 | 1.0000 | 15 |
| UAI | 1.0000 | 0.8983 | 0.8286 | 0.6471 | 17 |

Kilitli tanı setindeki confusion matrix (sabit esik, argmax sinif):

| Gercek sinif (n) | UAP tahmin | UAI tahmin | background |
|---|---:|---:|---:|
| UAP (15) | 15 (%100) | 0 | 0 |
| UAI (17) | **11 (%65)** | 2 (%12) | 4 (%24) |

Bu tablo Ultralytics gorselinden okunmustur. Bagimsiz olcum
(`teshis/degerlendirme/metrikler.py`, sinif-bagimsiz eslestirme) ayni yonu
biraz farkli buyuklukle dogruluyor:

| Gercek sinif (n) | | UAP tahmin | UAI tahmin | bulunamadi |
|---|---|---:|---:|---:|
| UAP (15) | v00 | 0 | 15 | 0 |
| UAP (15) | D3 | 0 | 15 | 0 |
| UAI (17) | v00 | 0 | **17** | 0 |
| UAI (17) | D3 | **8** | **8** | 1 |

Iki olcum de ayni sonuca varir: UAI sinifi cokmustur (17/17 dogru -> 8/17).
D3b'de ayni capraz kontrol yapildiginda gorsel ile bagimsiz olcum
**celiskili** cikti; ayrinti icin "D3b Sonucu" bolumundeki "Olcum notu"na
bakin. Bu nedenle projedeki karisiklik iddialari artik bagimsiz olcume
dayandirilmaktadir.

**D3 hipotezi guclu bicimde desteklendi ve etkinin yonu net.** Train
etiketlerinin %30'unun UAP<->UAI arasinda karistirilmasi, modelde tek yonlu
bir kayma yaratti: gercek UAI kutularinin %65'i UAP olarak tahmin ediliyor,
buna karsilik gercek UAP kutularinin tamami dogru siniflandiriliyor. Yani
UAP bir "cekici sinif" haline gelmis.

Bu kayma iki metrikte birden gorulur ve birbirini dogrular:

- **UAP precision 0.9417 -> 0.4730.** Model UAP dedigi kutularin yarisindan
  fazlasinda yaniliyor; cunku UAI'leri de UAP diye isaretliyor.
- **UAI recall 0.8286 -> 0.6471.** Gercek UAI'lerin ucte biri kaciriliyor,
  cunku UAP'a kaydiriliyorlar.

Genel precision'in 18,5 puan dusmesi (0.8975 -> 0.7129) bu senaryonun en
belirgin imzasidir ve D2b'nin (eksik etiket) precision dususunden farkli bir
mekanizmadan gelir: D2b'de model fazladan kutu uretir, D3'te ise urettigi
kutulara yanlis sinif etiketi yapistirir.

UAP/UAI bbox sayisi 15 ve 17 oldugu icin bu oranlar birkac ornekle
degisebilir; guclu istatistiksel genelleme iddiasi kurulmaz. Ancak etki
buyuklugu (11/17 yanlis siniflandirma, precision'da 47 puanlik dusus) rastgele
dalgalanmayla aciklanamayacak kadar buyuktur.

- reports/senaryo_D3/d1_metrics.json
- reports/senaryo_D3/gorseller/confusion_matrix.png
- veri_surumleri/v04_d3_uap_uai_sinif_karisikligi/manifest.json

Ilk kosu (egitim val'i olarak kilitli tanı setini kullaniyordu, bu yuzden
iyimser yanliydi; tarihsel kayit olarak korunuyor):
experiments/run_D3_42_local · reports/eski_D3_val_sizintili/d1_metrics.json
- veri_surumleri/v04_d3_uap_uai_sinif_karisikligi/manifest.json

### D1 Yeniden Kurgu (yolo26n cifti) — hipotez, dogru kurguda guclu bicimde dogrulandi

D1 mevcut kurguda basarisiz olmustu: insan recall farki -0.0147, z=-1,22,
anlamli degil. Teshisimiz suydu: tum senaryolar `main_model.pt`'den fine-tune
ile basliyor, ancak bu model **zaten ayni dataset'in tamami** uzerinde
egitilmis. Insan karelerinin %90'i cikarilsa bile model insanlari biliyor ve
kisa bir fine-tune bunu unutturmuyor.

Bu teshisi sinamak icin ayni bozulma, dataset'i hic gormemis genel amacli bir
baslangic modelinden (`yolo26n.pt`) yeniden kosuldu.

**Kosu esitligi.** Iki kosu da: `yolo26n.pt` baslangici, ortak protokol
(lr0=0.001, warmup=3), 30-epoch cos_lr programi, seed=42, batch=8, imgsz=768
ve **23 tamamlanmis epoch**. Esit epoch sayisi kasitlidir: v00n bir kesinti
nedeniyle 23 temiz epoch'ta kaldi, bu yuzden D1n de 23. epoch'ta durduruldu.

**Sonuc — hipotez guclu bicimde dogrulandi:**

| Sinif | bbox n | v00n recall | D1n recall | fark | z |
|---|---:|---:|---:|---:|---:|
| **insan** | 2.718 | 0.7171 | 0.4696 | **-0.2475** | **-18,58 *** |
| tasit | 1.264 | 0.7626 | 0.8046 | +0.0420 | +2,56 * |
| UAP | 15 | 1.0000 | 0.8667 | -0.1333 | -1,46 |
| UAI | 17 | 0.2941 | 0.6471 | +0.3529 | +2,06 * |

Sinif AP50: insan 0.8202 → 0.6481 (**-0.1721**), tasit -0.0161, UAP -0.0135,
UAI -0.0937. Genel: mAP50 -0.0739, precision -0.1184.

**Iki kurgunun karsilastirmasi — bulgunun ozu:**

| | main_model kurgusu | yolo26n kurgusu |
|---|---:|---:|
| insan recall farki | -0.0147 | **-0.2475** |
| z | -1,22 (anlamsiz) | **-18,58** |
| insan AP50 farki | +0.0260 | **-0.1721** |

Ayni veri bozulmasi, ayni protokol, ayni olcum seti. Tek fark baslangic
modeli. **D1'in ilk basarisizligi bozulmanin etkisiz olmasindan degil, deney
kurgusunun onu olcememesinden kaynaklaniyormus.** Zaten veriyi ogrenmis bir
modeli kisa fine-tune ile "unutturamazsiniz"; sinif yetersizliginin etkisini
gormek icin modelin o veriyi ilk kez ogreniyor olmasi gerekir.

`tasit` recall'inin yukselmesi (+0.0420) beklenen yan etkidir: insan kareleri
azalinca tasit egitimde goreli olarak agirlik kazanir. UAI'deki +0.3529 ise
n=17 uzerinde olculdugu ve v00n'in UAI taban degeri zaten cok dusuk oldugu
(0.2941) icin guvenilir bir bulgu sayilmaz.

> **Kapsam siniri — bu cift kendi icinde okunmalidir.** `yolo26n.pt` 2,57M
> parametreli nano bir modeldir; `main_model.pt` cok daha buyuktur. Ayrica bu
> kosular 23 epoch'ta kalmistir. Bu yuzden buradaki mutlak sayilar
> (v00n mAP50 0.8126) main_model serisiyle **karsilastirilamaz**; yalnizca
> ciftin kendi icindeki fark anlamlidir. Ayni nedenle bu iki kosu ajana
> senaryo olarak sunulmaz (`araclar.AJANA_VERILMEYEN`).

> **Provenans notu:** v00n'in `args.yaml` dosyasi, basarisiz devam denemesi
> sirasinda uzerine yazildigi icin `model: last.pt` gosterir. Kosunun gercek
> baslangic modeli `run_manifest.json` icinde `yolo26n.pt` olarak kayitlidir.

Istatistikler `teshis/degerlendirme/bootstrap.py` ile yeniden uretilebilir:
`python -m teshis.degerlendirme.bootstrap --kosu reports/yolo26n_senaryo_D1n/d1_metrics.json --referans reports/yolo26n_referans_v00n/d1_metrics.json --alan class_recall`

- reports/yolo26n_referans_v00n/d1_metrics.json · reports/yolo26n_senaryo_D1n/d1_metrics.json

### D6b Sonucu — temsil payi ile performans arasinda net doz-yanit iliskisi

D6b, 175 train karesini **40 kez** tekrarlayarak o sahneleri asiri temsil
eder. Egitim listesi 17.515 -> 24.340 satira cikar ve bu 175 kare egitimin
**%28,8'ini** kaplar. D1/D5 gibi manifest-only: etiketler degismez, goruntu
kopyalanmaz. Kosu: 11 epoch, 3,15 saat.

Bozulmanin kritik ozelligi, tekrarlanan karelerin **sinif bilesimidir**:

| Sinif | Tekrarlanan karelerdeki bbox payi |
|---|---:|
| tasit | %22,80 |
| insan | %10,39 |
| UAP | %0,06 |
| **UAI** | **%0,00** |

Yani asiri temsil edilen sahnelerde UAI hic bulunmuyor, UAP neredeyse yok.
Hipotez: bu siniflarin goreli agirligi dustugu icin performanslari bozulmali.

**`best.pt` (epoch 1) hipotezi gostermez.** mAP50 +0.0033, insan recall
+0.0394. Bu, D1/D2a/D4/D5'te de gorulen desendir: model hicbir epoch'ta
baslangictan iyi olamadigi icin en iyi checkpoint epoch 1'den secilir ve
bozulmaya neredeyse hic maruz kalmaz.

**`last.pt` (epoch 11) hipotezi net bicimde dogrular** ve temsil payi ile
performans arasinda **monotonik** bir iliski cikar:

| Sinif | bbox n | egitimdeki asiri temsil | AP50 farki | recall farki | z |
|---|---:|---:|---:|---:|---:|
| tasit | 1.264 | %22,80 | +0.0119 | +0.0356 | +2,52 * |
| insan | 2.718 | %10,39 | +0.0041 | +0.0224 | +1,91 |
| UAP | 15 | %0,06 | 0.0000 | 0.0000 | 0,00 |
| **UAI** | **17** | **%0,00** | **-0.1482** | **-0.6123** | **-3,59 *** |

Siralama tam olarak temsil payini izliyor: en cok temsil edilen sinif
(tasit) kazaniyor, ikinci sinif (insan) daha az kazaniyor, hic temsil
edilmeyen sinif (UAI) **cokuyor** — recall 0.9412'den 0.3289'a iniyor.

> **Istatistik notu:** bu tablodaki z degerleri
> `teshis/degerlendirme/bootstrap.py` ile yeniden uretilebilir:
> `python -m teshis.degerlendirme.bootstrap --kosu reports/senaryo_D6b_last_pt/d1_metrics.json
> --referans reports/referans_v00/d1_metrics.json --alan class_recall`
> UAI icin z, ilk raporda -3,74 yazilmisti; iki oran testi sayimlar uzerinde
> tanimli oldugu icin modul geri turetilen tam sayilardan hesaplar ve -3,59
> verir. Daha tutarli olan ikincisidir; sonuc degismez (p<0,05).
Etki n=17 olmasina ragmen istatistiksel olarak anlamli (z=-3,59), cunku
buyuklugu cok fazla.

UAP'nin degismemesi aciklanabilir: v00'da zaten recall 1.0000 ile tavanda ve
tekrarlanan karelerdeki payi sifir degil (%0,06), yani tamamen yok olmamis.

**Bulgunun anlami.** Tekrar agirligi, veriye hicbir "hata" eklemez: hicbir
etiket yanlis degildir, hicbir kutu silinmemistir, hicbir goruntu
bozulmamistir. Yalnizca bazi kareler daha sik gosterilir. Buna ragmen, o
karelerde bulunmayan bir sinif neredeyse tamamen kaybolur. Bu, veri
kalitesinin yalnizca "etiketler dogru mu" sorusu olmadigini; **dagilimin
kendisinin** bir kalite boyutu oldugunu gosterir.

Pratik karsiligi yaygin: ayni sahnenin cok kez cekilmesi, augment
kopyalarinin veri setine birden fazla girmesi veya "zor ornekleri cogalt"
turu bir dengeleme calismasinin yan etkisi. Hicbiri hata gibi gorunmez.

> **Sinir:** UAI icin n=17 oldugundan bu etkinin **buyuklugu** hakkinda kesin
> genelleme yapilamaz; %95 guven araligi genistir. Ancak yon ve anlamlilik
> (z=-3,59) rastgele dalgalanmayla aciklanamaz. D3/D3b ciftinde oldugu gibi,
> nadir siniflarda yon guvenilir, buyukluk degildir.

- reports/senaryo_D6b/d1_metrics.json · reports/senaryo_D6b_last_pt/d1_metrics.json
- veri_surumleri/v09_d6b_tekrar_agirligi/manifest.json

### D6a Sonucu — sizintili val, gercek bozulmalardan daha buyuk bir yalan soyluyor

D6a, veri setinin **degerlendirme tarafindaki** bir kusuru gosterir: train
kareleri val icine de sizmissa, val skoru yapay olarak yukselir ve gelistirme
boyunca yaniltir.

**Bu senaryo yeniden egitim gerektirmez.** Sizinti modeli degil olcumu bozar;
bu yuzden zaten egitilmis saglikli referans (v00) modeli iki farkli kumede
olculur. Kilitli tanı setine 264 train karesi eklenerek %20 sizintili bir
degerlendirme kumesi uretildi (1.056 temiz + 264 sizan = 1.320 kare).

**Ayni model, iki farkli kume:**

| Olcum | temiz tanı seti | sizintili kume | sisme | goreli |
|---|---:|---:|---:|---:|
| mAP50 | 0.9200 | 0.9285 | +0.0085 | +0,9% |
| **mAP50-95** | **0.6707** | **0.6994** | **+0.0287** | **+4,3%** |
| Precision | 0.9175 | 0.9210 | +0.0035 | +0,4% |
| Recall | 0.8785 | 0.8846 | +0.0060 | +0,7% |

Sinif bazinda: tasit AP50 +0.0256, insan AP50 +0.0146.

**En cok mAP50-95 sisiyor (+4,3%), mAP50'nin (+0,9%) bes kati.** Mekanizma
acik: model sizan kareleri egitimde gormus ve kutu konumlarini ezberlemistir;
siki IoU esikleri (0.5-0.95) tam da konum hassasiyetini olctugu icin
ezberlemeden en cok onlar yararlanir. Yani sizinti, **en guvendigimiz metrigi
en cok bozuyor.**

#### Asil tehlike: sizinti, gercek bozulmalari maskeleyebilir

Sizintinin urettigi sisme (+0.0287 mAP50-95), projedeki bircok **gercek
bozulmanin** olculen etkisinden daha buyuk:

| Kaynak | mAP50-95 degisimi |
|---|---:|
| **D6a sizinti sismesi** | **+0.0287** |
| D1 sinif yetersizligi | +0.0240 |
| D5 kaynak kaymasi (`best.pt`) | +0.0044 |
| D3b sinif karisikligi | +0.0042 |
| D4 kucuk nesne kaybi | -0.0006 |

Yalnizca %20 sizinti, D4'un ve D3b'nin gercek etkisini tamamen ortecek
buyuklukte bir sisme uretiyor. Sizintili bir val setiyle calisan ekip, D4
gibi bir bozulmayi olcemez: bozulmanin dusurdugu skoru sizintinin sismesi
geri kaldirir ve metrik "normal" gorunur.

Bu, D5'in dersinin tamamlayicisidir. D5 "val setinizin **kaynak cesitliligi**
onemlidir" diyordu; D6a "val setinizin **temizligi** onemlidir" diyor. Ikisi
birlikte, projenin kilitli ve degistirilmeyen bir tanı seti tutma kuralinin
neden metodolojinin merkezinde oldugunu gosterir.

**Kural 3 ihlal edilmedi:** temiz karsilastirma tabani olarak kilitli tanı
seti kullanildi, test setine hic dokunulmadi. Senaryo config'indeki
"test-val farki" ifadesi ayni olguyu anlatir; burada temiz taban olarak tanı
seti alinmistir.

- reports/senaryo_D6a/d1_metrics.json
- veri_surumleri/v08_d6a_split_sizintisi/manifest.json

### D5 Sonucu — erken durdurmanin gizledigi felaket

D5, egitim setini yalnizca **tek bir kaynakla** (`aaterm`) sinirlar:
17.515 kareden 11.064'u kalir, 6.451'i cikarilir (hituav 2.292, termal 3.785,
sentetik 273, tf2026 101). Etiketler hic degistirilmez; bu bir **veri secimi**
bozulmasidir. val ve test kasitli olarak tum kaynaklari icermeye devam eder.
Kosu: 11 epoch (erken durdu), 1,51 saat.

Bu senaryo, projedeki tek senaryodur ki **`best.pt` ile `last.pt` tamamen
farkli hikayeler anlatir** ve ikisi de anlamlidir.

**`best.pt` (epoch 1) — dagitilacak model:**

| Olcum | v00 | D5 best | fark |
|---|---:|---:|---:|
| mAP50 | 0.9200 | 0.9092 | -0.0107 |
| Precision | 0.9175 | 0.8813 | -0.0363 |
| Recall | 0.8785 | 0.8301 | -0.0485 |

Kaynak bazli bakildiginda hipotez **desteklenmiyor** gibi gorunur:

| Kaynak | Egitimde | n | v00 | best | fark | z |
|---|---|---:|---:|---:|---:|---:|
| aaterm | VAR | 885 | 0.9107 | 0.9164 | +0.0057 | +0,42 |
| hituav | yok | 2.165 | 0.8402 | 0.8162 | -0.0240 | -2,09 |
| termal | yok | 858 | 0.7145 | 0.8531 | **+0.1386** | **+6,98** |
| tf2026 | yok | 106 | 0.9906 | 0.9528 | -0.0378 | -1,66 |

Egitimde hic gorulmeyen `termal` kaynagi **belirgin sekilde yukseliyor**.
Sebep, epoch 1'de modelin henuz tek kaynaga uyum saglamamis olmasi; bunun
yerine calisma noktasi daha permissive tarafa kaymis ve bu, taban recall'i en
dusuk olan kaynaga (termal 0.7145) en cok yaramis.

**`last.pt` (epoch 11) — egitimin dogal sonu:**

| Olcum | v00 | D5 last | fark |
|---|---:|---:|---:|
| mAP50 | 0.9200 | **0.3352** | **-0.5848** |
| Precision | 0.9175 | 0.3868 | -0.5308 |
| Recall | 0.8785 | 0.3009 | -0.5776 |

Ve kaynak kirilimi hipotezi **tam olarak dogruluyor**:

| Kaynak | Egitimde | n | v00 | last | fark | z |
|---|---|---:|---:|---:|---:|---:|
| **aaterm** | **VAR** | 885 | 0.9107 | 0.8825 | **-0.0282** | -1,95 (anlamli degil) |
| hituav | yok | 2.165 | 0.8402 | 0.3977 | **-0.4425** | **-29,98** |
| tf2026 | yok | 106 | 0.9906 | 0.4245 | **-0.5661** | **-9,06** |
| termal | yok | 858 | 0.7145 | 0.6480 | -0.0665 | -2,95 |

Egitimde tutulan **tek kaynak korunuyor** (istatistiksel olarak anlamli kayip
yok), digerleri cokuyor. Bu, alan kaymasinin ders kitabi tanimidir.

#### Bu senaryonun pratik dersi

D5'in asil bulgusu su: **kaynak cesitliligi olan bir val setinde erken
durdurma, alan kaymasi felaketini onledi.** Model 11 epoch boyunca yalnizca
`aaterm` gordu ve o yone dogru cokerken, tum kaynaklari iceren val seti bunu
fark etti ve en iyi checkpoint olarak epoch 1'i secti.

Bunun tersi senaryo gercek hayatta cok daha yaygindir: bir ekip yalnizca kendi
topladigi kaynaktan veri toplar, val setini de **ayni kaynaktan** ayirir. O
durumda val, cokusu goremezdi (aaterm korunuyor!), egitim yakinsayana kadar
surer ve dagitilan model diger sensorlerde/sahnelerde catastrofik sekilde
basarisiz olurdu.

Bu yuzden D5, "veri toplama cesitliligi" kadar **"validasyon setinin kaynak
cesitliligi"** hakkinda da bir uyaridir. `results.csv`'ye `best.pt` kaydedildi
(dagitilacak model odur), ancak `last.pt` olcumu latent kirilganligi
belgeledigi icin burada birlikte raporlanir.

- reports/senaryo_D5/d1_metrics.json · reports/senaryo_D5_last_pt/d1_metrics.json
- reports/kirilim/D5.json · reports/kirilim/D5_last.json
- veri_surumleri/v07_d5_kaynak_alani_kaymasi/manifest.json

### D4 Sonucu — projenin en temiz kontrollu deneyi

D4, egitim etiketlerinden etkin piksel boyutu **16 px'in altinda** kalan
29.499 kutuyu siler (tum train kutularinin %22,4'u; %76'si `insan`).
Goruntuler ve diger kutular dokunulmaz, yani model bu nesneleri arka plan
olarak ogrenir. Kosu: 11 epoch (erken durdu), 2,34 saat.

Uretim asamasinda dogrulandi: **silinen kutularin tamami** `cok_kucuk_16_alti`
bandinda cikti. Bunun sebebi bozulma esiginin ve olcum bandinin ayni
fonksiyondan (`teshis/degerlendirme/metrikler.py::etkin_sqrt_alan`) turemesi;
boylece "egitimden cikarilan bant" ile "recall'i olculen bant" birebir
ortusuyor.

**Boyut bandi bazli recall (v00'a gore):**

| Boyut bandi | bbox n | v00 | D4 | fark | z |
|---|---:|---:|---:|---:|---:|
| **cok_kucuk (<16 px)** | 1.167 | 0.7446 | 0.2922 | **-0.4524** | **-21,87** |
| kucuk (16-32 px) | 2.011 | 0.8344 | 0.8364 | +0.0020 | +0,17 |
| orta (32-64 px) | 520 | 0.9308 | 0.9192 | -0.0116 | -0,71 |
| buyuk (>64 px) | 316 | 0.9873 | 0.9873 | 0.0000 | 0,00 |

**Bu, projedeki en temiz sonuctur.** Bozulan bant 45 puan cokuyor (z=-21,87)
ve diger uc bandin hicbirinde istatistiksel olarak anlamli degisim yok. Etki
tam olarak hedeflenen yere dusuyor, hicbir yere sizmiyor.

#### D4 ile D1 nasil ayirt edilir

D4'te silinen kutularin %76'si `insan` sinifindan; yani D4 de D1 gibi
agirlikla insan'i etkiliyor. Toplam sinif metrikleri ikisini ayirt etmeye
yetmez — ayrim **sinif x boyut** kiriliminda ortaya cikar:

| Kirilim | bbox n | v00 | D4 | fark |
|---|---:|---:|---:|---:|
| insan, <16 px | 1.129 | 0.7564 | 0.2914 | **-0.4650** |
| insan, 16-32 px | 1.574 | 0.8355 | 0.8437 | **+0.0082** |
| tasit, 16-32 px | 436 | 0.8303 | 0.8096 | -0.0207 |
| tasit, 32-64 px | 480 | 0.9333 | 0.9229 | -0.0104 |

D4 yalnizca **kucuk** insan kutularini kaciriyor; ayni sinifin buyuk
kutularinda hicbir kayip yok (hatta +0,008). Sinif yetersizligi olsaydi
(D1'in iddiasi) kayip tum boyutlara yayilirdi. Bu, ajanin iki senaryoyu
birbirinden ayirabilmesi icin gereken ayirt edici kanittir ve yalnizca
boyut-katmanli olcumle gorunur.

Karsilastirma icin insan recall'inin toplam degeri:

| Kosu | insan recall | v00'a gore fark | z |
|---|---:|---:|---:|
| v00 (referans) | 0.7391 | — | — |
| D1 | 0.7244 | -0.0147 | -1,22 (anlamli degil) |
| D4 `best.pt` | 0.5746 | **-0.1645** | **-12,78** |
| D4 `last.pt` | 0.4522 | **-0.2869** | **-21,55** |

D4'un `best.pt`'si de epoch 1'den secildi (v00/D1/D2a gibi), yani bu etki
bozulmaya yalnizca 1 epoch maruz kalmis agirliklarla bile olusuyor. `last.pt`
(epoch 11) ile etki neredeyse ikiye katlaniyor. Bu, D1 ile arasindaki farkin
maruziyet suresinden degil, bozulmanin **turunden** kaynaklandigini gosterir:
D4 modele "bu nesneler arka plandir" diye aktif yanlis bilgi ogretir; D1 ise
yalnizca ornek sayisini azaltir.

- reports/senaryo_D4/d1_metrics.json
- reports/kirilim/v00.json · reports/kirilim/D4.json
- veri_surumleri/v06_d4_kucuk_nesne_sinyal_kaybi/manifest.json

### D3b Sonucu — bol ornekli siniflar simetrik etiket gurultusunu soguruyor

> **DUZELTME (2026-08-26).** Bu bolumun ilk hali, Ultralytics'in urettigi
> `confusion_matrix.png` gorselinden okunan "gercek tasit kutularinin %28'i
> insan olarak tahmin edildi" bulgusuna dayaniyordu. Ajanin arac katmanina
> bagimsiz bir karisiklik olcumu eklendiginde bu rakam dogrulanamadi ve
> **hatali oldugu anlasildi**. Asagidaki bolum duzeltilmis olcumlere gore
> yeniden yazildi; yanlis bulgunun nasil ortaya cikip nasil yakalandigi
> "Olcum notu" basliginda aciklanmistir.

D3b, D3 ile **ayni bozulmayi** (ayni takas orani 0.30, ayni seed, ayni
protokol) bol ornekli `tasit` ve `insan` siniflarina uygular. Train'de 131.309
hedef satirin 39.393'u degistirildi (25.209 tasit->insan, 14.184 insan->tasit).
Kosu: 30 epoch, 6,12 saat, `best.pt` epoch 11'den secildi.

**Genel metrikler (v00'a gore):**

| Olcum | fark |
|---|---:|
| mAP50 | -0.0262 |
| mAP50-95 | +0.0042 |
| Precision | +0.0097 |
| Recall (genel) | -0.0495 |
| tasit recall | -0.0063 (z=-0.42, anlamli degil) |
| insan recall | +0.0099 (z=+0.84, anlamli degil) |

**Sinif karisikligi (bagimsiz olcum, `teshis/degerlendirme/metrikler.py`):**

| Gercek sinif | n | | dogru | capraz | bulunamadi |
|---|---:|---|---:|---:|---:|
| tasit | 1.264 | v00 | 1.131 | 2 | 131 |
| tasit | 1.264 | **D3b** | **1.081** | **4** | **179** |
| insan | 2.718 | v00 | 2.180 | 4 | 534 |
| insan | 2.718 | **D3b** | **2.140** | **4** | **574** |

**Bulgu: bozulma sogurulmus.** Egitim etiketlerinin %30'u karistirilmis
olmasina ragmen, tanı setinde capraz sinif hatasi neredeyse hic artmiyor
(tasit icin 2 -> 4 kutu, insan icin 4 -> 4). Model bunun yerine biraz daha
temkinli davraniyor: kacirilan kutu sayisi artiyor (131 -> 179 ve 534 -> 574).

Sebep, simetrik etiket gurultusunun bol veriyle **ortalamada sonumlenmesidir**.
`tasit` ve `insan` icin egitimde sirasiyla 83.600 ve 47.709 satir var; %30'u
bozulsa bile %70'i dogru kaliyor ve iki sinif termal goruntude belirgin sekilde
farkli oldugu icin model dogru esleme ogrenmeye devam ediyor.

**D3 ile karsilastirma — asil bulgu bu:**

| | D3 (UAP/UAI) | D3b (tasit/insan) |
|---|---|---|
| egitimdeki hedef satir | 391 | 131.309 |
| takas orani | %30 | %30 |
| tanı setinde bbox n | 15 / 17 | 1.264 / 2.718 |
| capraz hata (v00 -> senaryo) | UAI: 0/17 -> **8/17** | tasit: 2/1264 -> **4/1264** |
| sonuc | **sinif cokuyor** | **bozulma soguruluyor** |

Ayni bozulma orani, nadir siniflarda sinifi yok ederken bol siniflarda
neredeyse etkisiz kaliyor. Etiket gurultusune dayaniklilik, bozulmanin
oraniyla degil **sinifin ornek sayisiyla** belirleniyor. Bu, D3'un neden
UAP/UAI'de bu kadar sert bir etki urettigini de aciklar: 391 satirin 117'si
bozulunca modelin ogrenecek dogru sinyali kalmiyor.

#### Olcum notu — Ultralytics confusion matrix gorseline neden guvenilmedi

Bu senaryonun ilk raporu, `confusion_matrix.png` gorselinden okunan
"tasit: 755 dogru, 358 insan, 151 bulunamadi" degerlerine dayaniyordu.
Bagimsiz olcum bunu dogrulamadi ve **Ultralytics'in gorseli kendi
metrikleriyle celisiyor**:

| Kaynak | tasit dogru tespit |
|---|---:|
| Ultralytics'in raporladigi `tasit` recall (0.8275 x 1264) | ~1.046 |
| Ultralytics `confusion_matrix.png` | 755 |
| Bagimsiz olcum (`metrikler.py`) | 1.081 |

Gorsel, ayni kosunun raporladigi recall ile tutarli degil; bagimsiz olcum ise
tutarli. Fark, conf esigi veya IoU/guven sirali eslestirme denenerek
yeniden uretilmeye calisildi ancak hicbiri 358 rakamini vermedi (en yuksek 30).
Bu nedenle rapor, ic tutarliligi dogrulanabilen bagimsiz olcume dayandirildi.

Ders: **turetilmis bir gorsel, kendi urettigi sayisal metriklerle capraz
kontrol edilmeden manset bulgu yapilmamalidir.** Projedeki karisiklik
iddialarinin tamami artik `metrikler.py`'nin sinif-bagimsiz eslestirmesiyle
uretiliyor ve ayni fonksiyon ajanin `sinif_karisikligini_getir` aracini da
besliyor.

- reports/senaryo_D3b/d1_metrics.json
- reports/kirilim/d3b_20260826.json (bagimsiz karisiklik olcumu)
- veri_surumleri/v05_d3b_tasit_insan_karisikligi/manifest.json

### D3b Sonucu (ILK, HATALI SURUM — tarihsel kayit)

> Bu bolum, Ultralytics confusion matrix gorselinin yanlis okunmasina
> dayanan ilk raporun kaydidir. Icerigi **gecersizdir**; yukaridaki
> duzeltilmis D3b bolumu gecerlidir. Hatanin nasil yakalandigi orada
> "Olcum notu" basliginda anlatiliyor. Kayit, bulgunun nasil duzeltildigini
> izlenebilir kilmak icin silinmeden birakildi.

Ilk raporun manset iddiasi: "gercek tasit kutularinin %28'i (358/1264)
insan olarak tahmin ediliyor (z=20,3); AP tabanli metrikler bu felaketi
tamamen gizliyor." Bagimsiz olcum bu rakami dogrulamadi (gercek deger
4/1264) ve Ultralytics gorselinin kendi raporladigi recall ile celistigi
goruldu.

---

## E Serisi — Egitim Arizalari

D serisi **veriyi** bozar, egitim protokolunu sabit tutar. E serisi tersini
yapar: veri temizdir (`v00_saglikli`), bozulan **egitim/cikarim
protokoludur**. Bu yuzden E kosulari D kosulariyla ayni tabloda
karsilastirilamaz; her biri kendi saglikli referansiyla okunur.

Sapmalar `senaryolar/egitim_protokolu.yaml` icindeki `e_serisi` blokunda
**beyan edilir**, koda dagilmis CLI bayraklariyla degil. `kos.py --e-senaryo E2`
sapmalari oradan okur, kosu manifestine `protokol_sapmalari` alani olarak
yazar ve baslangicta ekrana basar. Boylece hangi kosunun protokolden nerede
ayrildigi hem YAML'dan hem kosu ciktisinin kendisinden okunabilir.

### E1 Sonucu — asiri uyum gerceklesti, ama checkpoint secimi onu TAMAMEN gizliyor

E1'in beklenen kaniti: *"train yuksek, val dusuk, train_val farki acilir."*
Egitim egrisi bunu birebir verdi. Ama kilitli sette olculen sonuc, E1'in
neredeyse saglikli gorunmesine yol acti — ve sebebi, senaryonun kendisi
kadar onemli.

**Egitim egrisi: ders kitabi asiri uyum.** 1000 kare, 200 epoch, augmentasyon
tamamen kapali.

| | epoch 1 | epoch 200 | degisim |
|---|---:|---:|---:|
| train_box_loss | 1.0218 | 0.1245 | **-0.8974** |
| train_cls_loss | 0.5025 | 0.0966 | -0.4059 |
| val_box_loss | 1.3867 | 1.4803 | **+0.0936** |
| val_cls_loss | 0.7748 | 1.2745 | **+0.4997** |
| mAP50 (val) | 0.9366 | 0.8449 | -0.0918 |

train kaybi cokerken (box loss 8 kat asagi) val kaybi **yukseliyor**.
train/val cls_loss farki 0.2723'ten 1.1778'e cikiyor: **4.3 kat**. Model 1000
kareyi ezberliyor ve genelleme yetenegini kaybediyor.

**Ama kilitli settaki sonuc bunu gostermiyor.**

| Kilitli tanı seti | v00 | E1 `best.pt` | E1 `last.pt` |
|---|---:|---:|---:|
| mAP50 | 0.9200 | 0.9190 (**-0.0010**) | 0.8318 (-0.0881) |
| mAP50-95 | 0.6707 | 0.6844 (**+0.0138**) | 0.6030 (-0.0677) |
| Precision | 0.9175 | 0.8818 (-0.0357) | 0.8456 (-0.0719) |
| Recall | 0.8785 | 0.8771 (**-0.0015**) | 0.8094 (-0.0691) |

`best.pt` ile E1, referanstan **ayirt edilemiyor** — mAP50-95'te hatta onun
uzerinde. 200 epoch suren, ders kitabi niteliginde bir asiri uyum kosusu,
standart raporlamayla **tamamen saglikli gorunuyor**.

**Neden.** `best.pt`, en yuksek val mAP50'yi veren epoch'un checkpoint'idir
ve o epoch **1 numarali epoch**. Model zaten yakinsamis `main_model.pt`'den
basladigi icin ilk epoch'ta tepedeydi; sonraki 199 epoch onu yalnizca bozdu.
Checkpoint secimi, kosunun tum kotu gecmisini atiyor.

#### Bu senaryonun asil dersi

Asiri uyum, **metriklerden degil egitim egrisinden** teshis edilir.
Yalnizca `best.pt` metriklerine bakan bir ajan (veya insan) burada "sorun
yok" der ve haklidir — o checkpoint gercekten saglikli. Yanlis olan, o
checkpoint'in kosuyu temsil ettigi varsayimidir.

Bu, D5'in dersinin ("erken durdurmanin gizledigi felaket") daha keskin bir
tekrari ve iki sey icin dogrudan kanit:

1. **Kanit sozlesmesi neden egitim egrisi ISTIYOR.** Sartname bolum 9 bunu
   sart kosuyor; E1 nedenini gosteriyor. `kanit.json` bu kosu icin
   `egitim_egrisi.train_val_farki.fark_acildi_mi = true` yaziyor - metrikler
   sessizken egri konusuyor.
2. **`_last_pt` varyantlari neden tutuluyor.** Ayni kosunun iki checkpoint'i
   arasindaki 0.088'lik mAP50 farki, kosunun saglikli mi bozuk mu oldugu
   sorusunun cevabini degistiriyor.

`results.csv`'ye iki satir birden yazildi (`E1` ve `E1 last_pt`); tek satir,
bu senaryo hakkinda yanlis bir izlenim birakirdi.

- reports/senaryo_E1/d1_metrics.json (best.pt)
- reports/senaryo_E1_last_pt/d1_metrics.json (last.pt)
- experiments/run_20260831_211150_E1_42/results.csv (egitim egrisi)
- veri_surumleri/v10_e1_overfitting_alt_kume/manifest.json

### E2 Sonucu — HIPOTEZ DESTEKLENMEDI: yakinsamis modelde epoch kesmek underfitting uretmez

E2'nin konfigdeki beklenen kaniti: *"train ve val birlikte dusuk, loss hala
iner."* Olculen bunun tam tersi.

| Olcum | v00 (referans) | E2 (5 epoch) | Fark |
|---|---:|---:|---:|
| mAP50 | 0.9200 | 0.9118 | -0.0081 |
| mAP50-95 | 0.6707 | 0.6800 | **+0.0094** |
| Precision | 0.9175 | 0.9118 | -0.0057 |
| Recall | 0.8785 | 0.8391 | -0.0394 |

Sinif bazinda: `tasit` +0.0020, `insan` +0.0141, `UAP` +0.0000, `UAI` -0.0487
(n=17). Yani E2, referanstan ayirt edilemiyor; mAP50-95'te hatta onun
**uzerinde**.

**Neden.** D serisi protokolu `main_model.pt` uzerinden fine-tune eder ve bu
model zaten yakinsamistir. E2'nin egitim egrisi bunu acikca gosteriyor:

| epoch | mAP50 | mAP50-95 |
|---|---:|---:|
| 1 | 0.9301 | 0.6810 |
| 3 | 0.8457 | 0.6155 |
| 5 | 0.9247 | 0.6911 |

Model **birinci epoch'ta zaten 0.93**. Ortadaki dus-cik dalgalanmasi
fine-tune gurultusu; bes epoch sonunda baslangic seviyesine donuyor.
Yakinsamis bir modelde epoch sayisini kesmek underfitting uretmez, cunku
ogrenilecek sey zaten ogrenilmistir.

**Ders.** "Az epoch = underfitting" esitligi yalnizca sifirdan (veya genel
amacli bir modelden) egitimde gecerlidir. Bu, D1'in ilk hipoteziyle ayni
turden bir hatadir: senaryonun tezi, kurulumun gerceklerini hesaba katmadan
yazilmisti.

Underfitting'i bu projede gostermenin dogru yolu, `yolo26n.pt` gibi genel
amacli bir modelden az epoch ile baslamak ve **v00n** referansiyla
karsilastirmaktir (v00n zaten mevcut: 23 epoch, mAP50 0.8126). Bu kosu
yapilmadi; E2 negatif sonuc olarak kayda gecti.

- reports/senaryo_E2/d1_metrics.json
- experiments/run_20260830_221852_E2_42/results.csv (egitim egrisi)

### optimizer=auto tuzagi — protokoldeki lr0 hic uygulanmamis

E1 kosusu baslarken Ultralytics'in su satiri fark edildi:

```text
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.001' and 'momentum=0.937'
and determining best 'optimizer', 'lr0' and 'momentum' automatically...
AdamW(lr=0.00125, momentum=0.9)
```

`optimizer` hicbir yerde belirtilmedigi icin varsayilan `auto` devredeydi ve
Ultralytics **lr0'i yok sayip** kendi degerini secti. Kontrol edildi: v00,
E2 ve E1 dahil tum kosularin `args.yaml` dosyasinda `optimizer: auto` yazili.

**Sonuclara etkisi yok.** Tum kosular ayni auto secimini aldigi icin
birbirleriyle karsilastirilabilirligi bozulmadi; protokolun asil isi
(kosulari kiyaslanabilir kilmak) yerine getirildi. Bozulan sey, **beyan
edilen degerin gercegi yansitmamasiydi**: `lr0: 0.001` okunuyordu, AdamW
0.00125 kosuyordu.

**Asil risk E3'teydi.** E3'un tum tezi "lr0 100 kat yuksek". `optimizer: auto`
ile bu deger yok sayilacak, E3 sessizce saglikli bir kosuya donusecek ve
"kararsizlik gozlenmedi" diye raporlanacakti — hatasiz calisan, yanlis
sonuc veren bir deney. E3'un sapmasi artik `optimizer: AdamW` degerini de
acikca tasiyor; auto'nun kendi secimi de AdamW oldugu icin tek degisen
ogrenme oranidir (0.00125 -> 0.10, ~80 kat).

`sabit` blok bilerek `optimizer: auto` olarak birakildi: acik bir optimizer
yazmak lr0'i baglayici hale getirir ve yeni kosulari mevcut D serisiyle
karsilastirilamaz kilardi. Uc test bu tuzagi kalicilastiriyor
(`test_e_serisi_protokol.py`), en onemlisi: **lr0'i degistiren bir sapma,
optimizer'i acikca yazmak zorundadir.**

### E4 Sonucu — cikarim cozunurlugu recall'u kor eder, precision'a dokunmaz

E4 tek E senaryosudur ki **egitim gerektirmez**: uyumsuzluk cikarim
tarafindadir. Protokol-uyumlu v00 referansi (imgsz=768) kilitli tanı setinde
bes cozunurlukte degerlendirildi.

| imgsz | egitime oran | mAP50 | mAP50-95 | precision | recall | mAP50 farki |
|---|---|---|---|---|---|---|
| 512 | 0.67 | 0.602 | 0.395 | 0.878 | 0.518 | -0.318 |
| 640 | 0.83 | 0.795 | 0.568 | 0.904 | 0.721 | -0.125 |
| **768** | **1.00** | **0.920** | **0.671** | **0.917** | **0.879** | **egitim cozunurlugu** |
| 1024 | 1.33 | 0.887 | 0.650 | 0.891 | 0.878 | -0.033 |
| 1280 | 1.67 | 0.878 | 0.625 | 0.858 | 0.855 | -0.042 |

**Uc bulgu:**

**1. Tepe tam egitim cozunurlugunde.** Egri 768'de maksimum yapiyor. Bu,
taramanin kendi ic kontrolu: eger tepe baska yerde ciksaydi, ya egitim
cozunurlugu kaydedilmemis ya olcum bozuk olurdu.

**2. Kucultmek buyutmekten cok daha pahali.** 768 -> 512 (x0.67) mAP50'yi
0.318 dusururken, 768 -> 1280 (x1.67) yalnizca 0.042 dusuruyor. Asimetri
mantikli: kucultmek termal imzalari kalici olarak yok eder, buyutmek ise
var olan bilgiyi yalnizca yeniden orneklemekle kalir.

**3. Uyumsuzluk modeli yaniltmaz, KOR EDER.** Precision cozunurluk boyunca
neredeyse sabit (0.858-0.918), recall ise cokuyor (0.518-0.879). Model
buldugunu dogru buluyor; sorun bulamamasi. Bu, D serisindeki etiket
bozulmalarindan farkli bir imzadir: orada precision da bozulur.

#### Sinif kirilimi: kayip kucuk nesnelerde yogunlasiyor

imgsz 512'de recall kaybi (iki oran z-testi, bbox sayilari uzerinde):

| sinif | n (bbox) | recall@768 | recall@512 | fark | z | p | %95 GA ortusmesi |
|---|---|---|---|---|---|---|---|
| tasit | 1264 | 0.834 | 0.763 | -0.070 | 4.41 | 1.0e-05 | ayrik |
| insan | 2718 | 0.739 | 0.713 | -0.026 | 2.16 | 3.1e-02 | ortusuyor |
| UAP | 15 | 1.000 | 0.242 | -0.758 | 4.17 | 3.1e-05 | ayrik |
| UAI | 17 | 0.941 | 0.353 | -0.588 | 3.59 | 3.3e-04 | ayrik |

**Sinir:** UAP (n=15) ve UAI (n=17) bbox sayilari cok dusuktur; buradaki
buyuk yuzde farklari birkac kutunun kaybina karsilik gelir. Yine de fark
istatistiksel olarak ayakta kaliyor — Wilson %95 araliklari **ortusmuyor**
(UAP: [0.796, 1.0] vs [0.109, 0.520]). Genel makro recall dususunun buyuk
kismini bu iki sinif surukluyor; `insan` sinifinda fark p<0.05 olsa da
guven araliklari ortustugu icin pratik olarak kucuktur.

Bu, D4 (kucuk nesne sinyal kaybi) ile ayni yonde ama farkli kokenli bir
bulgudur: D4'te sinyal **veriden** silinmisti, burada **cikarim
cozunurlugunden** siliniyor. Ikisini ayirt eden sey, E4'te veri surumunun
tamamen temiz olmasidir.

#### Konfigden bilincli sapma

`senaryolar/egitim/e4_cozunurluk_uyumsuzlugu.yaml` tek bir cift tanimliyordu:
`imgsz_egitim: 640`, `imgsz_degerlendirme: 1280`. Bunun yerine mevcut
protokol-uyumlu 768 modeli uzerinde bes noktali tarama yapildi. Gerekce: tek
cift yalnizca bir sayi verir, tarama egrinin **bicimini** gosterir (tepe
noktasi ve asimetri ancak boyle gorulur); ayrica 640'ta yeni bir model
egitmek, egitim protokolunu degistirmeden ayni olguyu olcmenin pahali
yoludur. Sapma `reports/senaryo_E4/e4_cozunurluk_taramasi.json` icinde
`not` alaninda kayitlidir.

- reports/senaryo_E4/e4_cozunurluk_taramasi.json
- reports/senaryo_E4/e4_sinif_anlamlilik.json
- reports/senaryo_E4_imgsz512/ ... senaryo_E4_imgsz1280/ (ham degerlendirmeler)

---

## Ajan Denemesinin Bilinen Siniri: kosu_05 iki degiskenle ayriliyor

Ajana verilen her kosunun saglikli referanstan (v00) **tek** degiskenle —
veri surumuyle — ayrilmasi gerekir. Bir kosu bu kurali ihlal ediyor.

`d2b_20260820_final` (ajanda **kosu_05**), ayni eksik-etiket bozulmasini
tasir ama `final_best.pt` baslangic modelinden ve 30 epoch ile egitilmistir;
digerleri `main_model.pt` ve 11 epoch'tur. Bozulma gercekten mevcuttur, bu
yuzden beklenen teshis (`eksik_etiket`) gecerlidir; ancak ajanin gordugu
precision/recall farkinin bir kismi bozulmadan degil **baslangic modelinden**
gelir. Ayni belgede "D2b Final_best Karsilastirmasi" bolumu bu iki kosu
arasindaki farki zaten olcuyor: mAP50 -0.0077, precision +0.0236,
recall -0.0299. Yani konfonderin buyuklugu bilinmektedir ve bozulmanin
etkisinden kucuktur.

**Neden kapatilmadi.** `kosu_NN` numaralari `results.csv` satir sirasina
dayanir; ortadan bir kosu cikarmak sonraki tum numaralari kaydirir ve yarim
kalan ajan denemesinin tamamlanmis kosularini gecersiz kilardi. Konfonder bu
nedenle kapatilmak yerine `teshis/ajan/araclar.py::BILINEN_TABAN_MODEL_SAPMASI`
altinda **beyan edildi** ve `test_taban_model_sapmalari_beyan_edilmis_olanlarla_sinirli`
testiyle sinirlandirildi: beyan edilmemis yeni bir taban model sapmasi
eklenirse test kirilir.

Kalici cozum, numaralandirmayi satir sirasindan koparmaktir (ornegin run_id
uzerinden sabit bir esleme). Deneme bastan kosulacaksa bu once yapilmalidir.

**Bu, aynen v00n/D1n ciftinin ajandan dislanmasiyla ayni gerekcedir** — orada
fark tamamen model kapasitesinden geliyordu ve kosular tamamen cikarildi.
Buradaki fark, bozulmanin da gercekten mevcut olmasidir.

---

## Sartnameye Uyum Denetimi (2026-08-30)

`docs/proje-brifingi-v2.1.md` ile mevcut durum karsilastirildi. Uc bosluk
bulundu; ikisi kapatildi, biri olcum bekliyor.

### 1. C3 bootstrap protokolu — KISMEN KAPATILDI

Sartname bolum 8, guven araliklarinin **goruntu birimli ve kaynak grubuna
gore tabakali** bootstrap ile hesaplanmasini istiyor. Gerekce acik: ayni
karedeki kutular iliskilidir; kutu bazli hesap araligi yapay olarak daraltir.

Projede yayimlanmis araliklarin buyuk kismi ise kutu sayilari uzerinde
**Wilson** ile hesaplandi — tam da sartnamenin uyardigi yontem.

`tabakali_goruntu_bootstrap` eklendi ve fark olculdu:

| kumelenme | bootstrap / Wilson genislik orani |
|---|---|
| kare basina 1 kutu (kumelenme yok) | 1.02x |
| kare basina 3-4 kutu (gercekci) | 1.50x |
| kare basina 5 kutu | 2.19x |
| kare basina 10 kutu | 2.91x |

**Sonuc:** yayimlanmis araliklar muhtemelen ~1.5 kat dar. Aralari genis olan
bulgular (E4'te UAP ve UAI) bundan etkilenmez. **Sinirda olanlar yeniden
kontrol edilmelidir** — ozellikle E4'te `tasit`: araliklar 0.026 aralikla
ayrik, %50 genisleme bu payi buyuk olcude yer.

Kalan is: `goruntu_kayitlari` iceren yeni kirilim olcumleri uretmek
(`metrikler.py` artik bunu yaziyor, ancak mevcut `reports/kirilim/*.json`
dosyalari eski surumle uretildi ve bu alani icermiyor).

### 2. Kanit sozlesmesi (`kanit.json`) — KAPATILDI (icerik hala eksik)

Sartname bolum 9, her egitim kosusu icin tek bir
`experiments/<kosu_adi>/kanit.json` uretilmesini istiyor: sinif bazli
metrikler bbox sayisi ve bootstrap GA ile, boyut binleri, kaynak bazli
metrikler, confusion matrix, hata ornekleri, egitim egrileri, veri surumu ve
hiperparametreler, kare **ve** benzersiz kaynak sayisi.

Bu bilgilerin tamami projede **mevcuttu**, ancak dort ayri yere dagilmisti:
`reports/senaryo_*/d1_metrics.json`, `reports/kirilim/*.json`,
`experiments/*/results.csv`, `experiments/*/run_manifest.json`.

`teshis/degerlendirme/kanit.py` bunlari tek semada birlestiriyor ve 15
kosunun tamami icin dosya uretiyor. Ancak **hicbiri sozlesmeyi tam
karsilamiyor**; modul eksigi silmiyor, `sozlesme_durumu.eksikler` altinda
tek tek yaziyor:

| Eksik madde | Etkilenen kosu |
|---|---|
| hata ornekleri (`hata_galerisi_*`) | 15 / 15 — yalnizca D2a'nin galerisi var |
| goruntu birimli guven araligi | kirilimi olan kosular (eski olcum, `goruntu_kayitlari` yok) |
| boyut/kaynak kirilimi | d6a, v00n, d1n, E4, E2 (`reports/kirilim/` dosyasi yok) |

Ogrenme orani alani iki deger tasir: `lr0_beyan_edilen` (0.001) ve
`lr0_gecerli` (0.00125). `optimizer=auto` beyani yok saydigi icin yalnizca
birincisini yazmak yaniltici olurdu.

**Uretim sirasinda bir hata yakalandi.** Ilk surum kanit dosyasinin yerini
`weights_path`'ten turetiyordu. D6a ve E4 kendi agirliklarini egitmez,
v00'un agirliklarini yeniden degerlendirir; ucu de ayni `weights_path`'i
gosterir. Sonuc: E4'un kaniti v00'un dizinine yazildi ve **v00'unkini ezdi**
— dosya adi dogru, icerigi baska kosunun (v00'un kanit dosyasi UAP recall
0.2415 gosteriyordu; bu E4'un imgsz=512 degeri, v00'un gercek degeri 1.0).
Artik dizini yalnizca onu **egiten** kosu sahipleniyor; yeniden
degerlendirmeler `reports/kanit/<run_id>.json` altina yaziliyor. Dort test
bu davranisi koruyor.

Ayni is sirasinda `results.csv`'de bir belirsizlik de giderildi: iki farkli
kosu (`d2b_20260820_main` ve `d2b_20260820_final`) ayni `scenario` degerini
(`D2b`) tasiyordu ve demo bunu run_id'ye gore elle yamaliyordu. Senaryo adi
kaynagindan netlestirildi (`D2b final_best`), demo yamasi kaldirildi. Ajanin
kosu kumesi ve numaralari degismedi.

### 3. C2 kontrolu (seed 7) — ACIK, ve en onemlisi bu

Sartname bolum 8'deki kontrol kosullari:

| Kod | Tanim | Amac | Durum |
|---|---|---|---|
| C1 | Saglikli yapilandirma, seed 42 | Referans taban | **var** (v00) |
| C2 | Ayni yapilandirma, **seed 7** | Ajan seed kaynakli dalgalanmayi "sorun" saniyor mu? | **YOK** |
| C3 | Bootstrap guven araligi | Farklar anlamli mi | kismen (yukari bkz.) |

`results.csv`'deki **her kosu seed 42**. Yani su an projenin merkezi
iddiasini — "ajan bozulmayi metrik imzasindan teshis edebilir" — sinayan
negatif kontrol eksik: ajanin, hicbir bozulma olmayan ama yalnizca farkli
seed ile egitilmis bir kosuda **sorun uydurup uydurmadigini** bilmiyoruz.

Bu, D veya E serisine bir senaryo daha eklemekten daha degerlidir: bir
teshis sisteminin yanlis pozitif orani olculmeden dogruluk iddiasi eksik
kalir. Tek bir egitim kosusu maliyeti vardir (v00 protokolu, seed 7).

# Termal Teshis Ajani

Bu belge projenin ana takip, onboarding ve teslim dokumanidir. Bir mentor veya
baska bir AI ajan bu dosyayi okuyarak projenin amacini, dosya yapisini, mevcut
durumunu, degismez kurallarini ve sonraki adimini anlayabilmelidir.

## 1. Projenin Amaci

Bu proje termal drone goruntulerinde YOLO nesne tespit modelinin kontrollu veri
ve egitim arizalari altinda nasil bozuldugunu olcer. Ajan katmani bu olcumlerden
kanita dayali teshis uretir.

Sinif sozlesmesi degismez:

| ID | Sinif |
|---:|---|
| 0 | tasit |
| 1 | insan |
| 2 | UAP |
| 3 | UAI |

Arastirma sorusu: Termal nesne tespit sistemi hangi veri, etiket, dagilim,
egitim ve alan kaymasi kosullarinda bozulur; bir LLM/ajan bu bozulmayi yeterli
kanitla teshis edebilir mi?

## Bakım Günlüğü

Bu bölüm, projede yapılan düzenleme/temizlik/eksik-giderme işlerinin
kronolojik kaydıdır. Her mühendislik değişikliğinden sonra buraya yeni bir
madde eklenir; böylece hangi sorunun ne zaman ve nasıl giderildiği README
üzerinden takip edilebilir. En yeni kayıt en üstte durur.

### 2026-08-26 (3) — v00 sağlıklı referans eğitildi: D1 hipotezi çöktü, diğer üçü sağlamlaştı

Katalog ve 8 senaryo config'inin tamamı `kaynak_surum: v00_saglikli` diyordu
ama böyle bir sürüm hiç üretilmemişti. Karşılaştırmalar, protokolle hiç
fine-tune edilmemiş `main_model.pt`'ye karşı yapılıyordu — yani her senaryo
farkı `(bozulma etkisi) + (fine-tune etkisi)` toplamıydı.

v00 üretildi (manifest-only, kaynak dataset'e dokunulmadan) ve senaryolarla
birebir aynı protokolde eğitildi (11 epoch, 2,28 saat). Sonuç, fine-tune'un
tek başına küçük olmadığını gösterdi: **insan recall'i 7,7 puan düşürüyor**
(0.8157 → 0.7391), precision'ı 2 puan yükseltiyor. Yani bir çalışma noktası
kayması yaratıyor.

**D1 hipotezi çöktü.** Raporlanan "insan recall -0.0912"nin -0.0765'i temiz
fine-tune'un kendi etkisiymiş. D1'e özgü kısım yalnızca **-0.0147** ve
istatistiksel olarak anlamlı değil (z=-1.22, n=2718). Hiçbir sınıfta anlamlı
fark yok.

Nedeni öğretici ve deney kurgusuyla ilgili: tüm senaryolar `main_model.pt`'den
başlıyor, ama bu model zaten **aynı dataset'in tamamı** üzerinde eğitilmiş.
İnsan karelerinin %90'ı çıkarılsa bile model insanları zaten biliyor ve 11
epoch'luk kısa bir fine-tune bunu unutturmuyor. Bu, iki bozulma türünü
ayırıyor: **aktif bozulmalar** (D2a/D2b/D3 — modele yanlış bilgi öğretir,
mevcut bilgisiyle çelişir) az epoch'ta bile güçlü etki üretiyor; **pasif
bozulma** (D1 — sadece daha az örnek) eğitilmiş bir modeli etkilemiyor.

Destekleyici kanıt: v00, D1 ve D2a'da `best.pt` **epoch 1'den** seçilmiş
(model hiçbir epoch'ta başlangıçtan iyi olamamış); D2b/D2b_fb/D3'te epoch
3/11/14. Eşit maruziyeti test etmek için D1 ve v00'ın `last.pt`'leri (epoch
11) de ölçüldü — sonuç değişmedi (-0.0150, z=-1.27).

**Diğer üç senaryo v00 tabanında da sağlam durdu** ve imzaları netleşti:
D2a mAP50-95'i en çok düşürüyor (-0.0544), D2b precision'ı düşürürken insan
recall'ini +0.1016 artırıyor (fazladan kutu üretiyor), D3 precision'ı en sert
düşürüyor (-0.2047) ve düşüş karıştırılan sınıfta yoğunlaşıyor (UAP precision
-0.4887). Bu ayrışma, ajanın metrik imzasından teşhis koyabilmesinin temeli.

- README'ye "v00 Sağlıklı Referans ve Senaryo Karşılaştırması" otoriter tablosu
  eklendi; her sayı kaynak JSON'dan programatik doğrulandı. Eski D1 tablosunun
  başına uyarı konuldu.
- D2a, D2b ve D2b final_best sınıf bazlı precision/recall için yeniden
  değerlendirildi (metrik JSON'ları eski formattaydı).
- `results.csv`'ye v00 satırı **sona** eklendi — araya eklemek ajanın
  `kosu_NN` eşlemesini kaydırırdı; test bunu doğruluyor.
- `demo/app.py` yeni senaryoda çöküyordu (`scenario_info` sabit kodluydu, ilk
  denetimde işaret ettiğim kırılganlık). Demo render testi bunu otomatik
  yakaladı; hem v00 girdisi eklendi hem de eksik girdide çökmek yerine
  varsayılana düşecek şekilde düzeltildi.
- **Açık öneri:** D1'i geçerli kılmak için senaryo, dataset'in tamamını görmüş
  `main_model.pt` yerine genel amaçlı bir başlangıç modelinden (repoda mevcut
  `yolo26n.pt`) eğitilmeli; v00 da aynı başlangıçtan yeniden koşulmalı.

### 2026-08-26 (2) — D3 sızıntısız yeniden koşuldu: gizlenen etki ortaya çıktı

D3, düzeltilmiş config ile yeniden koşuldu (24 epoch, 5,6 saat). Tek fark:
eğitim val'i artık operasyonel `dataset/images/val`; kilitli tanı seti
eğitimde hiç kullanılmıyor. Veri bozulması değişmedi (aynı `v04` sürümü,
aynı etiketler, seed=42).

Sızıntı, D3'ün etkisini **gizliyormuş** — düzeltilince tablo tersine döndü:

| Ölçüm | Sızıntılı koşu | Sızıntısız koşu | |
|---|---:|---:|---|
| Precision farkı | **+0.0110** | **-0.1846** | en çarpıcı değişim |
| mAP50 farkı | -0.0258 | -0.0413 | |
| mAP50-95 farkı | -0.0328 | -0.0497 | |
| UAP AP50 farkı | -0.0622 | 0.0000 | |
| UAI AP50 farkı | 0.0000 | -0.1359 | |

Sızıntılı koşuda precision baseline'ın *üstünde* çıkıyordu — bir bozulma
senaryosu için mantıksız bir sonuçtu ve zaten şüphe uyandırmıştı.
Sızıntısız koşuda precision **18,5 puan düşüyor** ve bu, D3'ün en belirgin
imzası hâline geliyor.

Confusion matrix da çok daha net bir hikâye anlatıyor. Sızıntılı koşu
çift yönlü, dağınık bir karışıklık gösteriyordu (UAP→UAI %73, UAI→UAP %41).
Sızıntısız koşu ise **tek yönlü** bir kayma gösteriyor: gerçek UAI
kutularının %65'i UAP olarak tahmin ediliyor, gerçek UAP'lerin tamamı doğru
sınıflandırılıyor. Yani UAP bir "çekici sınıf" hâline gelmiş. Bu, iki sınıf
bazlı metrikle birbirini doğruluyor: UAP precision 0.9417 → 0.4730 (UAI'ler
UAP diye işaretleniyor), UAI recall 0.8286 → 0.6471 (gerçek UAI'ler
kaçırılıyor).

Ders: değerlendirme setinin checkpoint seçimine sızması sadece sonucu
iyimser yapmakla kalmamış, **aranan etkiyi de maskelemiş**. `best.pt` tanı
setinde en iyi görünen epoch'tan seçildiği için, bozulmanın o sette
yarattığı hasar sistematik olarak küçültülmüş.

- `results.csv` D3 satırı yeni koşuyla değiştirildi (satır sırası korundu,
  ajan eşlemesi kaymadı). Sızıntılı koşu `experiments/run_D3_42_local` ve
  `reports/d3_sonuc/` altında tarihsel kayıt olarak duruyor.
- `demo/data_loader.py` D1 ve D3 için yeni rapor klasörlerine yönlendirildi.
- README'deki D3 uyarı bloğu kaldırıldı; tablo sınıf bazlı precision/recall
  ile genişletildi ve her sayı kaynak JSON'dan programatik doğrulandı.

### 2026-08-26 (1) — D1 ortak protokolle yeniden koşuldu: genel kaybın çoğu protokol artefaktıymış

D1, `senaryolar/egitim_protokolu.yaml` ortak protokolüyle (lr0=0.001,
warmup_epochs=3) yeniden koşuldu. Yeni koşu eskisinden **yalnızca** bu iki
parametrede farklı — diğer 27 eğitim parametresi (batch, imgsz, seed,
augmentasyon, patience, cos_lr, başlangıç modeli, veri sürümü) birebir aynı
olduğu programatik olarak doğrulandı. Süre de neredeyse aynı: 11 epoch /
88 dk (ilk koşu 11 epoch / 89,5 dk).

Sonuç, D1'in yorumunu önemli ölçüde değiştirdi:

| Ölçüm | İlk koşu (lr0=0.0005) | Ortak protokol | |
|---|---:|---:|---|
| mAP50 farkı | -0.0270 | **-0.0082** | genel kayıp büyük ölçüde kayboldu |
| mAP50-95 farkı | -0.0286 | **-0.0041** | neredeyse sıfır |
| Precision farkı | -0.0104 | **+0.0186** | baseline'ın üstüne çıktı |
| İnsan recall farkı | -0.0957 | **-0.0912** | değişmedi |

**Asıl bulgu ayakta, ama artık izole:** insan eğitim karelerinin %90'ı
çıkarılınca insan recall'i 9,1 puan düşüyor — bu etki iki koşuda da
neredeyse aynı, yani protokol değişikliğine dayanıklı. Buna karşılık ilk
koşunun gösterdiği **genel** performans kaybının büyük kısmı veri
bozulmasından değil, düşük öğrenme oranından kaynaklanıyormuş. Ortak
protokolle koşulduğunda geriye tam olarak beklenen imza kalıyor:
hedeflenen sınıfın recall'inde belirgin düşüş, genel tespit kalitesinde
neredeyse değişim yok. Bu, "tek değişken değişir" ilkesinin neden
önemsenmesi gerektiğinin somut kanıtı.

- `results.csv`'de D1 satırı yeni koşuyla değiştirildi (satır sırası
  korundu, böylece ajanın anonim `kosu_NN` eşlemesi kaymadı — testle
  doğrulandı). İlk koşu `experiments/run_20260817_222323_D1_42` ve
  `reports/d1_sonuc/` altında tarihsel kayıt olarak duruyor.
- `teshis/degerlendirme/d1_sonuc.py` artık sınıf bazlı **precision ve
  recall** de kaydediyor. Rule 8 "sınıf AP, recall, bbox n" istiyordu ama
  metrik JSON'u yalnızca AP tutuyordu; README tabloları için recall'ı elle
  Ultralytics çıktısından okumak gerekiyordu. Sınıf adları artık
  `ap_class_index` üzerinden eşleniyor, böylece val'de örneği olmayan bir
  sınıf olursa isimler kaymaz.

### 2026-08-25 (6) — D3'te değerlendirme seti sızıntısı bulundu; boyut bazlı metrik modülü yazıldı

**Ciddi bulgu — D3'ün kayıtlı sonucu iyimser yanlı.**
`scripts/local_d3.py`, ürettiği `data.yaml`'a eğitim val'i olarak
`val_diagnostic/images` yazıyordu. Diğer tüm senaryolar (D1, D2a, D2b, D2b
final_best) operasyonel `dataset/images/val` kullanıyor. Ultralytics
`best.pt`'yi val üzerindeki en iyi skora göre seçtiği için, **D3'ün
`best.pt`'si sonradan tüm senaryoları karşılaştırmak için kullandığımız
kilitli tanı setinin üzerinde seçilmiş oldu** — yani checkpoint seçimi
değerlendirme setine baktı.

Bunun pratik anlamı: D3 tablosundaki sayılar (özellikle baseline'ın
*üzerinde* çıkan `precision +0.0110` değeri, bir bozulma senaryosu için
beklenmedik bir sonuçtu) D1/D2a/D2b ile adil karşılaştırılabilir değil.
D3'ün UAP↔UAI çapraz karışıklık bulgusu confusion matrix'ten geldiği ve
çok belirgin olduğu için niteliksel olarak ayakta kalır; ancak **sayısal
tablo yeniden koşulmadan kesin kabul edilmemelidir.**

- `scripts/local_d3.py` düzeltildi (artık kaynak dataset'in val/test
  bölmelerini yazıyor, diğer senaryolarla aynı).
- `tests/test_veri_surumu_val.py`: hem üretici scriptlerin hem diskteki veri
  sürümlerinin kilitli tanı setini eğitim val'i yapmadığını doğrular. Mevcut
  `v04_d3_...` sürümü tarihsel kayıt olduğu için (koşunun provenansını
  koruyor) `xfail` ile açıkça işaretlendi; yeni bir ihlal testi kırar.
- **Açık karar:** D3'ün düzeltilmiş config ile yeniden koşulması gerekiyor
  (~5 saat GPU, orijinal koşu 17.438 sn sürmüştü). Bu karar kullanıcıya
  bırakıldı; D1 yeniden koşusu bitmeden ikinci koşu başlatılmadı (kural 10).

**Yeni yetenek — boyut bazlı metrikler.**
`teshis/degerlendirme/metrikler.py` bugüne kadar boş bir stub'dı, oysa
dosya sözleşmesi "sınıf, boyut ve kaynak grubu bazlı metrik" vaat ediyordu.
Artık kilitli tanı setindeki her gerçek kutuyu etkin piksel boyutuna göre
bantlara ayırıp bant başına recall hesaplıyor. Etkin boyut çözünürlüğe göre
normalize ediliyor (dataset'te hem 1024×1024 hem 640×512 kare var), formül
`teshis/veri/istatistik.py` ile aynı. Bant sınırları D4'ün 16px eşiğiyle
hizalı, böylece "eğitimden çıkarılan boyut bandı" ile "recall'i ölçülen
bant" birebir örtüşüyor. Toplam mAP, "model küçük nesneleri kaçırmaya
başladı" iddiasını gösteremez — küçük nesneler bbox sayısının küçük bir
kısmı olduğu için toplam metrikte kaybolurlar; bu modül tam da o boşluğu
kapatıyor. 15 birim testi (IoU, normalizasyon, bant sınırları, eşleştirme)
`tests/test_metrikler.py` içinde.

### 2026-08-25 (5) — Tamamlanan bölümler baştan sona test edildi; LLM paketindeki kayma düzeltildi

Bu tura kadar yapılan her şey sistematik olarak doğrulandı. Sonuç: **iki
gerçek hata bulundu ve düzeltildi**, geri kalan her şey temiz çıktı.

Doğrulananlar (hepsi geçti):

- 66 birim/sözleşme testi, tüm `.py` dosyalarının derlenmesi, temiz git ağacı.
- `results.csv`'deki 5 koşunun **her metriği** kaynak rapor JSON'larıyla
  birebir karşılaştırıldı (`mAP50`, `mAP50-95`, precision, recall ve 4 sınıf
  AP50'si) — sapma yok; her `weights_path` diskte mevcut.
- README'deki D3 tablosunun tüm sayıları kaynak veriden yeniden hesaplandı —
  birebir tutuyor.
- 14 CLI giriş noktasının tamamı (`teshis.*` modülleri ve `scripts/*.py`).
- Ajan katmanı uçtan uca (yerel): 6 koşunun tamamı için araçlar, araç
  dispatch'i (geçerli / bilinmeyen araç / geçersiz argüman), **anonimlik
  sızıntısı taraması** (senaryo kodu, manifest, veri sürümü adı, run_id
  sızmıyor), kayıtlı Gemini cevabının şema doğrulaması ve puanlamanın
  kayıtlı `llm_score.json` ile birebir eşleşmesi.
- Demo: 5 sayfa × 6 senaryo × 4 galeri sıralaması, Streamlit AppTest ile
  headless çalıştırıldı — hiçbirinde istisna yok, tarayıcıda 0 kırık görsel.

**Bulunan hata 1 — LLM paketi senaryolardan geri kalmıştı.**
`scripts/prepare_llm_trial.py` dört koşuyu dosya yollarıyla hardcode
ediyordu. D2b final_best ve D3 eklenince ajan araçları 6 koşu sunar hale
geldi ama `answer_key.json` 4 koşuda kaldı; yeni iki koşu **sessizce
puanlanamaz** durumdaydı (`paketi_puanla` cevap anahtarı üzerinden döndüğü
için fazladan cevaplar yok sayılıyordu). Script artık paketi ve cevap
anahtarını doğrudan `teshis/ajan/araclar.py`'den türetiyor — yani LLM'ye
verilen paket ile ajanın araçlarla gördüğü veri artık ayrışamaz; yeni bir
senaryo `results.csv`'ye eklendiğinde pakete otomatik giriyor.

**Bulunan hata 2 — puanlama kalıpları konuma bağlıydı.**
`ANAHTAR_KALIPLAR` `kosu_01..kosu_04` ile anahtarlanmıştı. `kosu_NN`
numaraları `results.csv` satır sırasından türetildiği için araya bir satır
eklenirse kayar ve statik cevap anahtarı **sessizce yanlış senaryoyu**
puanlamaya başlardı. Kalıplar artık senaryonun kendisine bağlı olan
`expected` etiketiyle anahtarlanıyor (`sinif_yetersizligi`, `eksik_etiket`,
`uap_uai_sinif_karisikligi`, …). Refactor davranışı korudu: kayıtlı Gemini
cevabı yeniden puanlandığında sonuç yine birebir `0.833`.

Küçük düzeltme: `prepare_llm_trial.py`'de argparse yoktu, bu yüzden
`--help` yardım göstermek yerine scripti **çalıştırıp** kanıt paketinin
üzerine yazıyordu. Artık argparse var ve mevcut paketin üzerine `--force`
olmadan yazmayı reddediyor (üzerine yazmak, kayıtlı `gemini_response.json`'ı
üretildiği paketten kopardığı için).

Kalıcı korumalar: `tests/test_llm_paketi.py` (paket ↔ araç tutarlılığı,
cevap anahtarı kayması, pakette senaryo adı sızıntısı) ve
`tests/test_demo_render.py` (her sayfa/senaryo/sıralama kombinasyonu).
Test sayısı 43 → 66.

**Not:** Kayıtlı LLM paketi bilerek 4 koşuluk bırakıldı. 6 koşuya
genişletmek `gemini_response.json`'ı üretildiği paketten koparır ve bu
ortamda `GEMINI_API_KEY` olmadığı için deneme yeniden koşulamaz. Ajan bir
sonraki kez çalıştırılacağında `scripts/prepare_llm_trial.py --force` ile
paket yenilenmeli ve LLM denemesi tekrarlanmalıdır.

### 2026-08-25 (4) — Demo yeniden tasarlandı, kullanılmayan tüm görseller konsola bağlandı

Sorun: demo, sık kullanılan bir dashboard şablonuydu (koyu teal/turuncu
gradyan hero, yuvarlak kartlar) ve eğitim/değerlendirme sırasında üretilen
görsellerin çoğunu hiç göstermiyordu. Özellikle 50 görüntülük D2a hata
galerisi ve her koşunun epoch bazlı `results.csv` eğitim eğrileri tamamen
kullanılmıyordu.

Yapılanlar:

- **Yeni tasarım yönü — ölçüm aleti / mühendislik konsolu.** Monospace
  tipografi, keskin köşeler, ince ızgara çizgileri; gradyan, gölge ve
  yuvarlak köşe kaldırıldı. Metrikler `MAP50  0.9073  ▾ -0.0258` biçiminde
  hizalı readout satırları olarak gösteriliyor. Yalnızca sistemde hazır
  bulunan monospace yazı tipleri kullanılıyor — sunumda internet olmasa da
  görünüm bozulmaz.
- **Yeni "Hata Galerisi" bölümü:** `reports/*_hata_galerisi/` klasörleri
  otomatik keşfediliyor. FN/FP dağılım grafiği, özet istatistikler ve
  skora/FN'e/FP'ye/IoU'ya göre sıralanabilir kareler. Bu materyal daha önce
  hiç kullanılmıyordu.
- **Epoch bazlı eğitim eğrileri:** `experiments/<run>/results.csv` artık
  interaktif olarak çiziliyor (doğrulama metrikleri + eğitim kayıpları,
  sekmeli). Genel Bakış'ta her koşu için unicode sparkline özeti var.
  Koşu klasörü `results.csv`'deki `weights_path` sütunundan türetiliyor,
  ayrıca eşleme tablosu tutulmuyor — yeni senaryo otomatik gelir.
- **Eşik eğrileri** (BoxPR/BoxF1/BoxP/BoxR), **etiket-vs-tahmin** karşılaştırmaları
  (3 çift, önceden yalnızca 3 tekil görsel kapalı bir expander içindeydi) ve
  **bozulmuş eğitim verisi önizlemesi** (`train_batch*.jpg`, `labels.jpg`)
  eklendi.
- **Düzeltilen gerçek hata:** `st.bar_chart` varsayılan olarak yığıyordu, bu
  yüzden metrik profili grafiğinin y ekseni 0-3 arasına çıkıyor ve
  mAP+precision+recall toplanmış gibi görünüyordu. `stack=False` eklendi
  (bu hata eski tasarımda da vardı).
- `use_container_width` yerine `width="stretch"` kullanıldı (deprecation
  uyarıları gitti); `requirements-demo.txt` bu API için `streamlit>=1.49`
  olarak güncellendi.
- Beş bölümün tamamı tarayıcıda doğrulandı: sayfa geçişleri, senaryo seçimi,
  galeri sıralaması, tüm görsellerin yüklendiği (0 kırık) ve sunucu tarafında
  hata/uyarı olmadığı kontrol edildi.

### 2026-08-25 (3) — D3 senaryosu tamamlandı; Ultralytics çıktı yolu hatası düzeltildi

D3'ün eğitimi bu oturumdan önce zaten bitmişti (`experiments/run_D3_42_local`,
22 epoch, `best.pt` mevcut) ama hiç diagnostic değerlendirmesi yapılmamıştı —
roadmap'te "D3-D6 veri senaryoları" hâlâ eksik görünüyordu. Bunu bitirmek,
90 dakikalık yeni bir eğitimden çok daha ucuzdu (~20 saniyelik bir val
koşusu), o yüzden önce bunu tamamladım.

- `python -m teshis.degerlendirme.d1_sonuc --model experiments/run_D3_42_local/weights/best.pt --data val_diagnostic/data.yaml --output reports/d3_sonuc --scenario D3`
  çalıştırıldı.
- **Yan bulgu:** bu ortamdaki Ultralytics sürümü (8.4.121), göreceli bir
  `project=` yolu verildiğinde çıktıyı sessizce `runs/detect/<project>/`
  altına yazıyor — bizim kodumuz ise dosyaları doğrudan `<project>` altında
  bekliyor (demo, rapor klasörleri). D1/D2a/D2b bu sorunu yaşamamıştı çünkü
  o çalıştırıldığı sırada farklı bir Ultralytics sürümü kurulu olmalıydı;
  D3'te görsel kanıtlar `runs/detect/reports/d3_sonuc/...` altında
  kayboluyordu. [teshis/degerlendirme/d1_sonuc.py](teshis/degerlendirme/d1_sonuc.py),
  [model.py](teshis/degerlendirme/model.py), [karsilastir.py](teshis/degerlendirme/karsilastir.py)
  içindeki `project=` parametreleri `.resolve()` ile mutlak yola çevrildi
  (kos.py ve local_d2b.py/local_d3.py'de zaten böyleydi). Regresyon testi:
  [tests/test_degerlendirme_project_yolu.py](tests/test_degerlendirme_project_yolu.py).
- D3 sonucu README'ye eklendi (bkz. "D3 Sonucu" bölümü). Confusion matrix,
  aggregate AP50'nin gizlediği güçlü bir UAP<->UAI çapraz karışıklığını
  ortaya çıkardı (true UAP'nin %73'ü UAI olarak tahmin edildi) — bu, D-serisi
  içindeki en net nedensel kanıt örneklerinden biri.
- `results.csv`'ye D3 satırı eklendi; `demo/data_loader.py` ve
  `demo/app.py` D3'ü tanıyacak şekilde güncellendi. Bu arada
  `demo/app.py`'deki hardcoded `"Tamamlanan kosu": "5"` metrikini
  `len(scenarios)` ile dinamik hale getirdim — yeni bir senaryo eklendiğinde
  bir daha elle güncellenmesi gerekmeyecek.
- `.claude/launch.json` eklendi (streamlit demo'yu port 8577'de başlatan
  kısayol); demo, tarayıcıda D3 dahil doğrulandı (ana sayfa metrikleri ve
  grafikler), ancak bu oturumun otomasyon tarayıcısında Streamlit'in
  sidebar'ı hiç DOM'a render olmadı (konsol/sunucu hatası yok) — bu koda
  ait bir regresyon degil gibi görünüyor (sidebar kodu değiştirilmedi);
  D3'e özgü veri fonksiyonları doğrudan Python'da ayrıca doğrulandı.
  Kullanıcı gerçek tarayıcısında sidebar'ı kontrol etmek isteyebilir.

### 2026-08-25 (2) — Ajan katmanı gerçek koda kavuştu, ilk birim testleri eklendi

Sorun: `teshis/ajan/*.py` ve `teshis/servis/*.py` tamamen tek satırlık
docstring'den ibaretti. Projenin adı "Termal Teşhis Ajanı" olmasına rağmen
gerçekte çalışan tek şey `scripts/run_gemini_trial.py` içindeki, hiç araç
(tool) kullanmayan tek seferlik bir Gemini metin çağrısıydı. Ayrıca
`tests/` klasöründe hiç test dosyası yoktu.

Yapılanlar:

- [teshis/ajan/araclar.py](teshis/ajan/araclar.py): gerçek function-calling
  araçları (`kosu_listesini_getir`, `kosu_metriklerini_getir`,
  `baseline_farkini_getir`, `bbox_sayilarini_getir`). `katalog.yaml`'daki
  `manifest_ajana_verilmez: true` kuralını kod seviyesinde uyguluyor —
  hiçbir fonksiyon manifest veya senaryo adı okumuyor, sadece
  `results.csv`/`reports/` metriklerini anonim `kosu_NN` kimlikleriyle
  sunuyor.
- [teshis/ajan/semalar.py](teshis/ajan/semalar.py): ajan çıktısı için
  `TESHIS_SEMASI` + `teshis_dogrula` (programatik doğrulama — eskiden
  Gemini'nin JSON'u hiç doğrulanmadan diske yazılıyordu) ve Gemini
  function-calling için `ARAC_BILDIRIMLERI`.
- [teshis/ajan/puanlama.py](teshis/ajan/puanlama.py): pilot puanlama
  mantığı `scripts/score_llm_trial.py`'den buraya taşındı, tek kaynak
  oldu; script artık ince bir CLI sarmalayıcısı. Taşıma sonrası çıktı
  gerçek `reports/llm_trial/gemini_response.json` ile bit-bit karşılaştırıldı,
  fark yok.
- [teshis/ajan/ajan.py](teshis/ajan/ajan.py): gerçek Gemini function-calling
  döngüsü (`teshis_uret`, `kor_deneme_calistir`). **Not:** bu kısım canlı
  bir `GEMINI_API_KEY` ile bu ortamda uçtan uca test edilmedi (internet
  erişimi yok); kullanmadan önce tek bir `kosu_id` ile duman testi
  yapılması önerilir. Araç/şema katmanı tamamen yerel test edildi.
- `tests/`: `conftest.py` + 4 yeni test dosyası, toplam 40 test, hepsi
  gerçek proje verisiyle (results.csv, reports/) çalışıyor ve geçiyor.
  `test_egitim_protokolu.py` özellikle önceki maddedeki lr0/warmup_epochs
  hatasının bir daha sessizce geri gelmemesini garanti eden bir sözleşme
  testi içeriyor.
- `requirements-dev.txt` eklendi (`pytest`); `pyproject.toml`'a
  `[tool.pytest.ini_options]` eklendi.
- **Açık iş:** `teshis/servis/*` (API, anomali, arıza, loglama) hâlâ boş
  iskelet; roadmap'te "Aşama 2" olarak işaretli, bu turda kapsanmadı.

### 2026-08-25 (1) — Eğitim protokolü merkezileştirildi, Git kanıt takibi tutarlı hale getirildi

Sorun: `teshis/egitim/kos.py` (D1'de kullanıldı) `lr0=0.0005`,
`warmup_epochs=2` kullanıyordu; `scripts/kaggle_d2a.py`, `kaggle_d2b.py`,
`local_d2b.py`, `local_d3.py` ise `lr0=0.001`, `warmup_epochs=3` kullanıyordu.
Bu, D1 ile D2a/D2b/D3 arasındaki metrik farkının bir kısmının veri
bozulmasından değil, farklı optimizasyon ayarından kaynaklanabileceği
anlamına geliyordu — projenin "senaryolar arasında tek değişken değişir"
temel ilkesini zayıflatan gerçek bir metodoloji açığıydı.

Yapılanlar:

- [senaryolar/egitim_protokolu.yaml](senaryolar/egitim_protokolu.yaml):
  tüm D serisi koşularda paylaşılan sabit optimizasyon/augmentasyon
  değerlerinin tek kaynağı. `bilinen_sapmalar` alanında D1'in tamamlanmış
  koşusunun (`run_20260817_222323_D1_42`) bu protokolden önce, farklı
  `lr0`/`warmup_epochs` ile üretildiği açıkça kayıt altına alındı.
- [teshis/egitim/protokol.py](teshis/egitim/protokol.py): bu YAML'ı okuyan
  `egitim_kwargs()` yardımcı fonksiyonu.
- [teshis/egitim/kos.py](teshis/egitim/kos.py), [scripts/local_d2b.py](scripts/local_d2b.py),
  [scripts/local_d3.py](scripts/local_d3.py), [scripts/kaggle_d2a.py](scripts/kaggle_d2a.py),
  [scripts/kaggle_d2b.py](scripts/kaggle_d2b.py): hepsi artık hiperparametreleri
  kendi içlerinde tekrar tanımlamak yerine `egitim_kwargs()` üzerinden alıyor.
  Kaggle scriptleri repo kökü Kaggle'a yüklendiği için `teshis` paketini
  `sys.path` üzerinden import edebiliyor.
- **Açık iş / kullanıcı kararı gerekiyor:** D1'in mevcut sonucu (README'deki
  D1 tablosu, `results.csv` satırı `d1_20260817_222323`) düzeltilen protokolle
  yeniden üretilmedi. D1'i D2a/D2b/D3 ile kesin biçimde karşılaştırmadan önce
  D1'in bu ortak protokolle yeniden koşulması önerilir (~90 dakika GPU
  süresi). Bu yeniden koşu kullanıcı onayı olmadan başlatılmadı.
- `.gitignore`: `reports/`, `experiments/`, `veri_surumleri/` için tutarsız
  bir politika vardı — bazı kanıt dosyaları (`d2b_sonuc`, `llm_trial`) elle
  force-add edilmişti, diğerleri (`d1_sonuc`, `d2a_sonuc`,
  `model_karsilastirma*`, çoğu veri sürümü manifesti) hiç Git'e girmemişti.
  Artık kural tutarlı: her senaryonun `manifest.json`, `data.yaml`,
  `run_manifest.json`, `args.yaml`, `results.csv` ve rapor JSON/CSV'leri
  otomatik olarak izleniyor; ağır PNG/JPG/HTML çıktılar ve kopyalanan
  dataset görüntü/etiket klasörleri kasıtlı olarak Git dışı kalmaya devam
  ediyor (repo boyutu ve orijinal dataset'e dokunmama kuralı korunuyor).

## 2. Mevcut Durum

Tamamlananlar:

- [x] Proje klasor iskeleti ve PROJECT_STRUCTURE.md dosya sozlesmesi.
- [x] Dataset taramasi: 21.846 goruntu, 154.141 bbox.
- [x] Sinif toplamlar: tasit 95.877, insan 57.802, UAP 268, UAI 194.
- [x] Yetim goruntu/etiket ve gecersiz YOLO satiri kontrolu.
- [x] Kilitli val_diagnostic: 1.056 goruntu, 4.014 bbox.
- [x] val_diagnostic bbox: tasit 1.264, insan 2.718, UAP 15, UAI 17.
- [x] main_model.pt secilen ana model olarak belirlendi.
- [x] D1 veri surumu uretildi.
- [x] GitHub private repository'ye ilk commit push edildi.

Aktif:

- [x] D1 egitimi tamamlandi ve ortak protokolle yeniden kosuldu.
- [x] D1 best.pt val_diagnostic ile degerlendirildi.
- [x] D1 ile saglikli model farki raporlandi.
- [x] D2a lokalizasyon gurultusu kosusu ve diagnostic raporu tamamlandi.
- [x] D2b eksik etiket kosusu iki baslangic modeliyle tamamlandi.
- [x] Gemini 3.6 Flash ile anonim LLM pilotu yapildi.
- [x] Gemini pilotu gizli cevap anahtariyla otomatik puanlandi.
- [x] Streamlit ara sunum demosu kuruldu ve localhost'ta dogrulandi.
- [x] D3 UAP/UAI sinif karisikligi: veri surumu, yerel GPU egitimi ve
  diagnostic degerlendirmesi tamamlandi (sizintisiz config ile yeniden kosuldu).

Sonraki isler:

- [ ] D1 hata galerisi.
- [x] D2a veri senaryosu.
- [x] D2b veri senaryosu.
- [x] D3 veri senaryosu.
- [ ] D4-D6b veri senaryolari.
- [ ] E1-E4 egitim senaryolari.
- [ ] Bootstrap guven araliklari ve ajan karar akisi.
- [ ] Final test: sadece bir kez.

## 3. Degismez Kurallar

1. Orijinal dataset: C:/Users/ASUS/Desktop/HYZ/dataset.
2. Orijinal dataset silinmez, tasinmaz ve etiketleri degistirilmez.
3. Test final asamasina kadar kullanilmaz.
4. val_diagnostic gelistirme sirasinda degistirilmez.
5. Her senaryo tek bir hedef bozulma uygular.
6. Her veri degisikligi manifest.json ile kaydedilir.
7. Her egitim ayri run_YYYYMMDD_HHMMSS_SENARYO_SEED klasorundedir.
8. Sonuclar genel mAP ile birlikte sinif AP, recall, bbox n ve belirsizlik notu
   icermelidir.
9. UAP/UAI icin bbox n kucuk oldugundan kesin genelleme iddiasi kurulmaz.
10. Aktif kosu varken ikinci kosu baslatilmaz.
11. Dosya yapisi degisecekse once README ve PROJECT_STRUCTURE guncellenir.

## 4. Sabit Yollar

Proje: C:/Users/ASUS/Desktop/termal_teshis

Dataset: C:/Users/ASUS/Desktop/HYZ/dataset

Modeller:

- C:/Users/ASUS/Desktop/termal_teshis/main_model.pt
- C:/Users/ASUS/Desktop/termal_teshis/final_best.pt

Tanı seti:

- C:/Users/ASUS/Desktop/termal_teshis/val_diagnostic/data.yaml
- C:/Users/ASUS/Desktop/termal_teshis/val_diagnostic/manifest.json

GitHub: https://github.com/pnarkz/teshis_llm

## 5. Dosya Yapisi

Ana Python paketi:

- teshis/veri: istatistik, manifest, veri surumu ve val uretimi.
- teshis/egitim: egitim kosucusu ve kosu manifestleri.
- teshis/degerlendirme: model, metrik, hata ve kanit analizi.
- teshis/ajan: ajan araclari, semalar ve puanlama.
- teshis/servis: API, log, ariza ve anomali katmani.

Senaryolar:

- senaryolar/katalog.yaml: tek resmi senaryo listesi.
- senaryolar/veri: D1, D2a, D2b, D3, D4, D5, D6a, D6b.
- senaryolar/egitim: E1, E2, E3, E4.

Ciktilar:

- veri_surumleri: manifest ve data.yaml tabanli veri varyantlari.
- experiments: egitim kosulari; Git disi.
- reports: agir metrik ve gorsel raporlar; Git disi.
- val_diagnostic: kilitli tanı val; Git disi.

Dosya adlarinin ayrintili sozlesmesi PROJECT_STRUCTURE.md dosyasindadir.

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

---

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
- [x] reports/model_karsilastirma_fair baseline raporu.
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
Son rapor: reports/d1_v2_sonuc/d1_metrics.json

Ilk kosu (protokol sapmali, tarihsel kayit olarak korunuyor):
experiments/run_20260817_222323_D1_42 · reports/d1_sonuc/d1_metrics.json

### E. Senaryolar

- [x] D2a lokalizasyon gurultusu.
- [x] D2b eksik etiket.
- [x] D3 UAP/UAI class 2-3 karisikligi.
- [ ] D4 kucuk nesne sinyal kaybi.
- [ ] D5 kaynak/alani kaymasi.
- [ ] D6a split sizintisi.
- [ ] D6b tekrar agirligi ve efektif n.
- [ ] E1 overfitting.
- [ ] E2 underfitting.
- [ ] E3 seed bazli kararsizlik.
- [ ] E4 640/768 uyumsuzlugu.

### D2a Sonucu

- [x] Kaggle kosusu: experiments/run_D2a_42.
- [x] Model: experiments/run_D2a_42/weights/best.pt.
- [x] Diagnostic degerlendirme: reports/d2a_sonuc/d2a_metrics.json.
- [x] Hata galerisi: reports/d2a_hata_galerisi/index.html.
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

- reports/d2a_hata_galerisi/index.html
- reports/d2a_hata_galerisi/gallery.json
- reports/d2a_hata_galerisi/images/

### D2b Ilk Sonucu

- [x] Yerel GPU kosusu: experiments/run_D2b_42_local.
- [x] Model: experiments/run_D2b_42_local/weights/best.pt.
- [x] Diagnostic degerlendirme: reports/d2b_sonuc/d1_metrics.json.
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

- reports/d2b_sonuc/d1_metrics.json
- reports/d2b_sonuc/d2b_val_diagnostic/confusion_matrix.png

### D2b Final_best Karsilastirmasi

- [x] Ayni D2b bozuk veri protokolu final_best.pt ile tekrarlandi.
- [x] Model: experiments/run_D2b_42_final_best_local/weights/best.pt.
- [x] Diagnostic degerlendirme: reports/d2b_final_best_sonuc/d1_metrics.json.

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

- reports/d2b_final_best_sonuc/d1_metrics.json
- reports/d2b_final_best_sonuc/d2b_final_best_val_diagnostic/confusion_matrix.png

### D3 Sonucu

- [x] Yerel GPU kosusu: experiments/run_20260826_001456_D3_42 (24 epoch,
  5,6 saat, patience=10 ile erken durdu).
- [x] Model: experiments/run_20260826_001456_D3_42/weights/best.pt.
- [x] Diagnostic degerlendirme: reports/d3_v2_sonuc/d1_metrics.json.
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

- reports/d3_v2_sonuc/d1_metrics.json
- reports/d3_v2_sonuc/d3_v2_val_diagnostic/confusion_matrix.png
- veri_surumleri/v04_d3_uap_uai_sinif_karisikligi/manifest.json

Ilk kosu (egitim val'i olarak kilitli tanı setini kullaniyordu, bu yuzden
iyimser yanliydi; tarihsel kayit olarak korunuyor):
experiments/run_D3_42_local · reports/d3_sonuc/d1_metrics.json
- veri_surumleri/v04_d3_uap_uai_sinif_karisikligi/manifest.json

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
| D3 sinif karisikligi | -0.0281 | -0.0215 | **-0.2047** | -0.0347 | guclu destek |

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

### Ara Sunum Konsolu

Streamlit arayuzu tamamlanan deneyleri egitim calistirmadan gosterir. Baslatma:

```powershell
python -m pip install -r requirements-demo.txt
streamlit run demo/app.py
```

Arayuzde su bolumler bulunur:

- Genel Bakis: her kosunun readout karti, epoch sparkline'lari, karsilastirma
  tablosu, metrik profili ve sinif bazli AP50.
- Senaryo Incele: baseline farki, epoch bazli egitim egrileri, confusion
  matrix, esik egrileri (PR/F1/P/R), etiket-vs-tahmin gorselleri ve bozulmus
  egitim verisi onizlemesi.
- Hata Galerisi: 50 goruntuluk D2a hata galerisi; FN/FP/IoU dagilimi ve
  siralanabilir kareler.
- Proje ve Senaryolar: her senaryonun nasil ve neden olusturuldugu.
- LLM Ajan: Gemini'nin anonim metrikler uzerinden urettigi pilot teshisler ve
  rubrik puanlari.

Konsol hicbir metrigi kendi icinde saklamaz; hepsi `results.csv`, `reports/`
ve `experiments/` altindan okunur. Yeni bir senaryo `results.csv`'ye
eklendiginde egitim egrisi ve readout otomatik gelir. Ayrinti icin
[demo/README.md](demo/README.md).

Demo kaynak kodu `demo/`, bagimliliklari `requirements-demo.txt` altindadir.

LLM pilot puanlamasi `scripts/score_llm_trial.py` ile yapilir. Script, Gemini
cevabini yerelde tutulan gizli `answer_key.json` ile karsilastirir. Puanlama
diagnosis, sayisal evidence ve limitations alanlarina esit agirlik verir. Bu
ilk puan bir benchmark degil, ajan gelistirme icin pilot rubriktir. Ilk Gemini
sonucu ortalama `0.833` olmustur; D1 nedenini acikca adlandiramamasi sonraki
prompt ve kanit tasariminin gelistirme hedefidir. Bunun icin ikinci paket
surumunde her anonim kosuya baseline'a gore toplam ve sinif bazli delta alanlari
eklendi. Gemini v2 bu paketle tekrar calistirildi ve skor yine `0.833` oldu.
Model degisimleri kanit olarak kullandi, ancak D1 ve D2b'nin nedensel adini
dogrudan koyamadi. Bu, LLM'nin metrik okuma becerisi ile nedensel veri teshisi
arasinda ayrim oldugunu gosteren kayitli bir sinirliliktir.

### F. Ajan

- [x] Gemini ile anonim metrik yorumlama pilotu.
- [x] LLM icin anonim girdi ve JSON cikti semasi olusturuldu.
- [x] Izinli araclari tanimla: [teshis/ajan/araclar.py](teshis/ajan/araclar.py)
  (kosu_listesini_getir, kosu_metriklerini_getir, baseline_farkini_getir,
  bbox_sayilarini_getir).
- [x] Pilot JSON girdi/cikti semasi olusturuldu.
- [x] JSON girdi/cikti semalarini production semasi olarak sabitle:
  [teshis/ajan/semalar.py](teshis/ajan/semalar.py)::TESHIS_SEMASI +
  ARAC_BILDIRIMLERI (Gemini function-calling formatinda).
- [x] Manifesti ajandan gizle: `araclar.py` hicbir fonksiyonda
  veri surumu manifestini okumaz; yalnizca results.csv ve reports/ metrik
  JSON'larini, anonim kosu_NN kimlikleriyle sunar.
- [x] Kanit, guven ve sinir alanlarini zorunlu kil:
  `semalar.teshis_dogrula` eksik alan, gecersiz confidence degeri, tek
  kanitli veya sayisal olmayan evidence, liste olmayan limitations
  durumlarini programatik olarak reddeder (bkz. tests/test_ajan_semalar.py).
- [~] yetersiz_kanit kararini destekle: sema ve prompt bunu bir secenek
  olarak taniyor; modelin bunu ne zaman secmesi gerektigi canli bir LLM
  kosusuyla henuz gozlemlenmedi.
- [x] Anonim senaryo adlarini ve puanlama cetvelini test et:
  tests/test_ajan_araclar.py, tests/test_ajan_puanlama.py.
- [ ] `teshis/ajan/ajan.py::teshis_uret` (gercek Gemini function-calling
  dongusu) canli bir `GEMINI_API_KEY` ile uctan uca dogrulanmadi — bu kod
  tabaninda internet erisimi yok. Calistirmadan once kucuk bir kosu ile
  (tek `kosu_id`) duman testi yapilmasi onerilir.

### G. Final

- [ ] Model ve parametreleri dondur.
- [ ] Testi sadece bir kez calistir.
- [ ] Test sonucunu commit/model/veri manifesti ile sakla.
- [ ] Mentor raporunu tamamla.
- [x] GitHub main branch'ini guncelle.

## Deney Degismezleri

Her deney senaryo kodu, config, veri surumu, manifest, model, Git commit'i,
seed, imgsz, batch, epoch, device, split sayilari, bbox sayilari, precision,
recall, mAP50, mAP50-95, sinif AP/recall, UAP/UAI bbox n, belirsizlik,
confusion matrix ve hata goruntulerini kaydetmelidir.

Test final asamasina kadar split=test kullanilamaz.

## Baska AI Ajana Okuma Sirasi

1. README.md.
2. PROJECT_STRUCTURE.md.
3. config.yaml.
4. senaryolar/katalog.yaml.
5. reports/veri_raporu.json.
6. reports/model_karsilastirma_fair/model_karsilastirma.json.
7. val_diagnostic/manifest.json.
8. Aktif deneyin run_manifest.json dosyasi.

Ajan bu sirayi okumadan yeni senaryo veya veri bolmesi baslatmamalidir. Aktif
kosu varsa once surec ve log kontrol edilir. Mevcut dosyalar silinmez; yeni
deney ayri klasore yazilir. Her degisiklik sonrasi syntax, Git status ve tekrar
calistirma komutu raporlanir.

## 13. Teknik Olmayan Okuyucu Icin Proje Ozeti

### Bu sistem ne yapiyor?

Sistem bir termal drone fotografina bakiyor ve fotograf icindeki nesneleri
bulmaya calisiyor. Sadece fotografi "arac var" diye siniflandirmiyor; her
nesnenin fotograf icindeki yerini bir dikdortgen kutu ile isaretliyor ve bu
kutunun sinifini yaziyor.

Ornek:

- Bir arac bulunursa: sinif tasit, kutu koordinatlari.
- Bir insan bulunursa: sinif insan, kutu koordinatlari.
- Bir hedef bulunursa: sinif UAP veya UAI, kutu koordinatlari.

Bu nedenle bu proje basit bir goruntu siniflandirma projesi degildir. Modelin
iki isi vardir:

1. Nesnenin nerede oldugunu bulmak.
2. Buldugu nesnenin hangi sinifa ait oldugunu soylemek.

### Bbox nedir?

Bbox, nesneyi cevreleyen bounding box yani dikdortgen kutudur. Dataset'teki
her etiket satiri bir bbox'i temsil eder. "4.014 bbox" dedigimizde 4.014
fotograf degil, fotograflardaki toplam 4.014 etiketli nesne kastedilir.

Bir fotograf birden fazla bbox icerebilir. Bu nedenle goruntu sayisi ile bbox
sayisi her zaman ayni degildir.

### Dataset neden train, val ve test diye ayrilir?

- Train: Modelin ogrendigi fotograflar.
- Val: Egitim sirasinda model ayarlarini ve en iyi agirliklari secmek icin
  kullanilan kontrol grubu.
- Test: Tum secimler bittikten sonra gercek performansi olcmek icin saklanan
  son sinav grubu.

Val model gelistirme sirasinda kullanilabilir. Test kullanilirsa ekip modelin
ayarlarini test sonucuna gore farkinda olmadan degistirebilir. Bu durumda test
artik bagimsiz bir sinav olmaktan cikar. Bu nedenle bu projede test final
asamasina kadar yasaktir.

### val_diagnostic neden var?

Mevcut val klasorunde ayni kaynaktan gelen benzer veya augment edilmis
goruntuler bulunabilir. val_diagnostic bu durumu azaltmak, model gelistirme
sirasinda daha tutarli bir kontrol noktasi olusturmak icin hazirlandi.

Bu set:

- Model gelistirme icin kullanilir.
- Test seti degildir.
- Olusturulduktan sonra degistirilmez.
- Her model ayni set uzerinde karsilastirilir.

### Metrikler nasil okunur?

Precision, modelin "bu nesne var" dedigi tahminlerin ne kadarinin dogru
oldugunu anlatir. Yuksek precision, modelin gereksiz kutu cizme ihtimalinin
daha dusuk oldugunu gosterir.

Recall, gercekte var olan nesnelerin ne kadarinin yakalandigini anlatir.
Yuksek recall, modelin nesne kacirma ihtimalinin daha dusuk oldugunu gosterir.

mAP, hem sinif dogrulugunu hem de kutunun gercek nesneyle ne kadar ortustugunu
birlikte olcen ozet metriktir. mAP tek basina yeterli degildir; bu projede
her zaman sinif bazli AP, recall ve bbox sayisi ile birlikte raporlanir.

Ornegin UAP mAP50 degeri 0.995 olsa bile bu deger sadece 15 bbox'a dayaniyorsa
sonucun belirsizligi yuksektir. Bu nedenle "UAP cok iyi" demeden once kac
ornekle olculdugune bakilir.

## 14. Bu Proje Neden Senaryolar Kullaniyor?

Modelin neden basarili veya basarisiz oldugunu normal bir egitim sonucundan
anlamak zordur. D1-D6 ve E1-E4 senaryolari kontrollu deneylerdir. Her deneyde
tek bir problem kasitli olarak olusturulur.

Ornekler:

- D1'de insan iceren egitim fotograflarinin cogu cikartilir. Amac, insan
  sinifi az olursa modelin ne kadar bozuldugunu olcmektir.
- D2b'de bazi etiketler eksik birakilir. Amac, etiketleme hatasinin etkisini
  olcmektir.
- D3'te UAP ve UAI etiketleri karistirilir. Amac, sinif ID hatasinin etkisini
  olcmektir.
- D5'te farkli kaynaklardan gelen goruntuler arasindaki alan farki incelenir.

Her senaryoda saglikli referans ile bozuk veri kosusu karsilastirilir. Boylece
"model kotu oldu" yerine "bu spesifik veri problemi recall'i su kadar
dusurdu" gibi olculebilir bir sonuc elde edilir.

## 15. D1'i Basit Dille Anlatim

D1 su soruya cevap arar:

> Egitim dataset'inde insan fotograflarinin buyuk bolumu olmasaydi model
> insanlarin yerini ve varligini yine ogrenebilir miydi?

D1'de:

- Orijinal dataset kopyalanmaz ve degistirilmez.
- Insan iceren train karelerinin yuzde 90'i egitim listesinden cikartilir.
- Val ve test etiketlerine dokunulmaz.
- Ayni main_model.pt ile egitim baslatilir.
- D1 modeli val_diagnostic uzerinde tekrar olculur.
- Insan recall ve insan AP degerlerinin baseline'a gore nasil degistigi
  incelenir.

Bu deney modelin ne kadar iyi oldugunu degil, belirli bir veri eksikligine ne
kadar dayanikli oldugunu gosterir.

## 16. Ajanin Rolu

Ajan modelin yerine gecmez ve kendi basina yeni gercekler uretmemelidir.
Ajanin gorevi:

1. Deney manifestini ve izin verilen raporlari okumak.
2. Saglikli ve bozuk kosunun metriklerini karsilastirmak.
3. Hangi sinifin etkilendigini belirlemek.
4. Hatanin veri mi, etiket mi, egitim mi yoksa alan farki mi olabilecegini
   siniflandirmak.
5. Kararini dosya yolu ve sayisal kanit ile aciklamak.
6. Veri yetersizse yetersiz_kanit demek.

Ornek ajan sonucu:

~~~text
Karar: sinif_yetersizligi
Kanit: D1 manifesti, insan bbox sayisi ve insan recall farki
Guven: orta
Sinir: UAP/UAI bbox sayisi dusuk; bu senaryo onlar hakkinda kanit saglamiyor
Sonraki olcum: D2b eksik etiket senaryosu
~~~

## 17. Mentor Sunumu Icin Kisa Anlatim

Bu proje, termal drone nesne tespit modelinin kontrollu kosullarda neden
bozuldugunu arastiran bir teshis altyapisidir. Once dataset sagligi ve model
baseline'i olculur. Sonra tek bir veri veya egitim problemi kasitli olarak
olusturulur. Ayni model bu bozuk kosulda tekrar egitilir ve kilitli bir
diagnostic validation setinde olculur. Sonuclar sinif bazli metrikler, bbox
sayilari, confusion matrix ve hata goruntuleriyle raporlanir. Son asamada bir
LLM/ajan bu kanitlari kullanarak arizanin nedenini ve guven seviyesini aciklar.

Bu yaklasimin ana farki, sadece daha yuksek mAP aramak yerine modelin hangi
kosullarda guvenilirligini kaybettigini olcmeye calismasidir.

## Sonraki Tek Adim

1. D1 hata galerisi uret ve insan kacirma orneklerini incele.
2. Hata galerisi sonucunu rapora ekle.
3. D1 checklist kutularini tamamla.
4. D2a'ya gec.

## Kaggle D2a Calistirma

Kaggle Notebook ayarlarinda GPU acilmali ve iki input eklenmelidir:

1. Dataset: images/train, labels/train ve diger splitleri iceren dataset.
2. Model: main_model.pt.

Sonra notebook'ta asagidaki komutla script calistirilir:

~~~text
!python /kaggle/working/teshis_llm/scripts/kaggle_d2a.py
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

Notebook'a scripts/kaggle_d2a.py ve scripts/kaggle_d2b.py dosyalarini birlikte
yukleyin. D2b scripti D2a icindeki ortak input bulma ve dosya linkleme
fonksiyonlarini kullanir.

~~~text
!python /kaggle/working/teshis_llm/scripts/kaggle_d2b.py
~~~

D2b ciktilari:

~~~text
/kaggle/working/v03_d2b_eksik_etiket/manifest.json
/kaggle/working/v03_d2b_eksik_etiket/data.yaml
/kaggle/working/experiments/run_D2b_42/weights/best.pt
~~~

Beklenen kanit: Eksik etiketler nedeniyle modelin recall'i ve ozellikle false
negative sayisi artabilir. Config beklentisi precision dususudur; bu beklenti
ayrica diagnostic val metrikleri ve hata galerisi ile kontrol edilmelidir.

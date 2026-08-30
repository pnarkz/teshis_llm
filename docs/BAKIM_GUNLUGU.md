# Bakim Gunlugu

Projede yapilan duzenleme, duzeltme ve olcum islerinin kronolojik kaydi.
En yeni kayit en ustte. Her muhendislik degisikliginden sonra buraya yeni bir
madde eklenir; boylece hangi sorunun ne zaman ve nasil giderildigi izlenebilir.

Bu bölüm, projede yapılan düzenleme/temizlik/eksik-giderme işlerinin
kronolojik kaydıdır. Her mühendislik değişikliğinden sonra buraya yeni bir
madde eklenir; böylece hangi sorunun ne zaman ve nasıl giderildiği README
üzerinden takip edilebilir. En yeni kayıt en üstte durur.

### 2026-08-30 — E serisi baslatildi: protokol sapmalari beyan edilir hale getirildi

**Sorun.** E senaryolari (egitim arizalari) ortak egitim protokolunu KASITLI
olarak bozmak zorundadir; D serisi ise protokolun sabit kalmasina dayanir.
Sapmalari CLI bayraklariyla vermek, hangi kosunun protokolden nerede
ayrildigini izlenemez hale getirirdi — projenin tek kaynak ilkesini bozan
tam da bu tur sessiz kaymalar olmustu (bkz. D1 protokol sapmasi).

**Cozum.** Sapmalar `senaryolar/egitim_protokolu.yaml` icinde `e_serisi`
blokunda beyan edilir. `teshis/egitim/protokol.py` iki fonksiyon ekler:
`e_senaryo_ayarlari()` ve `egitim_kwargs_e()`. Ikincisi, protokolde
**var olmayan** bir alana sapma tanimlanmasini reddeder — aksi halde E
kosusu sessizce protokolu genisletir ve D kosulariyla karsilastirilamaz
hale gelirdi. `kos.py --e-senaryo <kod>` sapmalari uygular, baslangicta
ekrana basar ve kosu manifestine `protokol_sapmalari` alani olarak yazar.

21 test (`tests/test_e_serisi_protokol.py`) bu yapiyi korur; ozellikle
"sapma protokolle ayni degeri tasiyamaz" ve "E sapmasi D protokolunu
kirletemez" kontrolleri.

**E4 tamamlandi.** E4, egitim gerektirmeyen tek E senaryosudur. Konfigdeki
tek cift (640/1280) yerine protokol-uyumlu v00 modeli (imgsz=768) bes
noktada olculdu; gerekce `scripts/senaryo_E4_cozunurluk_uyumsuzlugu.py`
docstring'inde. Bulgular:

- mAP50 tepesi tam egitim cozunurlugunde (0.920) — taramanin ic kontrolu.
- Kucultmek buyutmekten cok daha pahali: 512'de -0.318, 1280'de -0.042.
- Precision cozunurlukten neredeyse etkilenmiyor (0.858-0.918), recall
  cokuyor (0.518-0.879). Uyumsuzluk modeli **yaniltmiyor, kor ediyor** —
  D serisi etiket bozulmalarindan ayirt edici imza budur.
- Kayip kucuk siniflarda yogunlasiyor; Wilson %95 araliklari UAP, UAI ve
  tasit icin ortusmuyor (`insan` icin ortusuyor, pratik olarak kucuk).

**Yakalanan iki hata.**

1. `d1_sonuc.py` val ciktisini `<senaryo>_val_diagnostic` olarak
   adlandiriyordu; mevcut klasorler daha once elle `gorseller/` yapilmis
   ama uretici hic duzeltilmemisti. `test_belge_yollari.py` bunu yakaladi;
   uretici duzeltildi ve yeni E4 klasorleri tasindi.
2. Belgeye UAP recall@512 icin 0.241 yazildi; dogrusu 0.242. Neden: ozet
   JSON 4 haneye yuvarliyor (0.2415) ve bunu tekrar 3 haneye bicimlemek
   *cift yuvarlama* ile 0.241 veriyor. Dogrulama artik ham
   `d1_metrics.json` dosyalarina karsi yapiliyor
   (`tests/test_e4_belge_tutarliligi.py`).

**Sirada:** E2 kosuyor (5 epoch). Ardindan E1 ve E3 (iki seed).

### 2026-08-28 (3) — Proje mimari olarak yeniden duzenlendi

Proje buyudukce dosya duzeni ve README takip edilemez hale gelmisti. Yapisal
bir temizlik yapildi; hicbir olcum veya sonuc degistirilmedi.

**README bolundu.** 2207 satirlik tek dosya, konu bazli belgelere ayrildi.
Numaralandirma 1-5'ten 13-17'ye atliyordu ve 835 satirlik bakim gunlugu tam
ortada duruyordu. Yeni yapi:

| Belge | Icerik |
|---|---|
| `README.md` (~110 satir) | Giris kapisi: amac, yontem ozeti, senaryo tablosu, belge haritasi |
| `docs/BULGULAR.md` | Tum senaryo sonuclari |
| `docs/MIMARI.md` | Dosya sozlesmesi ve adlandirma kurallari |
| `docs/KURALLAR.md` | Degismez kurallar ve sabit yollar |
| `docs/CALISTIRMA.md` | Kurulum ve komutlar |
| `docs/BAKIM_GUNLUGU.md` | Bu dosya |
| `docs/SUNUM.md` | Teknik olmayan anlatim |

**reports/ adlandirmasi tek kurala baglandi:** onek = ne tur, sonek = hangisi.
Once `d1_sonuc`, `d1_v2_sonuc` ve `D1_last_sonuc` yan yana duruyordu ve
hangisinin guncel oldugu belli degildi. Simdi `senaryo_D1` (guncel),
`eski_D1_protokol_sapmali` (superseded) ve `senaryo_D4_last_pt` gibi adlar
kendini anlatiyor. Her rapor klasorunde val ciktisi artik tek tip `gorseller/`
adini tasiyor.

**scripts/ adlari ne yaptiklarini soyluyor:** `local_d3.py` aslinda hem D3
hem D3b uretiyordu ama adindan anlasilmiyordu; artik
`senaryo_D3_D3b_sinif_karisikligi.py`. LLM scriptleri `ajan_` onekiyle
gruplandi.

**Iki gercek hata bulundu ve duzeltildi:**

1. `reports/boyut_analizi/` ile `reports/kirilim/` ayni olcumleri tutuyordu
   (kirilim'de fazladan karisiklik matrisi de vardi). Cift kayit kaldirildi,
   yalnizca benzersiz olan `D5_last` tasindi.
2. **Demo sabit kodlu bir yol haritasi tutuyordu ve geride kalmisti.** D6a,
   D6b, v00n ve D1n eklendiginde demo bunlari sessizce "kanit yok"
   gosteriyordu. Harita kaldirildi; rapor klasoru artik senaryo adindan
   konvansiyonla turetiliyor. Simdi 14 kosunun tamami kanit gosteriyor.

**Yeni testler (14):** belgelerde adi gecen her yolun var oldugu, README
belge haritasinin eksiksizligi, rapor adlandirma kuralinin korundugu, demo'nun
sabit harita tutmadigi ve her kosunun kanitinin bulunabildigi. Boylece bu tur
kaymalar sessiz kalmaz. Toplam 233 test.

Kaldirilanlar: `PROJECT_STRUCTURE.md` (yerine `docs/MIMARI.md`),
`proje-brifingi-v2.2.md` kok dizinden `docs/` altina tasindi. Kok dizin 24
girdiden 20'ye indi.

### 2026-08-28 (2) — D1 doğru kurguda doğrulandı; iki yeni koruma eklendi

**D1 hipotezi, dataset'i hiç görmemiş bir başlangıç modelinden koşulduğunda
güçlü biçimde doğrulandı.** Aynı bozulma, aynı protokol, aynı ölçüm seti;
tek fark başlangıç modeli:

| | main_model kurgusu | yolo26n kurgusu |
|---|---:|---:|
| insan recall farkı | -0.0147 | **-0.2475** |
| z | -1,22 (anlamsız) | **-18,58** |
| insan AP50 farkı | +0.0260 | **-0.1721** |

D1'in ilk başarısızlığı bozulmanın etkisiz olmasından değil, **deney
kurgusunun onu ölçememesinden** kaynaklanıyormuş — 2026-08-26 (3) kaydındaki
teşhis doğrulandı. Zaten veriyi öğrenmiş bir modeli kısa fine-tune ile
unutturamazsınız.

İki koşu birebir eşitlendi: aynı başlangıç modeli, aynı 30-epoch LR programı
ve **23 tamamlanmış epoch** (v00n bir kesinti nedeniyle 23'te kaldığı için
D1n de 23'te durduruldu).

**Yeni koruma 1 — farklı taban modelden gelen koşular ajana verilmez.**
`AJANA_VERILMEYEN` sözlüğü eklendi. Ajan her koşuyu kosu_01 (main_model
tabanlı v00) ile karşılaştırır; farklı bir taban modelden gelen koşuda fark
bozulmadan değil model kapasitesinden gelir ve ajan bunu "bozulma" sanardı.
Her dışarıda bırakma yazılı gerekçe taşıyor, test bunu zorunlu kılıyor.
Kayıtlar `results.csv`'de duruyor, sadece ajana sunulmuyor.

**Yeni koruma 2 — `--devam` artık onay istiyor.** Bir önceki kayıtta eklediğim
özellik v00n koşusunda ölçülebilir şekilde eğitimi bozdu (cls_loss 0.7477 →
3.1210, mAP50 0.8136 → 0.2849). Ölçülen kanıt docstring'de somut sayılarla
duruyor; bilimsel karşılaştırmaya girecek koşularda kullanılamaz.

**`bootstrap.py` yazıldı.** README'deki tüm z değerleri ve güven aralıkları
tek seferlik scriptlerde hesaplanmıştı, repoda karşılığı yoktu — kimse
doğrulayamaz veya yeniden üretemezdi. Modül Wilson aralığı, iki oran testi ve
**görüntü birimli bootstrap** sağlıyor (aynı görüntüdeki kutular bağımsız
değildir; kutu birimli testler aralığı olduğundan dar gösterir). Doğrulama
sırasında bir tutarsızlık çıktı: UAI için ad-hoc script z=-3,74 veriyordu,
modül -3,59. İki oran testi sayımlar üzerinde tanımlı olduğu için modülünki
tutarlı; README düzeltildi, gerekçe yazıldı, sonuç değişmedi (p<0,05).

### 2026-08-28 — D6a ve D6b tamamlandı; D serisi bitti, yolo26n karşılaştırması başladı

**D6a (split sızıntısı)** tamamlandı ve önemli bir tasarım bulgusu verdi:
**bu senaryo yeniden eğitim gerektirmiyor.** Sızıntı modeli değil ölçümü
bozar, o yüzden mevcut v00 modeli iki kümede ölçüldü. Daha önce "kural 3 ile
çatışıyor" diye çekindiğim senaryo, kilitli tanı seti temiz taban olarak
kullanılınca test setine hiç dokunmadan yapılabildi.

%20 sızıntı mAP50-95'i **+0.0287 (+%4,3)** şişiriyor, mAP50'yi sadece
+0.0085. En çok mAP50-95'in şişmesi mekanizmayla tutarlı: model sızan
karelerin kutu konumlarını ezberlemiş, sıkı IoU eşikleri tam da onu ölçüyor.
Asıl tehlike, bu şişmenin projedeki birçok **gerçek** bozulmanın etkisinden
büyük olması (D4 -0.0006, D3b +0.0042, D5 +0.0044) — yani sızıntı bir
bozulmayı tamamen maskeleyebilir.

**D6b (tekrar ağırlığı)** tamamlandı ve projenin en net **doz-yanıt**
ilişkisini gösterdi. 175 kare 40× tekrarlanarak eğitimin %28,8'ini kapladı;
o karelerde UAI hiç yok (bbox payı %0,00). `last.pt`'de sonuç temsil payını
tam olarak izliyor:

| Sınıf | eğitimdeki pay | recall farkı | z |
|---|---:|---:|---:|
| tasit | %22,80 | +0.0356 | +2,52 * |
| insan | %10,39 | +0.0224 | +1,91 |
| UAP | %0,06 | 0.0000 | 0,00 |
| **UAI** | **%0,00** | **-0.6123** | **-3,59 *** |

Hiç temsil edilmeyen sınıf çöküyor (recall 0.9412 → 0.3289). Dikkat çekici
olan: bu bozulma veriye **hiçbir hata eklemiyor** — hiçbir etiket yanlış
değil, hiçbir kutu silinmedi. Yalnızca bazı kareler daha sık gösterildi.
Veri kalitesinin sadece "etiketler doğru mu" olmadığını, **dağılımın
kendisinin** bir kalite boyutu olduğunu gösteriyor.

**Yeni koruma — farklı ölçüm kümesi karışması.** D6a farklı bir
değerlendirme kümesinde ölçüldüğü için ajan katmanına yapısal engel eklendi:
`anonim_kosu_haritasi` artık yalnızca `KILITLI_DEGERLENDIRME_SETI` üzerinde
ölçülmüş koşuları sunuyor. Aksi halde ajan farklı tabanları kıyaslar ve
sızıntının şişmesini "bozulma" sanardı. Kayıt `results.csv`'de duruyor,
sadece ajana verilmiyor.

**Ayrıca:** `labels.cache` bir commit'e sızmıştı (Ultralytics'in ürettiği
makineye özgü önbellek); git takibinden çıkarıldı ve `.gitignore`'a genel
`*.cache` kuralı eklendi — mevcut kural yalnızca `experiments/` altını
kapsıyordu.

**Başlatıldı: yolo26n karşılaştırması.** D1'in mevcut kurguda başarısız
olmasının nedeni, `main_model.pt`'nin zaten tüm veriyi görmüş olmasıydı.
Genel amaçlı `yolo26n.pt`'den bir referans (`v00n`) ve bir D1 koşusu
başlatıldı (~6,4 saat/koşu). **Önemli sınır:** `yolo26n.pt` 2,57M
parametreli nano bir model; `main_model.pt` çok daha büyük. Bu çift kendi
içinde geçerli bir kontrollü karşılaştırmadır ("veriyi hiç görmemiş bir
model sınıf yetersizliğinden etkilenir mi?") ancak sayıları mevcut
main_model serisiyle **doğrudan kıyaslanamaz.**

### 2026-08-26 (10) — Function-calling ajanı çalıştırıldı; ilk gözlem alındı, üç hata düzeltildi

Ajan ilk kez canlı API ile koşuldu. **Deneme tamamlanamadı** (günlük API kotası
bitti) ancak ilk beş koşu başarıyla çalıştı ve projenin merkezi sorusuna dair
ilk gözlemi verdi.

**Gözlem — ajan doğru kanıta kendiliğinden yöneliyor.** Konsol çıktısındaki
araç kayıtları:

| Koşu | Araç çağrısı | Kırılım araçları |
|---|---:|---|
| kosu_01 | 6 | üçü de çağrıldı |
| kosu_02 | 7 | üçü de çağrıldı |
| kosu_03 | 7 | üçü de çağrıldı |
| kosu_04 | 6 | üçü de çağrıldı |
| kosu_05 | 7 | üçü de çağrıldı |

Ajan her koşuda `boyut_bazli_recall_getir`, `kaynak_bazli_recall_getir` ve
`sinif_karisikligini_getir` araçlarının **hepsini** çağırdı. Bu, tek atışlık
denemeden farklı bir yetenek: orada kanıt önüne konuyordu, burada kendisi
istedi. Ancak seçici değil **kapsamlı** davranıyor — her koşuda hepsini
çağırıyor; "hangi kırılıma bakmalı" sorusunu eleyerek değil, hepsini alarak
çözüyor. Bu, koşu başına ~7 API isteği demek ve kota tüketiminin ana sebebi.

> **Not:** Bu gözlem konsol çıktısından alınmıştır; beş koşunun teşhis
> **verileri kaydedilemedi** (aşağıdaki 1 numaralı hata). Sayısal sonuç,
> deneme yarın tamamlandığında raporlanacaktır.

**Üç hata bulundu ve düzeltildi:**

1. **Kısmi sonuç kaybı (en kritik).** Sonuçlar yalnızca döngünün sonunda
   diske yazılıyordu. Günlük kotanın büyük kısmını harcamış beş başarılı
   koşu, çalışma ortada kesilince tamamen kayboldu. Artık **her koşudan
   sonra** kaydediliyor.
2. **`Infinity` → geçersiz JSON.** Boyut bandı tanımındaki üst sınır
   `float("inf")` idi ve JSON'a `Infinity` olarak yazılıyordu; bu RFC 8259'a
   göre geçersizdir ve Gemini gövdeyi `400 INVALID_ARGUMENT` ile reddediyordu.
   İlk denemede dokuz koşunun tamamı bu yüzden düştü. Tek atışlık deneme
   çalışıyordu çünkü paket orada prompt **metnine** gömülüyor, JSON gövde
   alanı olarak gönderilmiyordu — hata ancak function-calling'de ortaya çıktı.
   Üç katmanda düzeltildi: kaynak (`metrikler.bant_araliklari()` okunabilir
   metin aralığı üretiyor), ajan (`json_guvenli()` tüm araç çıktılarını
   temizliyor) ve diskteki 13 analiz dosyası (yeniden çıkarım yapılmadan;
   bant tanımı ölçüm değil sabittir).
3. **Günlük/dakikalık kota ayrımı yoktu.** Ücretsiz katmanda iki sınır var:
   dakikada 5 ve **günde 20** istek. Retry mantığı ikisini ayırt etmiyordu;
   günlük kota bittiğinde de "60 sn bekle, tekrar dene" döngüsüne giriyordu.
   Artık `GunlukKotaBitti` ile ayırt ediliyor: beklemeden durur, tamamlananı
   kaydeder ve devam komutunu yazdırır.

**`--devam` bayrağı eklendi.** Koşu başına ~7 istek ve günde 20 istek
sınırıyla dokuz koşuluk deneme tek güne sığmıyor; birden fazla güne
yayılabilmesi gerekiyor. Tamamlanmış koşular atlanır, başarısızlar yeniden
denenir.

**Kalan iş:** kota yenilendiğinde `python -m teshis.ajan.ajan --devam` ile
deneme tamamlanacak, ardından iki deneme (tek atışlık vs ajan) aynı rubrikle
karşılaştırılacak. İlgi çekici soru: kanıtı kendisi isteyen ajan, kanıt
önüne konan modelden daha iyi teşhis koyabiliyor mu?

### 2026-08-26 (9) — İlk adil LLM denemesi koşuldu; rubriğin üç kusuru bulundu

Dokuz koşuluk paket Gemini ile koşuldu. Cevabın tamamı şemaya uygun geldi.
**Ham sonuç: `mean_score` 0.630** — değiştirilmemiş haliyle
`reports/ajan_denemesi/ham_kayit/` altında saklanıyor, çünkü ardından rubrikte
düzeltmeler yapıldı ve "puan yükseltmek için kural değiştirildi" şüphesine
karşı ilk kayıt sabitlenmeli.

**Model üç senaryoyu tam isabetle teşhis etti:**

| Koşu | Gizli senaryo | Modelin teşhisi |
|---|---|---|
| kosu_01 | sağlıklı referans | `referans_kosusu_saglikli` |
| kosu_06 | D3 (UAP/UAI karışıklığı) | `uai_uap_sinif_karisikligi_ve_ciddi_hassasiyet_kaybi` |
| kosu_08 | D4 (küçük nesne kaybı) | `kucuk_nesne_ve_kaynak_a_kritik_tespit_kaybi` |

**D3 ve D4, kırılım araçları eklenmeden önce teşhis edilemezdi** — ikisi de
toplam metriklerde görünmüyor. Araç katmanı yatırımı doğrudan karşılığını
verdi.

**Sonra rubrikte üç gerçek kusur çıktı:**

1. **Cevap anahtarı "uygulanan bozulmayı" kodluyordu, "kanıtta görüneni"
   değil.** D1 ve D3b için bu ikisi farklı: kendi istatistiksel analizimiz
   D1'in anlamlı etkisi olmadığını (z=-1.22) ve D3b'nin bozulmasının
   soğurulduğunu (çapraz hata 2→4) göstermişti. Model kosu_02 için
   "performans stabil, hafif iyileşme" dedi ve **doğru sayıları** gösterdi
   (mAP50-95 +0.024, insan AP50 +0.026) — yani doğru cevap verdiği için
   cezalandırıldı.
2. **Kısmi puan kuralı yalnızca İngilizce terim arıyordu.** Model Türkçe
   cevaplıyor; D2b için "hassasiyet kaybı" dedi, kural `precision` aradığı
   için kaçırdı.
3. **Sınırlama kuralı yalnızca UAP/UAI + (15|17) kalıbını kabul ediyordu.**
   kosu_04'ün "kaynak_d bbox sayısı 106'dır" gibi kusursuz bir küçük-örnek
   uyarısı 0 alıyordu.

**Yapılan düzeltmeler:**

- Sınırlama kuralı artık "bir grup adlandır + yanında sayı ver" istiyor.
  Kalibrasyon testli: "Az örnek var", "Veri sınırlı", hatta "UAP ve UAI
  sınıflarında örnek azdır" (sayısız) hâlâ 0 alıyor.
- Kısmi puan kuralı Türkçe karşılıkları da kabul ediyor.
- `TESPIT_EDILEMEYEN` tablosu eklendi: bozulmanın kilitli tanı setinde
  anlamlı iz bırakmadığı koşullar, projenin **kendi yayımlanmış istatistiksel
  analizine** dayanarak listeleniyor. O koşullarda "anlamlı değişim yok"
  cevabı da doğru sayılıyor.
- **İki skor birlikte raporlanıyor**, hangisinin kullanılacağı okuyucunun
  kararı: `mean_score` (katı — uygulanan bozulmanın adı) ve
  `mean_score_tespit` (tespit-farkındalıklı).

**Sonuçlar:**

| Rubrik | mean_score |
|---|---:|
| İlk hali (üç kusurlu) | 0.630 |
| Kusurlar düzeltilmiş, katı | **0.815** |
| Tespit-farkındalıklı | **0.852** |

0.630 → 0.815 artışının büyük kısmı sınırlama kuralının düzeltilmesinden
geliyor; modelin dokuz sınırlamasının tamamı gerçekten grup adı ve sayı
içeriyordu.

**Kalan gerçek eksikler** (rubrik düzeltmesiyle kapanmayanlar): model D2a'yı
(lokalizasyon gürültüsü) ve D2b'yi (eksik etiket) **nedeniyle**
adlandıramadı; belirtiyi doğru tarif etti ama nedene ulaşamadı. D2b için
mAP50-95'in mAP50'den daha çok düşmesi gibi ayırt edici imzalar pakette var
ancak model bunları nedene bağlamadı. Bu, ajan katmanının bir sonraki
geliştirme alanıdır.

**Metodolojik not:** bu deneme tek atışlık (tüm kanıt önceden veriliyor).
`teshis/ajan/ajan.py` içindeki function-calling sürümünde soru daha zor
olacak: ajan **hangi kırılıma bakması gerektiğini kendisi seçmek** zorunda.
İki ölçüm birlikte okunmalıdır.

### 2026-08-26 (8) — LLM deneme paketi yenilendi; ajanın karşılaştırma tabanı v00'a çevrildi

Paket dokuz koşuya genişletildi ve her koşu için **kırılım kanıtları** (boyut
bandı, kaynak grubu, sınıf karışıklığı) eklendi. Bunlar olmadan ajan D3b, D4
ve D5'i teşhis edemezdi.

**Ayrıca bir tutarsızlık düzeltildi.** Paket ve araçlar, farkları hâlâ eski
baseline'a (fine-tune edilmemiş `main_model.pt`) göre veriyordu — oysa
2026-08-26 (3) kaydında bu tabanın yanıltıcı olduğunu, her farkı
`(bozulma etkisi) + (fine-tune etkisi)` toplamı yaptığını tespit etmiştik.
Ajan yanlış tabana göre karşılaştırma yapıyordu; örneğin D1'in precision
farkını **+0.0186** (yanıltıcı) görüyordu, doğrusu **-0.0015**.

- `araclar.baseline_metriklerini_getir()` artık v00 koşusunu döndürüyor.
- v00, ajanın teşhis edeceği senaryolar listesinden çıkarıldı: `kosu_01`
  olarak karşılaştırma tabanı rolünü üstleniyor. Böylece hem taban doğru hem
  de aynı koşu iki kez sunulmuyor.
- `kosu_NN` numaraları yalnızca v00 sonrası koşularda kaydı; arşivlenen 4
  koşuluk denemenin kullandığı `kosu_01`–`kosu_04` eşlemesi değişmedi.
- Kırılım araçları artık `kosu_01` için de çalışıyor (farklar sıfır çıkar).

**Arşivleme:** kırılım araçları eklenmeden önce yapılan 4 koşuluk Gemini
denemesi (`mean_score` 0.833) `reports/ajan_denemesi_arsiv_4kosu/` altına
taşındı. O sonuç güncel paketle karşılaştırılamaz: paket artık dokuz koşu
içeriyor ve kırılım kanıtları sunuyor.

**Paketin durumu:** 9 koşu, 33 KB, sızıntı taraması temiz (kaynak adları
`kaynak_a`/`kaynak_b` olarak anonim), her koşuda kırılım kanıtı mevcut.
Sızıntı testi artık yalnızca `runs` bölümünü tarıyor — paketin görev tanımı
("Termal drone YOLO diagnostigi") alanın kendisini anlattığı için `termal`
kelimesi orada meşru olarak geçiyor.

**Sıradaki adım kullanıcıda:** `GEMINI_API_KEY` ortam değişkeni ayarlanıp
`python scripts/ajan_tek_atislik_calistir.py` çalıştırıldığında deneme yenilenir,
ardından `python scripts/ajan_puanla.py` puanlar. Bu, projenin merkezi
sorusunun ("bir LLM bu bozulmaları yeterli kanıtla teşhis edebilir mi?") ilk
kez adil koşullarda sınanması olacak — ajan artık üç gizli senaryoyu da
görebiliyor.

### 2026-08-26 (7) — Ajana kırılım araçları verildi; D3b'nin manşet bulgusu yanlış çıktı ve düzeltildi

**Neden bu iş yapıldı.** Ajan yalnızca toplam mAP/precision/recall ve 4 sınıf
AP50'si görüyordu. Ama D3b, D4 ve D5'in tamamı tam olarak o veride görünmüyor.
Yani ajan bu üç senaryoyu **yapısal olarak teşhis edemezdi** — LLM yetersiz
olduğu için değil, kanıtı ona vermediğimiz için. Projenin merkezi sorusu
("bir LLM bunu yeterli kanıtla teşhis edebilir mi?") bozuk aletle sınanıyordu.

**Yapılanlar:**

- `metrikler.py`'ye **sınıf-bağımsız eşleştirme** ve sayısal karışıklık
  matrisi eklendi. Mevcut `eslestir()` aynı sınıf şartı aradığı için "doğru
  yerde bulundu ama yanlış sınıf verildi" durumunu göremiyordu.
- Üç yeni ajan aracı: `boyut_bazli_recall_getir`, `kaynak_bazli_recall_getir`,
  `sinif_karisikligini_getir`. Her biri kırılımı sağlıklı referans ve farkla
  birlikte döndürür.
- Kaynak grupları `kaynak_a`, `kaynak_b` … olarak anonimleştirildi; takma
  adlar bbox sayısına göre sabit sıralanır, böylece aynı takma ad tüm
  koşularda aynı kaynağı gösterir (test doğruluyor).
- Dokuz koşunun tamamı için kırılım analizi üretildi (`reports/kirilim/`).
- 20 yeni test; anonimlik sızıntısı taraması dahil.

**Ve bu iş bir hatayı ortaya çıkardı.** Bağımsız karışıklık ölçümü devreye
girince, 2026-08-26 (4) kaydındaki D3b manşet bulgusu doğrulanamadı:

| Kaynak | taşıt doğru tespit |
|---|---:|
| Ultralytics'in raporladığı `tasit` recall (0.8275 × 1264) | ~1.046 |
| Ultralytics `confusion_matrix.png` | 755 |
| Bağımsız ölçüm (`metrikler.py`) | 1.081 |

Ultralytics'in görseli **kendi raporladığı recall ile çelişiyor**; bağımsız
ölçüm ise tutarlı. Fark, conf eşiği ve IoU/güven sıralı eşleştirme
varyantları denenerek yeniden üretilmeye çalışıldı, hiçbiri 358 rakamını
vermedi (en yüksek 30).

**Düzeltilmiş D3b bulgusu** — ve bu, öncekinden bilimsel olarak daha
sağlam: etiketlerin %30'u karıştırılmasına rağmen çapraz sınıf hatası
neredeyse hiç artmıyor (taşıt 2 → 4 kutu; insan 4 → 4). Model bunun yerine
biraz daha temkinli oluyor (kaçırılan kutu 131 → 179 ve 534 → 574). Simetrik
etiket gürültüsü bol veriyle ortalamada sönümleniyor.

**D3 ile kontrast asıl bulgu:** aynı %30 takas, 391 eğitim satırı olan
UAP/UAI'de sınıfı çökertiyor (UAI doğru: 17/17 → 8/17), 131.309 satırı olan
taşıt/insanda soğuruluyor. Etiket gürültüsüne dayanıklılık bozulmanın
oranıyla değil **sınıfın örnek sayısıyla** belirleniyor.

D3'ün kendi bulgusu bağımsız ölçümle **doğrulandı** (yön ve büyüklük aynı;
görsel 11/17 diyordu, bağımsız ölçüm 8/17 — ikisi de sınıfın çöktüğünü
gösteriyor).

**Ders, projeye kalıcı olarak eklendi:** türetilmiş bir görsel, kendi ürettiği
sayısal metriklerle çapraz kontrol edilmeden manşet bulgu yapılmamalı.
Karışıklık iddialarının tamamı artık `metrikler.py`'nin sınıf-bağımsız
eşleştirmesinden üretiliyor ve aynı fonksiyon ajanın aracını da besliyor.
Hatalı ilk bölüm silinmedi; "D3b Sonucu (ILK, HATALI SURUM)" başlığıyla
tarihsel kayıt olarak duruyor.

### 2026-08-26 (6) — D5 tamamlandı: erken durdurmanın gizlediği alan kayması felaketi

D5 (kaynak/alan kayması) koşuldu: eğitim seti yalnızca `aaterm` kaynağıyla
sınırlandı (17.515 → 11.064 kare), etiketlere dokunulmadı. 11 epoch, 1,51
saat.

**Bu, projedeki tek senaryo ki `best.pt` ile `last.pt` tamamen farklı hikâye
anlatıyor — ve ikisi de anlamlı.**

`best.pt` (epoch 1) hipotezi desteklemiyor gibi görünüyor: eğitimde hiç
görülmeyen `termal` kaynağı düşmek yerine +0.1386 yükseliyor (z=+6,98).
Sebep, epoch 1'de modelin henüz tek kaynağa uyum sağlamamış olması; çalışma
noktası daha permissive tarafa kaymış ve bu, taban recall'i en düşük kaynağa
en çok yaramış.

`last.pt` (epoch 11) ise hipotezi tam olarak doğruluyor:

| Kaynak | Eğitimde | n | v00 | last | fark | z |
|---|---|---:|---:|---:|---:|---:|
| **aaterm** | **VAR** | 885 | 0.9107 | 0.8825 | -0.0282 | -1,95 (anlamsız) |
| hituav | yok | 2.165 | 0.8402 | 0.3977 | **-0.4425** | **-29,98** |
| tf2026 | yok | 106 | 0.9906 | 0.4245 | **-0.5661** | **-9,06** |
| termal | yok | 858 | 0.7145 | 0.6480 | -0.0665 | -2,95 |

Eğitimde tutulan tek kaynak korunuyor, diğerleri çöküyor. Genel mAP50
0.9200'den **0.3352**'ye iniyor.

**Pratik ders — projenin en uygulanabilir bulgusu:** kaynak çeşitliliği olan
bir val setinde erken durdurma, bu felaketi önledi. Model 11 epoch boyunca
tek kaynağa doğru çökerken, tüm kaynakları içeren val bunu fark etti ve en iyi
checkpoint olarak epoch 1'i seçti.

Tersi senaryo gerçek hayatta çok daha yaygın: bir ekip kendi topladığı tek
kaynaktan veri toplar ve val setini de **aynı kaynaktan** ayırır. O durumda
val çöküşü göremezdi (`aaterm` korunuyor!), eğitim yakınsayana kadar sürer ve
dağıtılan model diğer sensörlerde catastrofik başarısız olurdu. Yani D5, veri
toplama çeşitliliği kadar **validasyon setinin kaynak çeşitliliği** hakkında
da bir uyarı.

- README'ye "D5 Sonucu" bölümü, otoriter tabloya iki D5 satırı (best/last)
  eklendi; her sayı kaynak JSON'dan programatik doğrulandı.
- `results.csv`'ye `best.pt` kaydedildi (dağıtılacak model odur); `last.pt`
  ölçümü latent kırılganlığı belgelediği için README'de birlikte raporlanıyor.

### 2026-08-26 (5) — D4 tamamlandı: projenin en temiz kontrollü deneyi

D4 (küçük nesne sinyal kaybı) koşuldu: 11 epoch, 2,34 saat. Eğitim
etiketlerinden etkin boyutu 16 px altındaki 29.499 kutu silindi (%22,4).

**Sonuç ders kitabı düzeyinde temiz:**

| Boyut bandı | n | v00 | D4 | fark | z |
|---|---:|---:|---:|---:|---:|
| **<16 px (silinen bant)** | 1.167 | 0.7446 | 0.2922 | **-0.4524** | **-21,87** |
| 16-32 px | 2.011 | 0.8344 | 0.8364 | +0.0020 | +0,17 |
| 32-64 px | 520 | 0.9308 | 0.9192 | -0.0116 | -0,71 |
| >64 px | 316 | 0.9873 | 0.9873 | 0.0000 | 0,00 |

Bozulan bant 45 puan çöküyor, diğer üç bantta anlamlı değişim yok. Etki tam
olarak hedeflenen yere düşüyor, hiçbir yere sızmıyor. Bu, bozulma eşiğinin ve
ölçüm bandının aynı fonksiyondan türemesi sayesinde mümkün oldu.

**D4 ile D1 ayrımı sınıf × boyut kırılımında ortaya çıkıyor.** D4'te silinen
kutuların %76'sı `insan`, yani D4 de D1 gibi ağırlıkla insan'ı etkiliyor.
Ama D4 yalnızca **küçük** insan kutularını kaçırıyor (-0.4650); aynı sınıfın
16-32 px bandında kayıp yok (+0.0082). Sınıf yetersizliği olsaydı kayıp tüm
boyutlara yayılırdı. Ajanın iki senaryoyu ayırt edebilmesi için gereken kanıt
budur ve yalnızca boyut-katmanlı ölçümle görünür.

Toplam insan recall'i: v00 0.7391 → D1 0.7244 (-0.0147, z=-1,22, anlamsız) →
D4 `best.pt` 0.5746 (-0.1645, z=-12,78) → D4 `last.pt` 0.4522 (-0.2869,
z=-21,55). D4'ün `best.pt`'si de epoch 1'den seçildi, yani bu etki bozulmaya
1 epoch maruz kalmış ağırlıklarla bile oluşuyor. Bu, D1 ile farkın maruziyet
süresinden değil bozulmanın **türünden** geldiğini doğruluyor: D4 modele "bu
nesneler arka plandır" diye aktif yanlış bilgi öğretir; D1 yalnızca örnek
sayısını azaltır.

- README'ye "D4 Sonucu" bölümü eklendi, otoriter tabloya D4 satırı ve son iki
  satırın (D3b, D4) neden tek başına okunamayacağına dair uyarı eklendi.
- `metrikler.py`'ye kaynak grubu kırılımı eklendi (modül sözleşmesinin
  eksik üçüncü boyutu); D5 için hazır.
- `scripts/senaryo_D5_kaynak_kaymasi.py` + 10 test yazıldı. D5 manifest-only çalışıyor:
  etiketlere dokunmuyor, görüntü kopyalamıyor.
- `results.csv`, demo ve ajan puanlaması D4'ü tanıyacak şekilde güncellendi.

### 2026-08-26 (4) — D3b tamamlandı  ⚠ bulgusu (7)'de düzeltildi

D3b (taşıt↔insan sınıf karışıklığı) koşuldu: 30 epoch, 6,12 saat, `best.pt`
epoch 11'den seçildi. D3 ile birebir aynı bozulma (%30 takas, seed=42), sadece
bol örnekli sınıf çiftine uygulandı — 39.393 satır değişti.

**Bu kaydın ilk hali hatalı bir manşet bulgu içeriyordu** ("gerçek taşıtların
%28'i insan olarak tahmin ediliyor"). O rakam Ultralytics'in
`confusion_matrix.png` görselinden okunmuştu ve bağımsız ölçümle
doğrulanamadı. Düzeltme ve gerekçesi için aşağıdaki **2026-08-26 (7)**
kaydına, güncel sonuçlar için README'deki "D3b Sonucu" bölümüne bakın.

Doğru olan kısımlar: genel metrikler bozulmayı göstermiyor (taşıt recall
-0.0063 z=-0.42; insan recall +0.0099 z=+0.84; insan precision +0.0015).
Genel recall düşüşü (-0.0495) neredeyse tamamen hiç dokunulmamış UAI
sınıfından geliyor (17 bbox, katkısı -0.0504) — Ultralytics recall'i makro
ortalama olduğu için 17 örnekli sınıf 2.718 örnekli sınıfla eşit ağırlık
taşıyor.

- `results.csv`'ye D3b satırı sona eklendi (ajan eşlemesi kaymasın diye).
- `demo/data_loader.py` ve `demo/app.py` D3b ve v00'ı tanıyacak şekilde
  güncellendi.

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
  `reports/eski_D3_val_sizintili/` altında tarihsel kayıt olarak duruyor.
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
  `reports/eski_D1_protokol_sapmali/` altında tarihsel kayıt olarak duruyor.
- `teshis/degerlendirme/d1_sonuc.py` artık sınıf bazlı **precision ve
  recall** de kaydediyor. Rule 8 "sınıf AP, recall, bbox n" istiyordu ama
  metrik JSON'u yalnızca AP tutuyordu; README tabloları için recall'ı elle
  Ultralytics çıktısından okumak gerekiyordu. Sınıf adları artık
  `ap_class_index` üzerinden eşleniyor, böylece val'de örneği olmayan bir
  sınıf olursa isimler kaymaz.

### 2026-08-25 (6) — D3'te değerlendirme seti sızıntısı bulundu; boyut bazlı metrik modülü yazıldı

**Ciddi bulgu — D3'ün kayıtlı sonucu iyimser yanlı.**
`scripts/senaryo_D3_D3b_sinif_karisikligi.py`, ürettiği `data.yaml`'a eğitim val'i olarak
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

- `scripts/senaryo_D3_D3b_sinif_karisikligi.py` düzeltildi (artık kaynak dataset'in val/test
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
`scripts/ajan_paket_hazirla.py` dört koşuyu dosya yollarıyla hardcode
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
sonraki kez çalıştırılacağında `scripts/ajan_paket_hazirla.py --force` ile
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

- `python -m teshis.degerlendirme.d1_sonuc --model experiments/run_D3_42_local/weights/best.pt --data val_diagnostic/data.yaml --output reports/eski_D3_val_sizintili --scenario D3`
  çalıştırıldı.
- **Yan bulgu:** bu ortamdaki Ultralytics sürümü (8.4.121), göreceli bir
  `project=` yolu verildiğinde çıktıyı sessizce `runs/detect/<project>/`
  altına yazıyor — bizim kodumuz ise dosyaları doğrudan `<project>` altında
  bekliyor (demo, rapor klasörleri). D1/D2a/D2b bu sorunu yaşamamıştı çünkü
  o çalıştırıldığı sırada farklı bir Ultralytics sürümü kurulu olmalıydı;
  D3'te görsel kanıtlar `runs/detect/reports/eski_D3_val_sizintili/...` altında
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
gerçekte çalışan tek şey `scripts/ajan_tek_atislik_calistir.py` içindeki, hiç araç
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
  mantığı `scripts/ajan_puanla.py`'den buraya taşındı, tek kaynak
  oldu; script artık ince bir CLI sarmalayıcısı. Taşıma sonrası çıktı
  gerçek `reports/ajan_denemesi/gemini_response.json` ile bit-bit karşılaştırıldı,
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
`warmup_epochs=2` kullanıyordu; `scripts/kaggle_D2a_lokalizasyon_gurultusu.py`, `kaggle_d2b.py`,
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
- [teshis/egitim/kos.py](teshis/egitim/kos.py), [scripts/senaryo_D2b_eksik_etiket.py](scripts/senaryo_D2b_eksik_etiket.py),
  [scripts/senaryo_D3_D3b_sinif_karisikligi.py](scripts/senaryo_D3_D3b_sinif_karisikligi.py), [scripts/kaggle_D2a_lokalizasyon_gurultusu.py](scripts/kaggle_D2a_lokalizasyon_gurultusu.py),
  [scripts/kaggle_D2b_eksik_etiket.py](scripts/kaggle_D2b_eksik_etiket.py): hepsi artık hiperparametreleri
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

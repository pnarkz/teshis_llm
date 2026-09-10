# Termal Teşhis Ajanı

Termal drone görüntüleriyle çalışan bir YOLO nesne tespit modelini
**kontrollü biçimde bozar**, bozulmanın ölçümlere nasıl yansıdığını ölçer;
sonra bir LLM ajanına bu ölçümleri **anonim** vererek nedeni kanıta dayalı
teşhis edip edemediğini sınar.

**Araştırma sorusu:** Termal nesne tespit sistemi hangi veri, etiket ve
dağılım koşullarında bozulur; bir dil modeli bu bozulmayı yeterli kanıtla
teşhis edebilir mi — ve ürettiği gerekçe savunulabilir mi?

Amaç daha iyi bir model eğitmek değil. Amaç, bir modelin **hangi koşullarda
güvenilirliğini kaybettiğini** ve bunun **ölçülebilir bir izi olup
olmadığını** göstermek.

Sınıf sözleşmesi değişmez: `0 taşıt`, `1 insan`, `2 UAP`, `3 UAI`.

---

## İçindekiler

- [Ne bulduk](#ne-bulduk)
- [Hızlı başlangıç](#hızlı-başlangıç)
- [Yöntem: bir farkı ne zaman "etki" sayıyoruz](#yöntem-bir-farkı-ne-zaman-etki-sayıyoruz)
- [Senaryolar ve koşular](#senaryolar-ve-koşular)
- [LLM teşhis ajanı](#llm-teşhis-ajanı)
- [Neyi söyleyemiyoruz](#neyi-söyleyemiyoruz)
- [Kullanılan araçlar](#kullanılan-araçlar)
- [Proje yapısı](#proje-yapısı)
- [Belgeler](#belgeler)
- [Testler](#testler)

---

## Ne bulduk

**1. Farklı arızalar farklı metrik imzası bırakıyor.** "Model kötü
çalışıyor" demek yetmiyor; **precision mı recall mı bozulduğu arızanın
türünü söylüyor.**

| Arıza | Precision | Recall | Ayırt edici imza |
|---|---|---|---|
| Yanlış çıkarım çözünürlüğü (E4) | **değişmiyor** | **çöküyor** | Bulduğunu doğru buluyor, ama bulamıyor |
| Eksik/karışık etiket (D2b, D3) | **çöküyor** | değişken | Olmayan şeye "var" diyor |
| Küçük nesne sinyali silinmiş (D4) | düşüyor | düşüyor | Kayıp küçük boyut bandında yoğunlaşıyor |

**2. Toplam mAP yerel bir çöküşü tamamen gizleyebiliyor.** D4'te en küçük
nesne bandında recall `0.7446 → 0.2922` düşüyor (gürültü bandının **44
katı**), ama toplam mAP50 farkı bunun yanında küçük kalıyor. Yalnızca genel
metriğe bakan bir denetim bu arızayı görmez.

**3. Gürültü ölçülmeden "etki" iddiası kurulamaz.** Aynı veri, aynı
protokol, yalnızca farklı rastgelelik tohumu ile eğitilen modeller arasında
bile belirgin fark var. Bu taban ölçülünce **yedi iddia zayıfladı** ve bir
senaryo (D6b) bulgu olmaktan çıktı.

**4. Ajan, kanıt gösterildiğinde sorun uydurmuyor.** Hiçbir bozulma
içermeyen kontrol koşularının **9 gözleminin 9'unda da** "anlamlı değişim
yok" dedi (Wilson %95 `[0.701, 1.000]`). Bu bir "model gelişti" bulgusu
değil: araçlar artık her alt grup farkına gürültü bandını ekliyor. Yanlış
pozitifi önleyen şey modelin kendisi değil, **ona gürültü tabanını
göstermek** — projenin ana tezinin ajan tarafındaki karşılığı.

**5. Ajanın hükmü kararlı, ama doğru nedeni bulmakta zayıf.** 13 koşunun
13'ü de üç tekrarında **aynı** hükmü verdi; sözel ifade değişiyor, hüküm
değişmiyor. Buna karşılık bozulma senaryolarında katı doğruluk 0.389.
Baskın hata türü uydurmak değil, **belirtiyi neden sanmak** (D2a) ve
**kaçırmak** (D5).

Ayrıntı ve bütün sayılar: **[docs/BULGULAR.md](docs/BULGULAR.md)**

---

## Hızlı başlangıç

### Sunum konsolu — projenin ana çıktısı

```bash
git clone https://github.com/pnarkz/teshis_llm.git
cd teshis_llm
python -m pip install -r requirements-demo.txt
python -m streamlit run demo/app.py
```

Yedi bölüm: Genel Bakış · Veri ve Sağlıklı Model · Deney Senaryoları ·
Karşılaştırma ve Gürültü · Hata Analizi · LLM Teşhis Ajanı · Sonuçlar ve
Sınırlamalar.

Konsol **yalnızca depoyla gelen ölçüm çıktılarını okur**; eğitim veya test
çalıştırmaz. Model ağırlıkları (`*.pt`), kilitli tanı seti
(`val_diagnostic/`) ve tam görsel arşivi (`reports/` altında 233 MB) Git
dışıdır — taze bir klonda bunlar **bulunmak zorunda değildir**:

| Eksik olan | Konsol ne yapar |
|---|---|
| `val_diagnostic/` | Etiketli örnek galerisi `demo/assets/ornekler` altındaki taşınabilir alt kümeyi kullanır ve bunu ekranda yazar |
| `reports/**/images` | Hata galerisi `demo/assets/sunum_gorselleri` altındaki küçültülmüş seti kullanır ve bunu ekranda yazar |
| `val_batch*.jpg` | "Örnek tahminler" bölümü kendini atlar |
| `*.pt` ağırlıkları | Hiçbir şey; konsol model çalıştırmaz |

Kenar çubuğundaki **Sistem durumu** paneli hangi çıktının bulunduğunu tek
bakışta söyler.

**Sunum yapacaksanız:** [docs/SUNUM_SENARYOSU.md](docs/SUNUM_SENARYOSU.md)
bir saatlik dakika dakika akış, alan dışından dinleyiciler için temel kavram
sözlüğü ve sayfa sayfa ne söyleneceğini içerir.

### Canlı ajan (isteğe bağlı)

```bash
python -m pip install "google-genai>=1.0"
```

`GEMINI_API_KEY` bir **ortam değişkeni** olarak tanımlanır; hiçbir zaman
depoya yazılmaz. PowerShell'de:

```powershell
$env:GEMINI_API_KEY = "..."
```

Ajan bölümü anahtarsız da çalışır — kayıtlı koşu modu varsayılandır ve API
harcamaz. Canlı mod seçildiğinde önce bir ön kontrol çalışır ve hangi
koşulun eksik olduğunu yazar.

### Yeni bir ajan deneyi koşmak

```bash
python scripts/ajan_deney.py --plan --tekrar 3          # önce planı gör
python scripts/ajan_deney.py --calistir --tekrar 3      # deneyi başlat
python scripts/ajan_deney.py --durum --tekrar 3         # nerede kaldı (API'ye gitmez)
python scripts/ajan_deney.py --devam --tekrar 3 --deney <kimlik>
python scripts/ajan_deney_puanla.py                     # puanlama ayrı adımdır
```

`--kuru-calistirma` API harcamadan tam prova yapar; çıktısı `KURU__` önekli
bir klasöre yazılır ve Git dışındadır — prova deney değildir.

### Eğitim ve ölçüm (GPU gerekir)

```bash
python -m pip install -e ".[egitim]"
cp config.example.yaml config.local.yaml   # kendi veri yollarınızı yazın
```

`config.local.yaml` Git dışıdır; yollar göreli yazılabilir ve proje köküne
göre çözülür. Bir senaryonun nasıl uygulandığı ve hangi komutla koşulduğu:
**[docs/KOD_HARITASI.md](docs/KOD_HARITASI.md)**.

---

## Yöntem: bir farkı ne zaman "etki" sayıyoruz

Projenin bütün ağırlığı bu beş kuralda.

**1. Sağlıklı referans (v00).** Veri hiç bozulmadan, senaryolarla **birebir
aynı protokolde** eğitilir. Bozulma karşılaştırmalarının tabanı her zaman
budur — fine-tune edilmemiş modele göre ölçüm yapmak, bozulma etkisiyle
fine-tune etkisini birbirine karıştırır.

**2. Karşılaştırılabilirlik.** Bir fark ancak aday ile referans **dört
kimlik alanında da** aynıysa bozulmaya atfedilebilir: başlangıç modeli,
değerlendirme kümesi, çıkarım çözünürlüğü, checkpoint. Her ölçeğin kendi
referansı ve kendi gürültü eşiği vardır; **başka ölçeğin eşiği ödünç
alınmaz** (`teshis/degerlendirme/karsilastirilabilirlik.py`).

**3. Tek değişken.** Her senaryoda yalnızca bir arıza uygulanır; eğitim
protokolü (`senaryolar/egitim_protokolu.yaml`) sabittir.

**4. Gürültü tabanı.** Hiçbir şey bozulmadan, yalnızca seed değiştirilerek
eğitilen kontrol koşuları arasındaki yayılım ölçülür. Bu bandın altında
kalan bir fark, **büyüklüğü ne olursa olsun** rastgelelikten ayırt edilemez.
Kontrol koşuları **ölçüm aracıdır, ölçüm nesnesi değildir** — hiçbir yerde
bulgu olarak derecelendirilmezler.

**5. Kırılımlı okuma.** Toplam mAP bazı bozulmaları tamamen gizler; sınıf,
nesne boyutu ve veri kaynağı kırılımlarıyla birlikte okunur.

### Kilitli ölçüm seti

Bütün koşular `val_diagnostic` setinde ölçülür: **1.056 görüntü, 4.014
bbox**. Bir kez seçilip kilitlenmiştir. Tek istisna **D6a**'dır: sızıntının
ölçümü ne kadar iyimser yaptığını göstermek için **kasıtlı olarak** sızıntılı
küme üzerinde değerlendirilir ve bu yüzden diğerleriyle aynı tabloda okunmaz.

**Test seti hiç kullanılmadı** ve bu bilinçli bir karardır. Buradaki hiçbir
sayı "nihai test performansı" değildir.

### Yön ayrımı

Bir metriğin beklenenin **tersine yükselmesi** bozulma kanıtı değildir,
eşiği aşsa bile. Derecelendirme yalnızca eşiği aşan **düşüşlere** bakar;
yükseliş gizlenmez, ayrıca söylenir. Tek kaynak:
`senaryo_ozeti.asan_yone_gore`.

---

## Senaryolar ve koşular

Bir **senaryo** bir hipotezdir; bir **koşu** o hipotezin bir kaydıdır. Bazı
hipotezlerin birden fazla kaydı var — aynı eğitimin son epoch'u, farklı bir
başlangıç modeli ya da başka bir rastgelelik tohumu.

**14 senaryo · 26 koşu.** Aritmetik: **20 senaryo koşusu + 6 altyapı koşusu
= 26.**

| Kod | Bozulma | Koşuları | Bulgunun özeti |
|---|---|---|---|
| D1 | Sınıf yetersizliği (insan karelerinin %90'ı) | D1, D1n | main_model kurgusunda **etkisiz**; yolo26n kurgusunda **güçlü** (z=-18,58) |
| D2a | Lokalizasyon etiket gürültüsü | D2a | mAP50-95 en çok düşer (-0.0544) |
| D2b | Eksik etiket (%25) | D2b, D2b final_best | precision -0.1087, recall **artar**: model fazladan kutu üretir |
| D3 | UAP/UAI sınıf karışıklığı | D3 | precision -0.2047; nadir sınıf çöküyor ama n=15/17 |
| D3b | taşıt/insan sınıf karışıklığı | D3b | bol veride bozulma **soğurulur** (çapraz hata 2 → 4 kutu) |
| D4 | Küçük nesne sinyal kaybı | D4, D4 last_pt | yalnızca <16 px bandı çöker (-0.4524, z=-21,9); diğer bantlar değişmez |
| D5 | Kaynak/alan kayması | D5, D5 last_pt | `best.pt`'de görünmez; `last.pt`'de eğitilmeyen kaynaklar çöker |
| D6a | Değerlendirme sızıntısı | D6a | sızıntılı val mAP50-95'i +0.0287 şişirir — birçok gerçek bozulmadan büyük |
| D6b | Tekrar ağırlığı | D6b, D6b last_pt | gürültü tabanı ölçülünce **bulgu olmaktan çıktı** |
| E1 | Aşırı uyum | E1, E1 last_pt | aşırı uyum gerçekleşti; `best.pt` onu tamamen gizliyor |
| E2 | Yetersiz eğitim | E2 | **negatif sonuç:** yakınsamış modelde epoch kesmek underfitting üretmiyor |
| E3 | Aşırı öğrenme oranı | **koşu yok** | **negatif sonuç:** 100 kat lr kararsızlık değil tam ıraksama üretti |
| E3b | Ölçülebilir yüksek öğrenme oranı | E3b seed42, E3b seed43 | kararsızlık ölçüldü: seed'e göre mAP50'de 29 kat oynaklık |
| E4 | Çıkarım çözünürlüğü uyumsuzluğu | E4 imgsz512 | recall'ı çökertir, precision'a dokunmaz — **eşlenik ölçüm** |

**Altyapı koşuları** (hiçbir senaryoya bağlı değil): `v00_saglikli`,
`v00_saglikli last_pt`, `v00n` (sağlıklı referanslar) ve `C2 seed7`,
`C2 seed13`, `C2 seed21` (gürültü tabanını veren kontroller).

**E3 hiç koşu üretmedi:** öğrenme oranı yüz kat artırıldığında eğitim
ıraksadı ve değerlendirilebilir bir model çıkmadı. Bu bir başarısızlık
değil, ölçümün sınırının kaydı; E3b aynı hipotezi on katla tekrarlıyor.

**Eşlenik ölçüm** (E4, D6a): aday ile referans **aynı ağırlık dosyasını**
kullanır, yalnızca çıkarım ayarı değişir. Eğitim rastgeleliği devrede
olmadığı için bu koşularda gürültü eşiği uygulanmaz.

---

## LLM teşhis ajanı

Ajan yalnızca anonim `kosu_NN` kimliği ve ölçüm araçlarını görür. Senaryo
adı, bozulma açıklaması, veri sürümü, dosya yolları ve cevap anahtarı ona
**hiçbir biçimde** gönderilmez. Körlük yapısaldır: filtre bir ad listesine
değil, her koşunun kendi manifestine bakar
(`teshis/ajan/araclar.py::ajana_uygun_mu`).

Ajan **8 araç** çağırabilir; hangi kanıtı isteyeceğine kendisi karar verir.
Puanlama, cevap üretildikten **sonra** ayrı bir yerel işlemde yapılır.

### Tekrarlı deney (`20260909T120445Z__a15487c9`)

13 koşu × 3 tekrar = **39 gözlem**. Sonuçlar **rol bazlı** verilir ve
**toplanmaz**: kontrol koşuları "sorun uyduruyor mu", bozulma senaryoları
"nedeni bulabiliyor mu" sorusunu ölçer.

| Rol | Koşu | Gözlem | Katı puan | Tespit-farkındalıklı |
|---|---|---|---|---|
| Sağlıklı referans | 1 | 3 | 1.000 | 1.000 |
| Kontrol (yalnızca seed farklı) | 3 | 9 | **1.000** | 1.000 |
| Bozulma senaryosu | 9 | 27 | 0.389 | 0.722 |

**Katı puan** uygulanan bozulmanın adını arar. **Tespit-farkındalıklı puan**,
bozulmanın kilitli tanı setinde anlamlı iz bırakmadığı koşularda (D1, D3b,
D6b) "anlamlı değişim yok" cevabını da doğru sayar.

Her gözlem model adını, araç sürümünü, Git commit'ini, çalışma
parametrelerini, **ham ve ayrıştırılmış cevabı ayrı ayrı** ve **her araç
çağrısının cevabının anlık kaydını** taşır.

Bu deney, eski `reports/ajan_denemesi/` sonuçlarıyla **birleştirilmez**:
ikisi farklı araçlarla ölçüldü. Konsolun ajan sayfası ikisini ayrı gösterir
ve testler bunu korur.

---

## Neyi söyleyemiyoruz

Bu bölümün amacı bulguları zayıflatmak değil; hangilerinin ne kadar
dayanıklı olduğunu açıkça söylemek.

- **Senaryo başına tek eğitim koşusu.** Ölçülen her metrik bir nokta
  tahminidir; aynı senaryo yeniden eğitilmediği için senaryo metriğine güven
  aralığı verilemez. (Ajan deneyinin tekrarı vardır; o ayrı bir ölçüdür —
  model kararlılığını ölçer, senaryo evrenindeki belirsizliği değil.)
- **Gürültü tabanı üç bozulmasız koşudan geliyor.** Az gözlemle band gerçek
  yayılımı olduğundan küçük gösterir; eşikler muhtemelen hâlâ dar.
- **Referans tek bir koşudur (v00)** ve sağlıklı koşuların en zayıfıdır.
  Daha sağlam bir taban onların ortalaması olurdu.
- **Nadir sınıflarda örnek yetersiz:** UAP (n=15), UAI (n=17). Bu
  sınıflardaki oranlar genellenemez ve hiçbir yerde tek başına kanıt
  sayılmaz.
- **`last.pt` ölçeğinde yalnızca bir bozulmasız koşu var**, yani orada
  gürültü eşiği hiç hesaplanamıyor.
- **Ajan sonucu tek bir modele ait** (gemini-3.6-flash) ve 9 bozulma
  senaryosu üzerinden hesaplandı; bu senaryoların dışına genellenemez.
- **Final test seti hiç kullanılmadı.**
- **Çalışma zamanı servisi (Aşama 2) tamamlanmadı.** Proje bir ölçüm ve
  teşhis altyapısıdır; canlı bir izleme servisi değildir.

---

## Kullanılan araçlar

### Model tarafı

| Ne | Ayrıntı |
|---|---|
| Mimari | YOLO nesne tespiti, **Ultralytics** (>=8.3) |
| Başlangıç ağırlığı | `main_model.pt`, fine-tune edildi |
| Eğitim / çıkarım çözünürlüğü | 768 px / 768 px |
| Batch / seed | 8 / 42 |
| Optimizer | `auto` — Ultralytics bu modda öğrenme oranını ve momentumu **kendi seçer**; beyan edilen `lr0` bağlayıcı değildir |
| Epoch | 30 planlandı, erken durdurma (sabır 10) ile **11**'de durdu |
| Sınıflar | taşıt, insan, UAP, UAI |
| Boyut bantları | <16 px · 16-32 px · 32-64 px · >64 px |

### Yazılım tarafı

| Katman | Kütüphane |
|---|---|
| Eğitim ve değerlendirme | `ultralytics`, `numpy`, `PyYAML`, `tqdm` |
| Ölçüm ve analiz | `pandas` + `teshis/degerlendirme/` (kendi modülleri) |
| Konsol | `streamlit`, grafikler `altair`, görüntüler `Pillow` |
| LLM ajanı | `google-genai`, model `gemini-3.6-flash`, fonksiyon çağırma |
| Testler | `pytest` |

Güven aralıkları, gürültü bandı hesabı ve karşılaştırılabilirlik kuralları
hazır bir kütüphaneden gelmiyor; `teshis/degerlendirme/` altında bu proje
için yazıldı ve her biri testle bağlı.

---

## Proje yapısı

```text
teshis/
  veri/              Veri tarama, sağlıklı referans, senaryo uygulamaları
  egitim/            Ortak eğitim koşucusu, protokol, koşu kaydı
  degerlendirme/     Metrikler, karşılaştırılabilirlik, gürültü, bootstrap, kanıt
  ajan/              LLM araçları, teşhis döngüsü, çıktı şeması, puanlama
senaryolar/          Deney tanımları ve parametreler (YAML)
scripts/             Komut satırı girişleri (ince); uygulama teshis/ altında
demo/                Streamlit konsolu; bolumler/ içinde yedi ekran
tests/               Davranış ve sözleşme testleri
docs/                Yöntem, kod haritası, bulgular, sunum
reports/             Ölçümler ve ajan deneyleri
```

Bir senaryonun uygulama dosyası, çalıştırma komutu ve sonuç klasörü tek
tabloda: **[docs/KOD_HARITASI.md](docs/KOD_HARITASI.md)**.

---

## Belgeler

| Belge | İçerik |
|---|---|
| [docs/BULGULAR.md](docs/BULGULAR.md) | **Tüm senaryo sonuçları.** Otoriter karşılaştırma tablosu ve her senaryonun ayrıntısı. |
| [docs/SUNUM_SENARYOSU.md](docs/SUNUM_SENARYOSU.md) | **Bir saatlik sunum senaryosu.** Temel kavram sözlüğü, kullanılan araçlar, sayfa sayfa ne gösterilecek ve söylenecek. |
| [docs/SUNUM.md](docs/SUNUM.md) | Teknik olmayan anlatım; konu bazlı. |
| [docs/KOD_HARITASI.md](docs/KOD_HARITASI.md) | Senaryo → uygulama dosyası → çalıştırma komutu → sonuç klasörü. |
| [docs/MIMARI.md](docs/MIMARI.md) | Dosya/klasör sözleşmesi ve adlandırma kuralları. |
| [docs/KURALLAR.md](docs/KURALLAR.md) | Değişmez kurallar, sabit yollar, deney değişmezleri. |
| [docs/CALISTIRMA.md](docs/CALISTIRMA.md) | Kurulum ve komutlar (yerel + Kaggle). |
| [docs/BAKIM_GUNLUGU.md](docs/BAKIM_GUNLUGU.md) | Kronolojik değişiklik kaydı; her düzeltmenin gerekçesi. |
| [docs/proje-brifingi-v2.1.md](docs/proje-brifingi-v2.1.md) | Projenin ilk şartnamesi. |

---

## Testler

```bash
python -m pip install -r requirements-dev.txt -r requirements-demo.txt
python -m pytest -q
```

Depoda **577 test** var. Taze bir klonda da geçer: kilitli tanı setine bağlı
testler otomatik olarak atlanır (`ultralytics` kurulu olmadan da tam paket
çalışır).

Testler yalnızca "çalışıyor mu" diye bakmaz; her biri kapatılan somut bir
hatayı yeniden üretir. Kritik olanlar mutasyonla doğrulandı — eski davranış
geri konduğunda testin gerçekten çöktüğü kontrol edildi. Örnekler:

- Gizli rol veya senaryo bilgisinin ajanın gördüğü üç yüzeye sızması
- Aynı üretimin iki dosyadan iki kez sayılması
- Aynı koşunun birden fazla geçerli tekrarının birbirinin üzerine yazması
- Eşiği aşan bir **yükselişin** bozulma kanıtı sayılması
- Görsel kanıt ölçütünün senaryoları ayırt edememesi
- Sunum metnindeki sayıların ölçümden ayrışması

---

## Bu projenin üç ana dersi

**1. Karşılaştırma tabanı yanlışsa tüm sonuçlar yanlıştır.** Fine-tune
edilmemiş bir modele göre ölçüm yapmak bozulma etkisi ile fine-tune etkisini
karıştırır. Aynı hata daha sinsi biçimde tekrarladı: farklı checkpoint veya
farklı başlangıç modeliyle üretilmiş **sağlıklı** koşular tek bir referansla
tartılınca "güçlü bozulma kanıtı" göründü.

**2. Toplam mAP yalan söyleyebilir.** D3b, D4 ve D5'in tamamı toplam
metriklerde görünmez; yalnızca doğru kırılımla ortaya çıkar.

**3. Aynı kural iki yerde yaşarsa biri geride kalır.** Bu projede bulunan
her ciddi hata bu şekle sahipti. En uç örneği: bir metriğin eşiği aşan
**yükselişi**, aynı büyüklükteki bir düşüşle karıştırılıyordu — ve aynı
hata **dört ayrı yerde** vardı. Bir sayı veya kural ikinci kez yazılacaksa,
türetilmelidir.

---

Güncel değişiklik kaydı: **[docs/BAKIM_GUNLUGU.md](docs/BAKIM_GUNLUGU.md)**

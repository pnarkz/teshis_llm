# PROJE BRİFİNGİ v2.1 — Termal YOLO Sistemi İçin Model Teşhis Ajanı

Sürüm: 2.1
Tarih: 2026-08-16
Durum: Uygulama planı — ölçüm geçerliliği güçlendirilmiş

> **v2.0'dan farkı:** Mimari aynı. Eklenenler: ölçülmüş veri gerçekleri (Ek A),
> karıştırılabilir senaryo çiftleri (Ek B), ajan puanlama cetveli (Ek C), iki ortam
> tanımı, `val_diagnostic`'in UAP/UAİ sınırı, D3 için özel değerlendirme kümesi ve
> arka plan ayrıklık kontrolü, C3'ün bootstrap güven aralığına çevrilmesi, kör test
> kuralları.

> **Bu doküman ne işe yarar:** Yeni bir sohbete veya kod yazacak bir ajana ilk mesajda
> verilir. Bu dokümanı okuyan bir kişi/ajan, önceki hiçbir konuşmayı görmeden projeye
> devam edebilmelidir. **Ek A atlanmamalıdır** — oradaki her sayı ölçülerek bulunmuştur
> ve yeniden keşfedilmesi zaman kaybıdır.

---

## 1. Projenin amacı

Bu proje, termal görüntülerde nesne tespiti yapan bir YOLO sisteminin yalnızca başarımını
ölçmez; başarım düştüğünde veya çalışma koşulları değiştiğinde **sorunun nedenini
kanıtlarıyla açıklayan** bir teşhis sistemi kurar.

Sistem iki katmandan oluşur:

1. **Deterministik analiz katmanı:** veri, eğitim, değerlendirme ve servis loglarından
   ölçülebilir kanıt üretir.
2. **Teşhis ajanı:** bu kanıtları araçlarla sorgular, olası kök nedenleri sıralar, elenen
   hipotezleri belirtir ve doğrulanabilir müdahale önerir.

LLM, ham metrikleri serbest biçimde yorumlayan tek katman olmayacaktır. Ajanın
güvenilirliği, kontrollü olarak oluşturulmuş arıza koşullarında **kör test** ile ölçülecektir.

Bu proje yeni bir YOLO mimarisi geliştirmez. Amaç en yüksek mAP'i elde etmek değil,
**bir modelin veya veri hattının neden güvenilmez hale geldiğini tespit etmektir.**

---

## 2. Gerçekçi kapsam

### Aşama 1 — Çalışan ve ölçülebilir çekirdek

- veri künyesi ve etiket sağlık raporu
- tanısal val (`val_diagnostic`) üretimi
- tek eğitim komutu ve deney kaydı
- sınıf, boyut, kaynak grubu ve hata tipi bazlı değerlendirme
- kontrollü veri/eğitim arızaları (D1-D6, E1-E4)
- deterministik teşhis motoru **ve kural tabanlı taban çizgisinin ölçülmesi**
- LLM ajanın araçlarla bu motoru kullanması
- kör test, puanlama ve araçsız karşılaştırma

### Aşama 2 — Çalışma zamanı izleme

- ONNX/FastAPI çıkarım servisi
- yapılandırılmış çıkarım logları
- bozuk görüntü, gecikme, CPU'ya sessiz düşme ve girdi dağılımı kayması tespiti
- ajan üzerinden servis teşhisi

Aşama 2, Aşama 1'in metrik ve kanıt sözleşmeleri sabitlendikten sonra yapılacaktır.

**Arayüz zorunlu değildir.** CLI ve JSON çıktıları projenin ana teslimidir. İstenirse en
son küçük bir Streamlit ekranı eklenebilir.

---

## 3. Ortamlar

Proje iki ortamda çalışır. Kod içine sabit yol yazılmaz; her yol `config.py` üzerinden
okunur ve ortam adı bir ortam değişkeni veya CLI bayrağıyla seçilir.

```yaml
ortamlar:
  lokal:
    veri_koku: "C:/Users/ASUS/Desktop/HYZ/dataset"
    data_yaml: "C:/Users/ASUS/Desktop/HYZ/dataset/data.yaml"
    gpu: "RTX 3060 Laptop 6 GB"
    kullanim:
      - veri istatistiği (M1)
      - val_diagnostic üretimi (M2)
      - değerlendirme ve kanıt üretimi (M4)
      - ajan ve puanlama (M8, M9)
      - kısa doğrulama çıkarımları
    kullanilmaz:
      - senaryo eğitim koşuları (6 GB yetersiz ve yavaş)

  kaggle:
    veri_koku: "/kaggle/input/datasets/pnarkcgz/dataset/dataset"
    gpu: "Tesla T4 x2  (device=[0,1])"
    kullanim:
      - tüm eğitim koşuları (D1-D6, E1-E4, C1-C2)
```

**Bilinen ortam tuzakları:**

- DDP modunda `results.save_dir` dict döner. Ağırlık yolu doğrudan
  `runs/<name>/weights/best.pt` olarak kurulmalı.
- GPU başına `batch=2`'ye düşmek BatchNorm'u bozar; en az 6 olmalı. (E3 senaryosunda
  bu **kasten** kullanılacak.)
- Kaggle oturumları uçucudur — ana süreç script tabanlı olmalı, notebook yalnızca
  keşif ve görselleştirme için kullanılmalı.

---

## 4. Veri ve sürüm politikası

**Sınıflar:**

| ID | Sınıf | Açıklama |
|---|---|---|
| 0 | `tasit` | Taşıt |
| 1 | `insan` | İnsan |
| 2 | `UAP` | Uçan Araba Park alanı — parlak dairesel branda |
| 3 | `UAI` | Uçan Ambulans İniş alanı brandası |

> UAP ve UAİ termalde renk görünmediği için birbirine çok benzer; tek ayırt edici işaret
> üzerlerindeki "UAP"/"UAİ" yazısıdır. Bu, D3 senaryosunun gerçek dünya karşılığıdır.

Mevcut veri kümesi **21.846 karedir**. Kaynak grupları `aaterm`, `termal`, `hituav`,
`sentetik` ve `tf2026` önekleriyle ayırt edilir. Tam künye **Ek A**'dadır.

### 4.1 Sentetik veri hakkında

Sentetik kareler yalnızca train'dedir ve UAP/UAİ nesnelerini farklı konum ve boyutlarda
üretmek için kullanılmıştır. **Aynı arka planın farklı sentetik varyantları bu projenin
bağlamında kasıtlı veri artırmadır; tek başına veri sızıntısı kabul edilmeyecektir.**

### 4.2 Roboflow tekrarları hakkında

Dosya adında `.rf.<32-hex-hash>` eki bulunan karelerde aynı kaynak görüntünün farklı
artırılmış dışa aktarımları vardır. Bu, indirilen açık veri setlerinden gelmiştir;
proje sahibi tarafından üretilmemiştir.

Kaynak adını çıkarmak için standart fonksiyon:

```python
import re
from pathlib import Path

_RF = re.compile(r'_(?:jpg|jpeg|png|webp)\.rf\.[0-9a-fA-F]{16,}$', re.I)

def kaynak_adi(dosya_adi: str) -> str:
    """aaterm__005994_jpg.rf.85e9cd47....jpg -> aaterm__005994"""
    return _RF.sub('', Path(dosya_adi).stem)
```

**Zorunlu raporlama kuralı:** Tüm raporlar iki sayı vermelidir —
(a) dosya/kare sayısı, (b) kaynak görüntüye indirgenmiş benzersiz sahne sayısı.

Roboflow tekrarları **otomatik olarak sızıntı ilan edilmez.** Gerçek durum ölçülmüştür
(Ek A.4) ve D6 senaryosunun taban çizgisidir.

### 4.3 Veri değişmezliği

`dataset/` ve test etiketleri deney sırasında değiştirilmez. Bozulmuş veri sürümleri ayrı
klasörlerde ve manifest ile üretilir. Bir deneyin hangi dosyalarla üretildiği sonradan
tekrar oluşturulabilmelidir.

---

## 5. Val tanımı

Bu projede "val" tek bir anlama gelmeyecektir.

### 5.1 Operasyonel val

Mevcut `dataset/{images,labels}/val` bölmesidir. Grup temsili ve split ayrımı açısından
kullanılabilir durumdadır; sentetik veri val/test'e alınmamıştır.

Ancak mevcut val sınıf dağılımı train ve test'ten belirgin biçimde farklıdır
(Ek A.3). Bu yüzden operasyonel val:

- eğitim sırasında hızlı geri bildirim için kullanılabilir;
- tek başına "modelin genel sağlığını" veya sınıflar arası karşılaştırmayı kesin biçimde
  temsil etmez;
- **bu dokümanda "sağlıklı val" olarak adlandırılmaz.**

### 5.2 Tanısal val (`val_diagnostic`)

Model seçimi ve ajan karşılaştırması için ayrıca üretilecektir. Koşullar:

1. Test ile **tamamen ayrı kaynak karelerden** oluşmalı (kaynak adı seviyesinde kontrol).
2. Aynı kaynak görüntünün Roboflow varyantları tekilleştirilmeli — **kaynak başına tek varyant**.
3. Sentetik kare içermemeli.
4. `aaterm`, `termal`, `hituav`, `tf2026` gruplarının her biri temsil edilmeli.
5. Dört sınıfın her biri bulunmalı; UAP/UAİ için kutu sayısı raporda ayrıca belirtilmeli.
6. Sınıf ve kaynak dağılımı açıkça tanımlanmalı ve manifest'e yazılmalı.
7. Yalnızca toplam mAP değil, sınıf ve kaynak grubu kırılımlarıyla raporlanmalı.

Amaç sınıfları yapay olarak eşitlemek değildir. Amaç, ölçümün tek bir kaynak grubuna veya
tek bir sınıfa aşırı bağımlı olmasını önlemektir.

### 5.3 ⚠ `val_diagnostic` UAP/UAİ sorununu ÇÖZMEZ

Bu, dokümanın en önemli sınırlarından biridir ve gizlenmeyecektir.

UAP/UAİ'nin **sentetik olmayan** tek kaynağı `tf2026` grubudur (toplam 145 kare;
87 UAİ + 102 UAP kutusu). Operasyonel val'e düşen payı 21 karedir →
**yaklaşık 15 UAP + 17 UAİ kutusu.** Tekilleştirme sonrası bu sayı azalabilir, artamaz.

**Sonuçları:**

- UAP/UAİ AP değerleri **her zaman** kutu sayısı ve bootstrap güven aralığıyla birlikte
  raporlanır. Çıplak `AP = 0,995` yazılması yasaktır.
- Ajan, bu sınıflarda düşük örnek sayısını kesin teşhis gibi sunmamalıdır →
  `genel_durum: "yetersiz_kanit"` kullanılmalıdır.
- **D3 senaryosu bu kümede ölçülemez** (Bölüm 7, D3 maddesine bakınız).

Zorunlu raporlama formatı:

```
UAP  AP: 0,995   bbox: 15   bootstrap %95 GA: [0,84 – 1,00]
UAI  AP: 0,995   bbox: 17   bootstrap %95 GA: [0,86 – 1,00]
```

### 5.4 Test seti

Geliştirme boyunca açılmaz. Son model ve karar protokolü sabitlendikten sonra
**yalnızca bir kez** kullanılır.

---

## 6. Ana modüller

```text
teshis/
  config.py                 # ortam ve yol yönetimi
  veri/
    istatistik.py           # M1 — künye, dağılımlar, etiket sağlığı
    val_olustur.py          # M2 — val_diagnostic üretimi
    surum_uret.py           # M3 — bozulmuş veri sürümü üretimi
    bozulmalar.py           # bozulma fonksiyonları
    manifest.py             # manifest yazma/okuma/doğrulama
  egitim/
    kos.py                  # M4 — tek komutla eğitim
    kayit.py                # results.csv kaydı
  degerlendirme/
    metrikler.py            # M5 — sınıf/boyut/grup kırılımları
    hata_tipolojisi.py      # FN/FP/lokalizasyon/sınıf karışıklığı
    bootstrap.py            # güven aralığı hesabı
    kanit.py                # kanıt sözleşmesi — ajanın okuyacağı format
    kural_tabanli.py        # M6 — deterministik teşhis kuralları (ajan taban çizgisi)
  ajan/
    araclar.py              # M7 — function calling araçları
    ajan.py
    semalar.py              # JSON şemaları
    puanlama.py             # M8 — kör test puanlama
  servis/                   # Aşama 2
    api.py
    loglama.py
    ariza.py
    anomali.py

senaryolar/                 # YAML yapılandırmaları
veri_surumleri/             # üretilen veri sürümleri + manifest
runs/                       # Ultralytics eğitim çıktıları
experiments/                # kanıt paketleri (ajanın okuduğu)
results.csv                 # deney kaydı
```

Her modül CLI'dan çalışabilmeli **ve** Python API olarak çağrılabilmelidir.

---

## 7. Arıza senaryoları

**10 çekirdek koşu.** Bileşik arızalar ayrı katalog maddesi değildir; son doğrulama
aşamasında iki veya üç çekirdek arızanın kombinasyonu olarak üretilir.

**Ortak sabitler:** model `yolo11s`, `imgsz=640`, sabit eğitim alt kümesi, epoch 60
(E1/E2 hariç), seed 42 (C2 hariç), değerlendirme `val_diagnostic` üzerinde
(D6-a ve D3 hariç — aşağıya bakınız).

### Veri kaynaklı senaryolar

**D1 — Sınıf yetersizliği**
Train'den `insan` kutusu içeren karelerin büyük bölümü (%90) çıkarılır.
*Beklenen kanıt:* insan recall/AP belirgin düşer; taşıt/UAP/UAİ neredeyse sabit; veri
istatistiğinde insan temsilinin azaldığı görülür.
*Ajan önerisi:* veri ekleme veya kontrollü örnekleme.

**D2 — Etiket kalitesi bozulması** (iki alt test, aynı aile)
- **D2-a:** kutu merkezleri, kutu boyutunun %15'i kadar rastgele kaydırılır.
  *Beklenen:* mAP50 az düşer, **mAP50-95 çok düşer**; lokalizasyon hata tipi artar.
- **D2-b:** kutuların %25'i rastgele silinir.
  *Beklenen:* **precision düşer**, recall görece korunur; FP'ler görsel olarak doğru nesneler.

*Ajan, iki durumu aynı "etiket sorunu" altında bırakmayıp ayırmalıdır.* Bu ayrım
puanlamada `+2` ile `+1` arasındaki farkı belirler.

**D3 — UAP/UAİ sınıf karışıklığı**
UAP ve UAİ etiketlerinin %30'u karşılıklı takas edilir.
*Beklenen kanıt:* confusion matrix'te iki yönlü çapraz karışıklık; iki sınıfta benzer düşüş.

> **⚠ Özel değerlendirme kümesi zorunlu.** `val_diagnostic`'te yalnızca ~15-17 UAP/UAİ
> kutusu vardır; %30 takas ~4-5 kutu demektir ve sinyal gürültüden ayrılamaz.
> Bu nedenle D3, **`val_d3` adlı senaryoya özel bir değerlendirme kümesinde** ölçülür:
>
> - Sentetik UAP/UAİ kareleri içerir (train havuzundan ayrılmış).
> - **Arka plan ayrıklık kontrolü zorunludur:** `val_d3` içindeki sentetik karelerin
>   türetildiği arka plan kareleri, eğitim setinde **hiçbir varyantıyla** bulunmamalıdır.
>   Kontrol yöntemi:
>   1. Copy-paste hattının arka plan eşlemesi kayıtlıysa doğrudan kullanılır.
>   2. Kayıtlı değilse algısal hash (pHash) ile eğitim karelerine karşı taranır;
>      Hamming mesafesi eşiğin altındaki eşleşmeler `val_d3`'ten çıkarılır.
>   3. Ayrıca `kaynak_adi()` ile Roboflow varyant kontrolü yapılır.
> - Bu kümenin amacı gerçek dünya AP'si kestirmek **değildir**; etiket karışıklığının
>   *tespit edilebilirliğini* ölçmektir. Manifest'e ve rapora bu şekilde yazılır.

**D4 — Küçük nesne sinyalinin kaybı**
Eğitim etiketlerinden `√alan_eff < 16 px` olan kutular filtrelenir (kareler kalır).
*Beklenen kanıt:* **yalnızca küçük boyut bininde** recall düşüşü; orta/büyük bin sabit.
*Ayrım:* E4'ten ayrıştırılmalıdır (Ek B).

**D5 — Kaynak alanı kayması**
Eğitim tek bir kaynak grubuyla (`aaterm`) sınırlandırılır; değerlendirme tüm gruplarda.
*Beklenen kanıt:* kaynak grupları arasında belirgin performans farkı.
*Ayrım:* D1 (sınıf dengesizliği) ile karıştırılmamalıdır (Ek B).

**D6 — Split veya tekrar ağırlığı problemi** (iki alt test)
- **D6-a:** kaynak görüntüler kasten train ve val'e dağıtılır (~%20 kaynak).
  *Beklenen:* val'in yapay biçimde yükselmesi; val/test uçurumu.
  *Not:* bu senaryo doğası gereği bozuk bir val kullanır; `val_diagnostic` yerine
  kasten sızdırılmış bir val ile ölçülür ve bu manifest'te belirtilir.
- **D6-b:** belirli kaynakların tekrar sayısı 15×'ten 40×'e çıkarılır.
  *Beklenen:* o sahnelerin baskınlaşması; nadir sınıflarda precision bozulması.

Gerçek veri kümesindeki Roboflow tekrarları (Ek A.4) taban çizgisi olarak ayrıca raporlanır.

### Eğitim kaynaklı senaryolar

**E1 — Overfitting**
Küçük veri alt kümesi (~1.000 kare) + uzun eğitim (200 epoch) + augmentasyon kapalı.
*Beklenen kanıt:* yüksek train metriği, düşük val metriği, açılan train/val farkı;
val loss'un dip yapıp yükselmesi.

**E2 — Underfitting / erken kesme**
Çok az epoch (5) ile eğitim durdurulur.
*Beklenen kanıt:* train ve val metriklerinin birlikte düşük olması; **loss'un hâlâ
düşüyor olması**; eğride plato yok.

**E3 — Eğitim kararsızlığı**
Öğrenme oranı aşırı yükseltilir (`lr0=0.1`) **veya** batch aşırı küçültülür
(GPU başına 2).
*Beklenen kanıt:* gürültülü loss/eğri; seed'ler arasında olağandışı oynaklık.
**Tek koşudan kesin hüküm verilmez; en az iki seed gerekir.**

**E4 — Eğitim/çıkarım sözleşmesi ihlali**
Model bir çözünürlükte eğitilip farklı çözünürlükte veya uyumsuz ön işlemeyle
değerlendirilir (640'ta eğit, 1280'de değerlendir).
*Beklenen kanıt:* **tüm boyut gruplarında** düşüş.
*Ayrım:* D4'ten ayrıştırılmalıdır (Ek B).

### Çalışma zamanı senaryoları (Aşama 2)

Bunlar eğitim koşularına karıştırılmaz; servis logları üzerinde ayrı değerlendirilir.

**R1 — Açık servis arızası:** bozuk görüntü, beklenmedik çözünürlük, istek hatası.
Log'da açık hata satırı üretir.

**R2 — Sessiz servis arızası:** GPU'dan CPU'ya düşme veya girdi kaynak dağılımının
değişmesi. **Hata log'u üretmeyebilir**; gecikme, güven skoru ve tespit sayısı dağılımı
birlikte incelenmelidir.

---

## 8. Kontrol koşulları

| Kod | Tanım | Amaç |
|---|---|---|
| **C1** | Sağlıklı eğitim yapılandırması, seed 42 | Referans taban. Ajan burada sorun uydurursa `−1` |
| **C2** | Aynı yapılandırma, seed 7 | Ajan seed kaynaklı dalgalanmayı "sorun" sanıyor mu? |
| **C3** | `val_diagnostic` üzerinde **bootstrap güven aralığı** hesabı | Metrik farklarının anlamlı olup olmadığını belirler |

### C3 — Bootstrap protokolü

> **Not:** C3 "aynı modeli tekrar değerlendirmek" değildir. TTA yoksa değerlendirme
> deterministiktir ve tekrar aynı sayıyı verir. Değişkenlik **örneklemden** gelir.

- **Yeniden örnekleme birimi: görüntü** (kutu değil). Aynı karedeki kutular ilişkilidir;
  kutu bazlı bootstrap güven aralığını yapay olarak daraltır.
- `val_diagnostic` görüntüleri yerine koymalı (with replacement) olarak yeniden örneklenir.
- Kaynak grubu bazında tabakalı (stratified) örnekleme yapılır — her grubun payı korunur.
- `B = 1000` tekrar. Her tekrarda tüm metrikler yeniden hesaplanır.
- **Yüzdelik yöntemi** ile %95 güven aralığı: `[p2.5, p97.5]`.
- Çıktı her sınıf için: `AP`, `bbox_sayisi`, `GA_alt`, `GA_ust`.

**Kullanım kuralı:** İki koşu arasındaki metrik farkı, güven aralıkları örtüşüyorsa
"fark var" diye raporlanmaz. Ajanın da bu kurala uyması beklenir.

---

## 9. Ölçüm katmanı (kanıt sözleşmesi)

Her eğitim koşusu için `experiments/<kosu_adi>/kanit.json` üretilir. İçeriği:

- genel `mAP50` ve `mAP50-95`
- sınıf bazlı precision, recall, F1, AP — **her biri bbox sayısı ve bootstrap GA ile**
- nesne boyutu binlerine göre recall (`√alan_eff` üzerinden, bkz. Ek A.5)
- `aaterm`, `termal`, `hituav`, `tf2026` bazında metrikler
- confusion matrix ve background kaynaklı FP/FN sayıları
- hata örnekleri: eksik tespit (FN), yanlış pozitif (FP), sınıf karışıklığı, kötü lokalizasyon
- train/val eğrileri, en iyi epoch, val loss dip epoch'u, train/val farkı
- kullanılan veri sürümü, seed, model, imgsz, batch, lr, augmentasyon ayarları
- kare sayısı **ve** benzersiz kaynak sayısı (Bölüm 4.2 kuralı)

**Tek başına mAP hiçbir teşhis için yeterli kanıt sayılmaz.**

---

## 10. Ajan araçları ve çıktı sözleşmesi

### 10.1 Araçlar

```text
veri_istatistigi(veri_surumu, kirilim)
egitim_yapilandirmasi(kosu_adi)
egitim_egrisi(kosu_adi)
metrik_getir(kosu_adi, kirilim)          # kirilim: genel|sinif|boyut|grup|guven_dagilimi
hata_ornekleri(kosu_adi, sinif, tur, adet)
karisiklik_matrisi(kosu_adi)
kosu_karsilastir(kosu_a, kosu_b)
log_ozeti(zaman_araligi, filtre)          # Aşama 2
```

Tüm metrik döndüren araçlar, sonuçları **bbox sayısı ve güven aralığıyla birlikte**
vermek zorundadır.

### 10.2 Çıktı şeması (zorunlu)

```json
{
  "genel_durum": "saglikli | sorunlu | yetersiz_kanit",
  "guven": 0.0,
  "teshisler": [
    {
      "kok_neden": "etiket_eksikligi",
      "siddet": "kritik | orta | dusuk",
      "aciklama": "1-3 cümle",
      "kanit": [
        {"arac": "metrik_getir", "bulgu": "insan precision 0,52 (n=6002), recall 0,79"}
      ],
      "elenen_hipotezler": [
        {"hipotez": "overfitting", "neden_elendi": "train/val farkı 0,02, uçurum yok"}
      ],
      "onerilen_mudahale": "",
      "dogrulama_olcumu": "",
      "curutucu_sonuc": ""
    }
  ]
}
```

**`yetersiz_kanit` durumu zorunlu olarak kullanılmalıdır** — özellikle UAP/UAİ gibi az
kutulu sınıflarda güven aralığı geniş olduğunda. Bunu doğru kullanan ajan ödüllendirilir
(Ek C).

---

## 11. Kör test kuralları

Bu bölümdeki kurallar ihlal edilirse **tüm ajan değerlendirmesi geçersizdir.**

1. **`manifest.json` ajana kesinlikle verilmez.** Bozulmanın türü, parametreleri ve
   senaryo kodu orada yazılıdır. Ajan yalnızca `experiments/<kosu_adi>/kanit.json` ve
   araçlar üzerinden bilgi alır.
2. **Senaryo adı ve kodu ajana verilmez.** Koşu adları nötr olmalıdır
   (`kosu_07`, `kosu_12` gibi — `d1_insan_seyrek` gibi değil).
3. **Veri sürümü adı da sızdırılmamalıdır.** `veri_istatistigi()` aracı sürüm adını
   değil, yalnızca istatistikleri döndürür.
4. Ajanın gördüğü tek şey: metrikler, eğitim eğrileri, veri istatistikleri, hata
   örnekleri, confusion matrix, eğitim yapılandırması.
5. Puanlayan taraf (insan veya `puanlama.py`) manifest'i görür; ajan görmez.

---

## 12. Ajan değerlendirmesi

### 12.1 Ölçülecekler

- kök nedeni doğru bulma (Ek C cetveli)
- doğru kanıtı kullanma
- yanlış pozitif teşhis oranı (C1/C2 üzerinde)
- birden fazla arızayı aynı anda bulabilme (bileşik koşular)
- **araçlı ajan ile tüm metrikleri tek promptta alan ajan arasındaki fark** (ablation)
- **kural tabanlı deterministik taban çizgisi ile ajan arasındaki fark**
- aynı girdide 5 tekrarın tutarlılığı
- gereksiz araç çağrısı sayısı

### 12.2 Üç taban çizgisi

Ajanın performansı üç şeye karşı raporlanır:

| Taban | Tanım |
|---|---|
| **T0 — Kural tabanlı** | `kural_tabanli.py`, LLM yok. Adım 6'da yazılır ve ölçülür |
| **T1 — Araçsız LLM** | Tüm metrikler tek promptta metin olarak verilir, araç yok |
| **T2 — Araçlı ajan** | Tam sistem |

Raporlanacak cümle formatı:

> *"T0 (kural tabanlı) 10 senaryonun X'ini doğru ayırdı. T1 (araçsız LLM) Y. T2 (araçlı
> ajan) Z. Ajanın kural tabanlı tabana katkısı Z−X senaryo."*

### 12.3 Öneri kalitesi

Öneri puanı, **teşhisin hangi ölçümle doğrulanacağının yazılmasına** göre artırılır.
"Veriyi artır" gibi genel öneriler tam doğru kabul edilmez. `dogrulama_olcumu` ve
`curutucu_sonuc` alanları doldurulmalıdır.

---

## 13. Deney disiplini

Her kayıt `results.csv` içine tek satır olarak yazılır:

```text
deney_adi, senaryo, veri_surumu, seed, model, imgsz_egitim, imgsz_degerlendirme,
epoch, batch, lr0, degisen_degisken, degerlendirme_kumesi,
mAP50, mAP50_95, AP_tasit, AP_insan, AP_UAP, AP_UAI,
n_tasit, n_insan, n_UAP, n_UAI,
recall_kucuk, recall_orta, recall_buyuk,
mAP_aaterm, mAP_termal, mAP_hituav, mAP_tf2026,
val_loss_dip_epoch, train_val_farki, sure_dk, agirlik_yolu
```

- Bir deneyde yalnızca **bir ana değişken** değiştirilir.
- Bileşik senaryolarda bileşenler manifest'te açıkça yazılır.
- Test verisi geliştirme sırasında kullanılmaz.

### Sağlık turu (her çalıştırmanın başında)

- dataset yolu ve YAML kontrolü
- görüntü/etiket eşleşmesi
- sınıf ID aralığı (0-3)
- boş ve bozuk etiketler
- modelin yüklenmesi
- tek görüntüde çıkarım
- rastgele seçilmiş örneklerde görsel kutu doğrulaması

> **Uyarı:** Örneklem **rastgele** olmalı, alfabetik ilk N olmamalı. Dosya adları grup
> önekiyle başladığı için alfabetik kesme tek gruba kilitlenir ve tüm istatistikler o
> grubun istatistiği olur. (Bu hata bir kez yapılmış ve düzeltilmiştir.)

---

## 14. Uygulama sırası

1. `veri.istatistik` — mevcut veri için tek JSON/HTML rapor
2. `val_olustur` — `val_diagnostic` üret, kaynak tekilleştirmesini manifest'le
3. `degerlendirme` — sınıf/kaynak/boyut/hata kırılımları + bootstrap GA
4. `egitim` — tek komutla eğitim ve `results.csv` kaydı
5. D1-D6 ve E1-E4 arızalarını üret ve doğrula
6. **`kural_tabanli.py`** — deterministik teşhis kurallarını yaz, ajanı devreye almadan
   T0 tabanını ölç
7. Ajan araçlarını ve JSON sözleşmesini ekle
8. Kör test, T1 araçsız taban ve tutarlılık deneylerini çalıştır
9. R1-R2 servis kolunu ekle (Aşama 2)
10. En son, tek seferlik test değerlendirmesi ve sınırların raporlanması

---

## 15. İlk teslim

İlk kod teslimi yalnızca üç parçadan oluşur:

1. `veri.istatistik` komutu ve JSON raporu
2. `val_olustur` komutu ve `val_diagnostic/manifest.json`
3. Mevcut model için `val_diagnostic` değerlendirme komutu (bootstrap GA dahil)

**Bu üç parça doğrulanmadan eğitim senaryoları ve LLM ajanı yazılmayacaktır.** Böylece
ajan, hatalı veya tanımı belirsiz bir val metriğini açıklamaya çalışmayacak; önce ölçüm
zemini sabitlenecektir.

---

## 16. Başarı ölçütleri

Proje başarılı sayılmak için yüksek mAP göstermesi gerekmez. Şunlar raporlanmalıdır:

- deterministik analiz, her arıza ailesini doğru kanıtla ayırabiliyor mu (T0)
- ajan kör testte kök nedeni ve kanıtı ne sıklıkla doğru buluyor (T2)
- sağlıklı kontrolde yanlış alarm oranı nedir (C1/C2)
- araç kullanımı teşhise ölçülebilir katkı sağlıyor mu (T2 − T1)
- ajan kural tabanlı tabana katkı sağlıyor mu (T2 − T0)
- az örnekli UAP/UAİ sınıflarında hangi sonuçlar belirsiz kalıyor
- çalışma zamanı sessiz arızaları, açık hata arızalarından daha geç veya daha hatalı mı
  yakalanıyor (Aşama 2)

Başarı iddiası bu ölçümlere dayanacak; tek bir mAP değerine veya LLM'in akıcı
açıklamasına dayanmayacaktır.

---

## 17. Kapsam dışı

- yeni YOLO mimarisi geliştirme (sıfırdan CNN/DETR/ViT yazma)
- elle yeni veri etiketleme
- tracking, track ID ve hareket analizi, video işleme
- RGB hattı / konum kestirimi
- `landing_status` / `motion_status` üretimi
- TEKNOFEST yarışma hattının yeniden kurulması
- zorunlu web arayüzü
- test setini tekrar tekrar kullanarak model seçme
- şampiyon modelin mAP'ini iyileştirmeye çalışmak — bu bir **teşhis** projesidir,
  modeller kasten bozulacaktır

---
---

# EK A — Ölçülmüş veri gerçekleri

> Bu bölümdeki her sayı tam veri üzerinde ölçülerek bulunmuştur. Yeniden keşfetmeye
> çalışma. Ölçüm tarihi: 2026-08-16.

## A.1 Hacim

```
Toplam kare      : 21.846
Benzersiz kaynak : 11.212
Toplam bbox      : 154.141

Split:  train 17.515  |  val 2.164  |  test 2.167
```

## A.2 Kaynak grupları ve **çözünürlük**

| Grup | Kare | **Çözünürlük** | Açıklama |
|---|---|---|---|
| `aaterm__` | 13.830 | **1024×1024** | Açık termal veri seti (Roboflow) |
| `termal__` (v1) | 4.732 | 640×512 | Açık termal veri seti (Roboflow) |
| `hituav__` | 2.866 | 640×512 | Açık termal veri seti |
| `sentetik` | 273 | 640×512 | Copy-paste UAP/UAİ kareleri, yalnızca train |
| `tf2026` | 145 | 640×512 | TEKNOFEST örnek videosundan elle etiketlenmiş |

**Kritik:** Veri **iki farklı çözünürlükte**. `aaterm` (%63 kare) 1024×1024, geri kalan
hepsi 640×512. `imgsz=640` eğitiminde aaterm 0,625× küçültülür, diğerleri 1,0× kalır.

**Sonuçları:**
- Efektif boyut hesabı zorunlu: `olcek = 640 / max(W, H)`, `kok_alan_eff = √alan × olcek`
- D4 ↔ E4 ayrımı bu bilgi olmadan yapılamaz
- `imgsz=1024`'e çıkmak val mAP'ini yükseltir (val'in %64'ü aaterm) ama hedef domainde
  (`tf2026`, 640×512) kazandırmaz — **metrik tuzağı**

Görüntü modu karışık: ~%85 RGB, ~%15 tek kanal gri (L). Ultralytics otomatik çoğaltır.

## A.3 Sınıf dağılımı ve **val dengesizliği**

| Sınıf | train | val | test | TOPLAM |
|---|---|---|---|---|
| tasit | 83.600 | **1.473** | 10.804 | 95.877 |
| insan | 47.709 | **6.002** | 4.091 | 57.802 |
| UAP | 237 | **15** | 16 | 268 |
| UAI | 154 | **17** | 23 | 194 |

**val dengesizliği — `val_diagnostic`'in var oluş gerekçesi:**

```
split    tasit/insan oranı
train         1,75
test          2,64
val           0,25      ← 7× sapma
```

val insan-baskın (%80 insan), test taşıt-baskın (%27 insan). Kare sayıları grup bazında
oransal ama içerik oransal değil. val'deki `tasit` AP'si yalnızca 1.473 kutuya dayanır.

Boş kare oranı da farklı: **val %17,3** · train %7,2 · test %7,9.

**Grup × sınıf:**

| Grup | UAI | UAP | insan | tasit |
|---|---|---|---|---|
| aaterm | 0 | 0 | 32.154 | 81.566 |
| hituav | 0 | 0 | 12.312 | 12.439 |
| v1_termal | 0 | 0 | 12.647 | 1.715 |
| sentetik | 107 | 166 | 579 | 0 |
| tf2026 | 87 | 102 | 110 | 157 |

> UAP/UAİ'nin sentetik olmayan tek kaynağı `tf2026`. Bu, Bölüm 5.3'teki sınırın kaynağıdır.

## A.4 Roboflow tekrarları ve sızıntı durumu

```
Çoğaltma oranı:  train 1,94×  |  val 2,05×  |  test 1,95×
Maksimum:        train 15×    |  val 5×     |  test 10×

Benzersiz kaynak: train 9.042  |  val 1.058  |  test 1.114
```

`aaterm` ve `termal__` gruplarında var; `hituav`, `tf2026`, `sentetik` gruplarında yok
(bu gruplarda `.rf.` eki bulunmaz).

**Sızıntı durumu — ölçüldü:**

```
Birden fazla split'e dağılmış kaynak : 2 / 11.212   (%0,02)
Ardışık kare komşuluğu sızıntısı     : 1 çift (hituav__1_130_30_0_09949/09950)
Aynı dosya adı birden fazla split'te : 0
```

**Yorum:** Gerçek sızıntı ihmal edilebilir düzeydedir; mevcut split stratejisi kaynak
seviyesinde doğru çalışmıştır. Ancak val'in **efektif bağımsız sahne sayısı 2.164 değil
~1.058**'dir → güven aralığı ~1,4× geniştir. Bu, bootstrap GA hesabının gerekçesidir.

Bazı sahneler eğitimde 15× ağırlık taşımaktadır — istenmeyen, gizli bir oversampling.
D6-b senaryosu bunu abartarak test eder.

## A.5 Kutu geometrisi

**Orijinal piksel (√alan):**

| Sınıf | medyan | %5 | %95 | COCO small payı |
|---|---|---|---|---|
| insan | 20,1 | 10,2 | 43,0 | **%79,7** |
| tasit | 54,7 | 19,4 | 154,3 | %17,2 |
| UAP | 76,0 | 40,7 | 101,1 | %0,4 |
| UAI | 76,0 | 39,6 | 285,2 | %0,5 |

**Efektif piksel (`imgsz=640` girdisinde) — D4'ün temeli:**

| Sınıf | medyan efektif | <8 px | <12 px | <16 px | <24 px |
|---|---|---|---|---|---|
| **insan** | **16,6 px** | %5,5 | %20,4 | **%45,7** | %88,3 |
| tasit | 35,9 px | %0,5 | %3,7 | %8,3 | %20,9 |

> İnsan medyanı, P3 (stride-8) katmanının güvenilir çalışma sınırı olan ~16 px'in tam
> üzerindedir. D4 senaryosunun eşiği (`< 16 px`) bu ölçüme dayanır.

**Önerilen boyut binleri (`kok_alan_eff`):** `<16` · `16-32` · `32-64` · `>64`

## A.6 Yoğunluk

```
Kare başına nesne:  train medyan 4, %99 = 55, maks 300
                    val   medyan 3, %99 = 12, maks 32
İnsan içeren karelerde kare başına insan: medyan 3, %90 = 9, maks 130
5+ insan içeren kare: 4.150 (%35,9)
```

En kalabalık kareler `aaterm` park alanı / trafik sahneleridir.

## A.7 Etiket sağlığı (mevcut durum — beklenen çıktı, yeni bulgu değil)

```
Kare dışına taşma      : 32   (hepsi termal__thermal_015xx bloğunda — tek toplu hata)
Sıfır boyutlu kutu     :  2
Birebir tekrar eden    : 22   (20 tasit, 2 insan — hepsi train'de)
√alan < 6 px           : 124  (119 insan, 5 tasit) — toplam insanın %0,2'si
Aşırı en-boy (>10/<0,1): 206  (194 tasit)
Görüntü/etiket eşleşme : tam, eksik yok
```

Sağlık turu bu sayıları üretmelidir. Farklı bir sayı çıkarsa veri değişmiş demektir.

## A.8 Sentetik vs gerçek branda dağılımı

KS testi (H0: aynı dağılım), `√alan` üzerinde:

```
UAI: D=0,364  p=3,35e-06  → FARKLI   (n_sentetik=107, n_gerçek=87)
UAP: D=0,357  p=1,12e-07  → FARKLI   (n_sentetik=166, n_gerçek=102)

UAI gerçek  : medyan 66,5  std 94,9  aralık 30–368
UAI sentetik: medyan 80,0  std 12,5  aralık 54–100
```

Sentetikler dar bir boyut bandına sıkışmıştır; çok yakın ve çok uzak branda senaryoları
yoktur. **D3'ün özel değerlendirme kümesi tasarlanırken bu sınır rapora yazılmalıdır.**

## A.9 Oversampling'in gerçek etkisi (ölçüldü)

`data_oversample.yaml` UAP/UAİ içeren kareleri 5× tekrarlar.

```
          ham_%   efektif_%   pay_degisim
insan     36,23     36,97        +0,74
tasit     63,48     61,59        -1,89
UAP        0,18      0,87        +0,69
UAI        0,12      0,57        +0,45

Epoch başına kare: ham 16.248 → efektif 17.744  (%9 artış)
```

Yalnızca 374 kareyi etkiler. **"Oversampling insan sınıfından kapasite çalıyor" hipotezi
yanlıştır** — insan payı düşmez, artar (sentetik karelerde 579 insan kutusu vardır).
D6-b senaryosu tasarlanırken bu taban dikkate alınmalıdır.

## A.10 Mevcut model referans metrikleri (C1 için taban)

**Şampiyon:** `runs/b2_s1_640/weights/best.pt` — YOLO11m, `imgsz=640`, ~40 MB

| Metrik | Değer | Not |
|---|---|---|
| val mAP50 | 0,944 | TTA ile 0,9466 |
| val mAP50-95 | 0,69 | |
| AP tasit | 0,921 | n=1.473 — dar taban |
| AP insan | 0,866 | P=0,893 · R=0,821 · F1=0,855 |
| AP UAP | 0,995 | **n=15 — güven aralığı geniş** |
| AP UAI | 0,995 | **n=17 — güven aralığı geniş** |

**Elenmiş yaklaşımlar (tekrar denenmesin):**

| Yaklaşım | Sonuç |
|---|---|
| SAHI (320 dilim, %25 örtüşme) | mAP 0,845 · 4,3× yavaş → elendi |
| 1280 fine-tune | 0,878 → elendi (iki confounding: fine-tune + batch=2) |
| 1280 çıkarım (640-eğitimli model) | 0,7913 → E4'ün gerçek dünya karşılığı |
| 960 + TTA | 0,9414 |

Eğitim eğrileri epoch 25-40'ta platoya oturmuş, val/box_loss epoch 39'da dip yapıp
yükselmiştir.

---
---

# EK B — Karıştırılabilir senaryo çiftleri

Ajan testinin değeri buradan gelir. Her senaryonun benzersiz ve bariz bir imzası olsaydı
test hiçbir şey ölçmezdi. Rapordaki hata analizinin iskeleti bu tablodur.

| # | Çift | Neden karışır | **Ayırt eden sinyal** |
|---|---|---|---|
| 1 | **D2-a ↔ D2-b** | ikisi de etiket kalitesi ailesinde | D2-a: mAP50 az, **mAP50-95 çok** düşer; lokalizasyon hata tipi artar · D2-b: **precision** düşer, FP'ler görsel olarak doğru nesneler |
| 2 | **D2-b ↔ D3** | ikisi de "etiket yanlış" | D2-b: FP'ler background'a karşı · D3: confusion matrix'te **iki yönlü çapraz**, iki sınıf simetrik düşer |
| 3 | **D4 ↔ E4** | ikisi de küçük nesnede çöküyor | D4: **yalnızca** `<16 px` bininde düşüş · E4: **her** binde düşüş, küçüklerde daha ağır |
| 4 | **D1 ↔ D6-b** | ikisi de sınıf dengesi sorunu | D1: **recall** çöker (bulamıyor) · D6-b: **precision** çöker (her yerde görüyor) |
| 5 | **D1 ↔ D5** | ikisinde de belirli bir kesim kötü | D1: **sınıf** bazlı ayrışma, gruplar benzer · D5: **kaynak grubu** bazlı ayrışma, sınıflar benzer |
| 6 | **E1 ↔ D6-a** | ikisinde de val ilişkisi tuhaf | E1: val **düşük**, train yüksek, val loss yükseliyor · D6-a: val **yüksek**, test düşük |
| 7 | **E2 ↔ E3** | ikisi de "eğitim yetersiz" görünür | E2: loss **düzgün iniyor**, erken kesilmiş, plato yok · E3: loss **gürültülü**, hiç oturmamış, seed'ler arası oynak |
| 8 | **E3 ↔ C2** | ikisi de seed hassasiyeti gösterir | E3: seed'ler arası fark **güven aralığından büyük** · C2: fark GA içinde → sorun yok |

**Rapor için:** Ajanın karışıklık matrisi (hangi senaryoyu hangisiyle karıştırdı)
çıkarılıp bu tabloyla karşılaştırılmalıdır. Beklenen karışıklıklar mı gerçekleşti, yoksa
öngörülmeyen bir karışıklık mı çıktı?

---
---

# EK C — Ajan puanlama cetveli

## C.1 Senaryo başına puan

| Puan | Kriter |
|---|---|
| **+2** | Kök nedeni **doğru belirledi** ve **doğru kanıtı gösterdi** |
| **+1** | Doğru problem alanını buldu, kök nedeni yanlış adlandırdı (örn. "etiket sorunu var" ama D2-a ile D2-b'yi ayıramadı) |
| **0** | Yanlış kök neden |
| **−1** | Sağlıklı kontrolde (C1/C2) olmayan sorun uydurdu, **veya** yanlış yönde müdahale önerdi |

### Gerekçe

Negatif puan zorunludur. Sağlıklı bir modele *"insan sınıfın yetersiz, veri topla"* diyen
bir ajan, sessiz kalan ajandan **daha kötüdür** — boşa GPU, zaman ve etiketleme emeği
harcatır.

## C.2 Ek puan kuralları

| Durum | Etki |
|---|---|
| `dogrulama_olcumu` ve `curutucu_sonuc` alanları anlamlı doldurulmuş | Tam puan koşulu. Boşsa en fazla `+1` |
| Öneri genel ("veriyi artır", "daha çok eğit") | Tam puan alamaz |
| `elenen_hipotezler` doldurulmuş ve gerekçeli | Kısmi doğrulukta `+1` sayılmasını sağlayabilir |
| UAP/UAİ'de `yetersiz_kanit` doğru kullanılmış | **+1 bonus** — dürüst çekimserlik ödüllendirilir |
| UAP/UAİ'de n=15 üzerinden kesin teşhis konmuş | **−1** — sahte kesinlik cezalandırılır |
| Bileşik senaryoda her iki arıza da bulunmuş | Her biri ayrı puanlanır (maks `+4`) |
| Bileşik senaryoda yalnızca biri bulunmuş | Bulunan için puan, diğeri `0` |

## C.3 Puanlama akışı

1. Ajan çıktısı JSON olarak alınır.
2. Puanlayan taraf `manifest.json`'u açar (ajan görmemiştir).
3. `kok_neden` alanı, manifest'teki `bozulma_turu` ile karşılaştırılır.
4. `kanit` dizisi, Ek B'deki "ayırt eden sinyal" ile karşılaştırılır.
5. C.2 kuralları uygulanır.
6. `puanlama.py` sonucu `ajan_sonuclari.csv`'ye yazar.

## C.4 Raporlanacak toplamlar

```
Senaryo sayısı            : 10 çekirdek + 3 kontrol + N bileşik
Maksimum puan             : hesaplanır
T0 (kural tabanlı) puanı  :
T1 (araçsız LLM) puanı    :
T2 (araçlı ajan) puanı    :
Yanlış pozitif oranı      : C1/C2'de −1 alma sıklığı
Tutarlılık                : 5 tekrarda aynı kok_neden oranı
Ortalama araç çağrısı     : senaryo başına
Gereksiz araç çağrısı     : kanıt olarak kullanılmayan çağrı sayısı
```

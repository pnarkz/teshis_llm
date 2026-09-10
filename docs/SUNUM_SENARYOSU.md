# Sunum Senaryosu — sayfa sayfa

Konsolu ekranda gezerken ne gosterecegini ve ne soyleyecegini sayfa sayfa
anlatir. Konu bazli anlatim icin: [SUNUM.md](SUNUM.md). Sayisal ayrinti:
[BULGULAR.md](BULGULAR.md).

## Bu metin neden sayi ezberletmiyor

Cumleler sayilari **ekrandan okutur** ("kapsam kutusundaki sayi"), metnin
icine yazmaz. Bu projede elle yazilan sayilar tekrar tekrar bayatladi — "bes
iddia zayifladi" cumlesi elle sayilmisti, dogrusu yediydi. Ekranda ne
yaziyorsa dogrusu odur; metin oraya isaret eder.

Iki istisna var ve ikisi de teste baglidir: ajan deneyinin kontrol sonucu ve
tekrar tutarliligi. Bunlar cumlenin kendisi oldugu icin yazili duruyorlar.

---

## 0. Bir saatlik akis

| Dk | Bolum | Ekran |
|---|---|---|
| 0-5 | Problem ve sistem | Genel Bakis |
| 5-12 | Temel kavramlar (asagidaki sozluk) | Genel Bakis, ana grafik uzerinde |
| 12-20 | Veri ve saglikli model | Veri ve Saglikli Model |
| 20-28 | Senaryo nedir, kac tane var | Deney Senaryolari |
| 28-40 | Bir senaryoyu bastan sona | Karsilastirma ve Gurultu |
| 40-45 | Hata neye benziyor | Hata Analizi |
| 45-55 | Kor LLM teshisi | LLM Teshis Ajani |
| 55-60 | Ne bulduk, ne bulamadik | Sonuclar ve Sinirlamalar |

Salonda alan disindan insanlar var. Kavramlari **ilk gectikleri yerde**
aciklayin, bastan toplu bir ders anlatmayin - dinleyici o an ekranda
karsiligini gorurse akilda kaliyor.

---

## 0.1 Temel kavramlar sozlugu

Her maddenin yaninda **nerede aciklanacagi** yazili.

### Nesne tespiti ve kutu (bbox) — *Veri sayfasi, Etiketli ornekler*

Model bir goruntude "burada bir insan var" demekle kalmaz, **nerede**
oldugunu da soyler. Bunu bir dikdortgenle isaretler: **bounding box**,
kisaca **bbox**. Bu projede dort sinif var: **tasit, insan, UAP, UAI**.

> "Yesil kutular gercek etiket - insanin isaretledigi dogru cevap. Kirmizi
> kutular modelin tahmini. Yesilin yaninda kirmizi yoksa model o nesneyi
> kacirmis demektir."

### IoU — *Hata Analizi, kare aciklamasinda*

Model bir kutu ciziyor, gercek kutu baska bir yerde. Ne kadar ortusuyorlar?
**IoU** (Intersection over Union) iki kutunun kesisim alanini birlesim
alanina boler. 1.0 tam ustuste, 0 hic ortusmuyor demek.

> "Kabul esigi 0.50: model bir nesneyi 'buldum' sayilmak icin gercek
> kutuyla en az yarim yarıya ortusen bir kutu cizmeli."

### Precision ve recall — *Genel Bakis, ana grafikten hemen sonra*

Iki farkli hata turu var ve bunlari ayirmak bu projenin temeli:

- **Precision (kesinlik):** modelin bulduklarinin ne kadari dogru?
  Dusuk precision = model olmayan seye "var" diyor (**fazladan kutu**).
- **Recall (duyarlilik):** gercekte var olanlarin ne kadarini buldu?
  Dusuk recall = model var olani goremiyor (**kacirma**).

> "Bir guvenlik kamerasi dusunun. Precision dusukse her golgeye alarm
> caliyor. Recall dusukse gercek bir insani kaciriyor. Ikisi farkli
> arizadir ve **hangisinin bozuldugu arizanin turunu soyler** - projenin
> en net bulgusu bu."

### mAP50 ve mAP50-95 — *Veri sayfasi, saglikli model metrikleri*

**AP (Average Precision)** bir sinif icin precision ve recall'i tek sayida
birlestirir. **mAP** bunun butun siniflar uzerindeki ortalamasidir (mean
AP). Sondaki sayi IoU esigidir:

- **mAP50**: IoU esigi 0.50. "Kutu kabaca dogru yerde mi?"
- **mAP50-95**: esik 0.50'den 0.95'e kadar onar onar artirilip ortalanir.
  "Kutu ne kadar **hassas** yerlestirilmis?" Her zaman mAP50'den dusuktur.

> "mAP50 0.92 demek, kabaca dogru yerde kutu cizme basarisi %92. Ayni
> model mAP50-95'te 0.67 - yani kutuyu bulmasi iyi, tam oturtmasi daha
> zor. Bu normal ve beklenen bir fark."

### Epoch, checkpoint, seed — *Veri sayfasi, egitim kunyesi*

- **Epoch:** modelin butun egitim verisini bir kez bastan sona gormesi.
- **Checkpoint:** egitim sirasinda kaydedilen model dosyasi. Iki tanesi
  onemli: **best.pt** (dogrulama skoru en iyi epoch) ve **last.pt** (son
  epoch). Normalde best.pt raporlanir.
- **Seed:** rastgeleligi sabitleyen sayi. Ayni seed ayni sonucu verir;
  farkli seed ayni kurulumda bile biraz farkli bir model uretir.

> "Bu ucu kunyede duruyor cunku **hangi checkpoint'i raporladiginiz
> sonucu degistirebiliyor**. E1 senaryosu bunun kanitidir: en iyi epoch'la
> baktiginizda model saglikli gorunuyor, son epoch'ta ariza ortaya
> cikiyor."

### Fine-tune — *Veri sayfasi, "Fine-tune ne kazandirdi"*

Sifirdan egitmek yerine, onceden egitilmis bir modeli kendi verinizle
kisa sure daha egitmek. Bu projede baslangic agirligi `main_model.pt`.

### Gurultu tabani (bu projenin omurgasi) — *Genel Bakis, ana grafikte*

**En onemli kavram budur ve ana grafikte gorseli var.**

Ayni veriyi, ayni ayarlari kullanip **yalnizca seed'i** degistirerek uc
model egittik. Bu modeller arasindaki fark, hicbir sey bozulmadan ortaya
cikan farktir - yani **saf rastgelelik**. Buna **gurultu bandi** diyoruz.

> "Bir senaryoda 0.03'luk bir dusus gordum diyelim. Bu bir bulgu mu? Once
> sunu sormam lazim: hicbir sey bozmadigim kosular arasinda da 0.03 fark
> cikiyor mu? Cikiyorsa o dusus bozulmanin degil, sansin eseri olabilir.
> Kontrolleri birden uce cikardigimda **yedi iddiam zayifladi ve bir
> senaryo bulgu olmaktan cikti**."

### Kilitli tani seti — *Veri sayfasi, ilk gosterge*

Butun olcumlerin yapildigi, bir kez secilip sonra hic degistirilmeyen
goruntu kumesi: **1.056 goruntu, 4.014 etiketli nesne**. Test seti bundan
ayridir ve hic kullanilmadi.

---

## 0.2 Kullanilan araclar

Sorulursa; bastan anlatmaya gerek yok.

### Model tarafi

| Ne | Ayrinti |
|---|---|
| Mimari | YOLO nesne tespiti, **Ultralytics** kutuphanesi (>=8.3) |
| Baslangic agirligi | `main_model.pt` (onceden egitilmis), fine-tune edildi |
| Egitim cozunurlugu | 768 px · **cikarim** da 768 px |
| Batch / seed | 8 / 42 |
| Optimizer | `auto` — **onemli:** Ultralytics bu modda ogrenme oranini ve momentumu kendi secer, beyan edilen `lr0` baglayici degildir |
| Planlanan / durulan epoch | 30 planlandi, erken durdurma (sabir 10) ile **11**'de durdu |
| Siniflar | tasit, insan, UAP, UAI |

### Yazilim tarafi

| Katman | Kutuphane |
|---|---|
| Egitim ve degerlendirme | `ultralytics`, `numpy`, `PyYAML`, `tqdm` |
| Olcum ve analiz | `pandas`, kendi yazdigim `teshis/degerlendirme/` modulleri |
| Konsol (bu ekran) | `streamlit`, grafikler `altair`, goruntuler `Pillow` |
| LLM ajani | `google-genai`, model **gemini-3.6-flash**, fonksiyon cagirma ile **8 arac** |
| Testler | `pytest` — depoda 500'un uzerinde test var |

> "Guven araliklari, gurultu bandi hesabi ve karsilastirilabilirlik kurallari
> hazir bir kutuphaneden gelmiyor; `teshis/degerlendirme/` altinda kendi
> yazdigim modullerde ve her biri testle bagli."

**Kod haritasi:** bir senaryonun nasil uygulandigi sorulursa
`docs/KOD_HARITASI.md` tek tabloda senaryo -> uygulama dosyasi -> calistirma
komutu -> sonuc klasoru veriyor.

---

## 1. Genel Bakis — ilk 60-90 saniye

**Hedef:** izleyici uc seyi anlasin — hangi soru, sistem nasil calisiyor,
neden onemli. Bu sayfada ayrinti anlatilmaz; her ayrinti kendi sayfasinda.

### Ekranda ne var

Baslik ve tek cumlelik tanim, dort gosterge, uc asamali sema, ornek bulgu
grafigi (D4), butun kosularin etki haritasi, iki kapanis sayisi.

### Ne soyle

> "Termal drone goruntuleriyle calisan bir nesne tespit modelini kontrollu
> olarak bozuyorum, bozulmanin olcumlere nasil yansidigini olcuyorum, sonra
> bir dil modeline bu olcumleri **anonim** verip nedeni bulup bulamadigini
> siniyorum."

**Dort gostergeyi tek tek okuma.** Yalnizca ikisine dokun:

> "Kilitli tani seti — butun olcumler ayni goruntu kumesinde yapildi, ve
> **test seti hic kullanilmadi**. Ajan deneyi ise tekrarli: her kosu ucer kez
> soruldu."

Uc asamali semada durulacak tek nokta:

> "Ucuncu asamada ajan **goruntulere bakmiyor**. Gordugu tek sey olcum
> ciktilari — genel metrikler ve kirilim tablolari, anonim kosu kimligiyle.
> Senaryo adini, ne yaptigimi ve cevap anahtarini hicbir zaman gormuyor."

Bu cumleyi atlarsan salondaki herkes "LLM goruntuye bakip teshis koyuyor"
sanir. En sik yanlis anlasilan nokta budur.

### Ana grafik — sayfanin kalbi

Baslik ekranda: *"Genel skor, kucuk nesnelerdeki kaybi gizleyebilir."*

> "Bu D4 senaryosu: kucuk nesnelerin sinyalini kasitli olarak zayiflattim.
> Daire saglikli referans, kare bozulmus model. Aradaki mesafe farkin
> kendisi.
>
> Gri serit onemli: o serit, **hicbir bozulma icermeyen** kosular arasinda o
> grupta gordugum yayilim. Yani rastgeleligin kendisi. Ucte uc grup seridin
> **icinde** — o farklar bozulma kaniti degil. Yalnizca en kucuk nesne grubu
> seridin disina cikiyor, ve bandin kirk kat uzerinde."

Sonra tek cumleyle genelle:

> "Genel mAP'ye baksaydim bu kaybi ya hic gormezdim ya da 'biraz dusmus'
> derdim. Kirilim olmadan bu ariza gorunmuyor."

**Uyari:** bu tek senaryonun sonucu. Ekranda da oyle etiketli ("ornek bulgu ·
D4"). "Butun senaryolarda boyle" deme.

### Etki haritasi — ozelden genele

> "Simdi butun kosulara ayni olcuyu uyguladim. Her hucre, o farkin gurultu
> bandina orani. Bir'in altindaki hucreler sonuk — cunku o buyuklukteki bir
> fark, hicbir sey bozmadigim kosular arasinda da goruluyor.
>
> Renk yonu de tasiyor: kirmiziya giden hucreler dusus, camgobegi olanlar
> **yukselis**. Yukselisi ayirmak zorundaydim, cunku bir metrigin beklenenin
> tersine yukselmesi bozulma kaniti degil."

Sorulursa: haritada bandi asan bir yukselis var (D1, mAP50-95). Ekranda
aciklama satirinda yazili; oradan oku.

### Kapanis — iki sayi

> "Iki sayi birakacagim. Birincisi: hicbir bozulma icermeyen, yalnizca
> rastgelelik tohumu farkli kontrol kosularinda ajan **dokuz gozlemin
> dokuzunda da** sorun uydurmadi. Ikincisi: on uc kosunun **on ucu de** uc
> tekrarinda ayni hukmu verdi — kelimeler degisiyor, hukum degismiyor."

**Ucuncu bir sayi ekleme.** Uc sayi verilirse hicbiri akilda kalmaz.

### Beklenen sorular

**"Ajanin genel basari orani ne?"**
> "Tek bir oran vermiyorum, cunku iki farkli soru olcuyorum. Kontrol
> kosulari 'uyduruyor mu' sorusunu, bozulma senaryolari 'nedeni bulabiliyor
> mu' sorusunu olcer. Birlestirilmis ortalama ikisini de yaniltir. Rol bazli
> tablo Ajan sayfasinda."

**"Neden mutlak fark yerine oran?"**
> "Cunku kucuk bir grupta buyuk gorunen bir fark, o grubun dogal yayilimi
> icinde olabilir. Once gurultuyu olctum, sonra etkiyi. Kontrolleri birden
> uce cikardigimda yedi iddiam zayifladi ve bir senaryo bulgu olmaktan
> cikti."

**"Test setini neden kullanmadin?"**
> "Kullanmadim ve bilerek kullanmadim. Model secimi ve butun karsilastirmalar
> tani setinde yapildi; test seti bir kez kullanildiginda bir daha tarafsiz
> olmaz."

### Bu sayfada YAPMA

- Dort gostergeyi tek tek okuma; ikisi yeter.
- Rubrik puanlarina girme (%50/%68/%83). Ayri seyleri olcuyorlar ve daha
  proje anlatilmadan puanlama sistemi anlatmak zorunda kalirsin. O tartisma
  Ajan sayfasinda.
- Grafikten senaryo detayina dalma. Biri D4'un nasil uygulandigini sorarsa
  "Senaryolar sayfasinda gosterecegim" de ve devam et.

---

## 2. Veri ve Saglikli Model — guvenilirlik zemini

**Hedef:** "olctugun sey guvenilir mi" sorusunu izleyici sormadan cevapla.
Bu sayfa bulgu anlatmaz; butun bulgularin uzerine kuruldugu iki seyi
gosterir — kilitli tani seti ve hic bozulmamis referans model.

**Sure:** 60-90 saniye. Uc sekmenin ucunde de durma; ikisi yeter.

### Sekme 1 — Veri seti

Dort gostergeden **yalnizca birine** dokun:

> "Butun olcumler tek bir kumede yapildi: kilitli tani seti. Val bolumunden
> secildi, kaynak tekilligi ve split ayrikligi gozetildi, sonra kilitlendi.
> Kilitlendikten sonra hicbir senaryo icin degistirilmedi."

Sinif dagilimi grafigini goster ve **kendi sinirlamani sen soyle**:

> "Burada bir dengesizlik var ve bunu bir sonuc olarak degil, sinirlama
> olarak sunuyorum. UAI ve UAP sinifları onlu sayilarda bbox ile temsil
> ediliyor. O siniflarda olculen bir oran tek bir nesneye asiri duyarlidir.
> Bu yuzden nadir sinif sonuclari projede hicbir yerde tek basina kanit
> sayilmiyor."

Bunu sen soylemezsen mentor sorar; sen soylersen yontem bilincin gorunur.

Kaynak dagiliminda tek cumle:

> "Kaynak gruplari ayri cekim kosullari demek. Model performansi kaynaga
> gore ciddi degisiyor — D5 senaryosu tam olarak bunu olcuyor."

Veri sagligi taramasini **atla**, sorulursa ac.

### Sekme 2 — Etiketli ornekler

Hizli gec. Amaci tek: verinin gercekten neye benzedigini gostermek.

> "Kutular calisma zamaninda etiket dosyasindan ciziliyor, onceden
> hazirlanmis bir goruntu degil. Yani ekranda gordugunuz kutu, olcumde
> kullanilan kutunun ta kendisi."

### Sekme 3 — Saglikli referans model

Dort metrik kartini okuma; ustundeki kutuyu oku:

> "Bu model butun karsilastirmalarin tabani. Senaryolarla **birebir ayni
> protokolde**, hic bozulmamis veriyle egitildi."

Sonra kimlik kartlarini goster — sayfanin en onemli yeri burasi:

> "Bir senaryonun metrigi bu modelle ancak dort alan da ayniysa
> karsilastirilabilir: baslangic modeli, degerlendirme kumesi, cikarim
> cozunurlugu, checkpoint. Bu dortlu tutmuyorsa fark bozulmadan degil,
> kurulum farkindan geliyor olabilir."

Bu kurali bir kez anlatirsan Karsilastirma sayfasindaki "eslenik olcum"
ve "esik yok" etiketleri kendiliginden anlasilir.

Kirilimli performans grafiklerinde tek cumle — ve bu cumle Genel
Bakis'taki ana grafigi hatirlatir:

> "Dikkat edin: **saglikli** modelde bile kucuk nesnelerde recall belirgin
> dusuk. Bu bozulma degil, verinin kendi zorlugu. Senaryolar bu tabanin
> UZERINE eklenen etkiyi olcuyor."

Egitim egrisi, sorulursa:

> "Train ve val kayiplari birlikte iniyor, asiri uyum imzasi yok. E1
> senaryosu ayni grafikte acilan bir makas gosteriyor."

### Beklenen sorular

**"Test setini gercekten hic kullanmadin mi?"**
> "Hic. Model secimi dahil her sey tani setinde yapildi. Test seti bir kez
> kullanildiginda bir daha tarafsiz olmaz; final asamasina sakladim."

**"Fine-tune ne kadar kazandirdi?"**
> "Ekranda sifir merkezli fark grafigi var. Ama dikkat: bu bir senaryo
> etkisi degil, iki farkli egitim durumunun karsilastirmasi. Bozulma
> senaryolarinin tabani her zaman v00; fine-tune edilmemis modele gore
> olcum yapsaydim bozulma etkisiyle fine-tune etkisi birbirine karisirdi."

**"Optimizer neden auto?"**
> "Ultralytics bu modda ogrenme oranini ve momentumu kendisi seciyor, yani
> beyan edilen lr0 baglayici degil. Kunyede acikca yaziyor. E serisinde
> ogrenme oranini degistiren senaryolar bu yuzden optimizer'i da acikca
> ayarliyor."

### Bu sayfada YAPMA

- Uc sekmeyi de bastan sona gezme; bir ve uc yeter.
- Egitim kunyesinin tamamini acma. Dort kimlik alani gorunuyor, gerisi
  sorulursa acilir.
- Sinif bazli performansta UAP/UAI'nin yuksek skorlarini basari gibi
  sunma; onlar 15 ve 17 bbox ile olculuyor.

---

## 3. Deney Senaryolari — senaryo mu, kosu mu?

**Hedef:** izleyici "14 senaryo" ile "26 kosu" arasindaki farki anlasin.
Bu ayrimi yapmazsaniz sayfalardaki sayilar birbirini tutmuyor gorunur.

### Once ayrimi kur

Sayfanin ustundeki **"14 senaryo, 26 kosu — hangisi hangisine bagli?"**
bolumunu acin.

> "Bir **senaryo** bir hipotezdir: 'kucuk nesne sinyalini silersem ne
> olur'. Bir **kosu** o hipotezin bir kaydidir. Bazi hipotezlerin birden
> fazla kaydi var."

Tablodan iki ornek gosterin, uc degil:

> "D4'un iki kaydi var: ana kosu ve ayni egitimin son epoch'u. E3b'nin iki
> kaydi var, ikisi yalnizca seed'de ayriliyor. D1'in ikinci kaydi farkli
> bir model ailesiyle egitildi."

Sonra aritmetigi soyleyin:

> "**20 senaryo kosusu + 6 altyapi kosusu = 26.** Altyapi kosulari bir
> hipotez degil: uc saglikli referans olcumun tabanini, uc kontrol kosusu
> gurultu esigini verir. Ikisi de **olcum aracidir, olcum nesnesi degil** —
> bu yuzden hicbir yerde bulgu olarak derecelendirilmiyorlar."

**Dikkat:** senaryolardan biri (**E3**) hic kosu uretmedi — ogrenme orani
yuz kat artirildiginda egitim iraksadi ve degerlendirilebilir bir model
cikmadi. Bu bir basarisizlik degil, olcumun sinirinin kaydi; E3b ayni
hipotezi on katla tekrarliyor.

### Sonra bir senaryo secin

D4'te kalin. Kart uzerinde ne olctugunu, neyin degistirildigini ve neyin
sabit tutuldugunu gosterin. Butun senaryolari tek tek gezmeyin.

---

## 4. Karsilastirma ve Gurultu — bir senaryo bastan sona

**Hedef:** tek bir senaryoyu yontemin butun adimlarindan gecirmek. Sunumun
en uzun bolumu burasi (yaklasik 12 dakika) ve **tek senaryoda** kalin.

### Sirasiyla

**1. Iki asamali secici.** Once senaryo, sonra kosu.

> "D4'u sectim. Bu senaryonun iki kaydi var; su an ana kosuya, yani en iyi
> epoch'la raporlanmis haline bakiyorum."

**2. Ne degistirildi / ne sabit kaldi.**

> "Tek degisen sey kucuk nesne esigi. Baslangic modeli, degerlendirme
> kumesi, cozunurluk, checkpoint ve seed sabit. Bu liste sekil degil:
> karsilastirmanin gecerliligi tam olarak bu dortlunun ayni olmasina
> bagli."

**3. Genel metrikler.** Y ekseni sifirdan basliyor.

> "Ekseni kirpsaydim bu farklar dramatik gorunurdu. Kirpmadim."

**4. Fark ve gurultu kusagi.** Sagdaki grafik.

> "Gri kusak gurultu bandi. Kusagin icinde kalan bir nokta saf
> rastgelelikten ayirt edilemez. Burada mAP50 ve precision disariya
> cikiyor, digerleri cikmiyor."

**5. Kirilimlar.** Boyut ve kaynak.

> "Genel metrik -0.02 dedi ama kirilima bakinca 0-16 px bandinda recall
> 0.74'ten 0.29'a dusmus. **Genel skor bu cokusu gizliyordu.**"

**6. Gorsel kanit.** Ayni kare, iki model.

> "Varsayilan olcut 'saglikli modelden en cok ayrisan' — yani bu bozulmanin
> hangi kareyi bozdugu. 'En fazla kacirilan' olcutunu secersem her
> senaryoda neredeyse ayni kare gelir, cunku o kare zaten en kalabalik
> olani."

Kaynak filtresini ve ornek seciciyi bir kez kullanin; birkac kare gezin.

**7. Hukum.**

> "mAP50 ve precision esigi asan bir dusus gosteriyor; digerleri gurultunun
> icinde. Bu yuzden 'guclu' etiketi aldi — birden fazla metrikte esigi asan
> bir dusus var."

### Sorulursa acilacaklar

- **Etki haritasi** (Genel Bakis'ta): butun kosular tek karede.
- **"Gurultu tabani olculunce ne degisti"**: n=1'den n=3'e cikilinca
  esiklerin nasil buyudugu ve hangi iddialarin geri cekildigi.
- **Alt grup gurultu bandi**: bir grubun band genisligi orneklem
  buyuklugune indirgenemez; `termal` grubu 858 bbox tasir ama bandi
  `hituav`in (2.165 bbox) bandindan on kat genistir.

---

## 5. Hata Analizi — hata neye benziyor

**Hedef:** metriklerin arkasindaki gercek goruntuleri gostermek. Kisa
tutun, bes dakika yeter.

> "Metrik bir ozettir; hatanin neye benzedigini soylemez. Burada her
> kosunun en sorunlu kareleri siralaniyor ve saglikli modelin ayni kareyi
> nasil gordugu yanina konuyor."

Secicide artik her kosu adiyla gorunuyor. Bir bozulma kosusu secin, birkac
kare gezin. Her karenin altindaki aciklama **tamamen turetilmistir**:
gercek nesne sayisi etiket dosyasindan, bulunan ve kacirilan sayilari olcum
kaydindan gelir.

Bir kez saglikli referansi secip sunu soyleyin:

> "Saglikli modelde de hatalar var. Bu bozulma degil, **verinin kendi
> zorlugu**. Senaryolar bu tabanin uzerine eklenen etkiyi olcuyor."

---

## 6. LLM Teshis Ajani — projenin asil sorusu

**Hedef:** korlugun yapisal oldugunu gostermek ve ajani calistirmadan once
ne olacagini izleyiciye tahmin ettirmek.

### Once tekrarli deneyin sonucu

Sayfanin ustundeki panel.

> "Bu, asagidaki eski denemeden **ayri** bir deney. 13 kosu, her biri uc kez
> soruldu: 39 gozlem. Her gozlem model adini, arac surumunu, kodun o anki
> halini, ham cevabi ve **her arac cagrisinin cevabinin anlik kaydini**
> tasiyor."

Uc satirlik rol tablosunu okuyun ve **toplamayin**:

> "Bu uc satir toplanmaz. Kontrol kosulari 'uyduruyor mu', bozulma
> senaryolari 'nedeni bulabiliyor mu' sorusunu olcer. Tek bir ortalama
> ikisini de yaniltir."

Iki sayiyi vurgulayin:

> "Kontrol kosularinda dokuz gozlemin dokuzunda da uydurmadi. Ve on uc
> kosunun on ucu de uc tekrarinda ayni hukmu verdi — kelimeler degisiyor,
> hukum degismiyor."

### Sonra tek bir kosuyu bastan sona

Bir kosu secin ve **gercegi acmadan** ilerleyin:

> "Ajana giden tek sey su: anonim bir kimlik ve olcum araclari. Senaryo
> adini, ne yaptigimi ve cevap anahtarini gormuyor. 'Ajana ne gidiyor, ne
> gitmiyor' tablosu bunu satir satir gosteriyor."

Arac cagrilarini gosterin:

> "Hangi kaniti isteyecegine kendisi karar verdi. Ben araclari onceden
> calistirip cevabina eklemedim — sirayla bunlari cagirdi."

Teshisi ve kanitlari okuyun, sonra **izleyiciye sorun**:

> "Sizce hangi bozulmayi uyguladim?"

Sonra "Gercek senaryoyu ve puani goster" dugmesine basin.

### Durustluk notu

> "Guven degeri ajanin kendi beyanidir, kalibre edilmis bir olasilik
> degildir. Dogrulukla iliskisini olcmedim."

---

## 7. Sonuclar ve Sinirlamalar — neyi bilmiyoruz

**Hedef:** bulgulari ozetlemek ve **sinirlari kendiniz soylemek**. Bir tez
savunmasinda en guclu bolum budur.

Alti bilimsel sonuc karti duruyor; ucunu okuyun:

1. Farkli arizalar farkli metrik imzasi birakiyor.
2. Toplam mAP yerel bir cokusu tamamen gizleyebiliyor.
3. Gurultu olculmeden "etki" iddiasi kurulamaz.

Hipotez tablosunu gosterin ve altindaki uyariyi **kendiniz okuyun**:

> "Bu tablo bir hipotez testi degildir ve oyle oldugunu iddia etmiyor.
> Beklenti sutunu serbest metin; makine tarafindan ayristirilamaz. Bu
> yuzden hukum sutunu yalnizca **olculen seyi** adlandiriyor: etki gurultu
> esigini asiyor mu, kac metrikte ve hangi yonde."

Sonra "Neyi HENUZ soyleyemiyoruz" bolumune gecin ve **hizli okumayin**:

> "Senaryo basina tek egitim kosusu var; senaryo metriklerine guven araligi
> veremiyorum. Gurultu tabani uc bozulmasiz kosudan geliyor, az gozlemle
> band gercek yayilimi oldugundan kucuk gosterir. Referans tek bir kosu ve
> saglikli kosularin en zayifi. Nadir siniflarda ornek yetersiz. Final test
> seti hic kullanilmadi — yani buradaki hicbir sayi 'nihai test
> performansi' degil."

Kapanis cumlesi:

> "Bu bolumun amaci bulgulari zayiflatmak degil. Hangilerinin ne kadar
> dayanikli oldugunu acikca soylemek. Gurultu tabanini olctukten sonra bir
> dizi iddiam geri cekildi ve bir senaryo bulgu olmaktan cikti — bu,
> olcumun calistiginin kanitidir."

---

## Sunum sirasinda dikkat

- **Ekrandaki sayiyi okuyun, ezberden sayi soylemeyin.** Bu metin bilerek
  sayi tasimiyor; iki istisnasi teste baglidir.
- **Bir sayfada takilirsaniz** "bunu Sonuclar sayfasinda gosterecegim" deyip
  devam edin. Butun sayfalar birbirine baglantilidir.
- **En sik yanlis anlasilan nokta:** ajanin goruntulere baktiginin
  sanilmasi. Genel Bakis'ta bir kez, Ajan sayfasinda bir kez daha soyleyin.
- **Canli ajan** ucretsiz katman sinirlarina tabidir (20 istek/gun). Kayitli
  kosu her zaman calisir; sunumun guvenli yolu odur.

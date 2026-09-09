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

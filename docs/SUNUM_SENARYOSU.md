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

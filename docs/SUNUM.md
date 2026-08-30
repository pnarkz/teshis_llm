# Sunum ve Teknik Olmayan Anlatim

Projeyi AI uzmani olmayan bir okuyucuya veya mentore anlatmak icin.

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

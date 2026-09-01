# Sunum ve Teknik Olmayan Anlatim

Projeyi AI uzmani olmayan bir okuyucuya veya mentore anlatmak icin.
Sayisal ayrinti icin: [BULGULAR.md](BULGULAR.md).

---

## 1. Proje bir cumlede

Termal drone goruntuleriyle calisan bir nesne tespit modelini **kasitli olarak
bozup**, bozulmanin metriklere nasil yansidigini olcuyoruz; sonra bir yapay
zeka ajanina bu olcumleri verip **nedeni bulup bulamadigini** siniyoruz.

Amac daha iyi bir model egitmek degil. Amac, bir modelin **hangi kosullarda
guvenilirligini kaybettigini** ve bunun **olculebilir bir izi olup olmadigini**
gostermek.

---

## 2. Ne bulduk

### 2.1 Bozulmanin turu metrik imzasindan okunabiliyor

En net sonuc bu. Farkli arizalar farkli imzalar birakiyor:

| Ariza | Precision | Recall | Ayirt edici imza |
|---|---|---|---|
| Cikarim cozunurlugu yanlis (E4) | **degismiyor** | **cokuyor** | Model buldugunu dogru buluyor, ama bulamiyor |
| Etiketler eksik/karisik (D2b, D3) | **cokuyor** | degisken | Model olmayan seyi "var" diyor |
| Kucuk nesne sinyali silinmis (D4) | dusuyor | dusuyor | Kayip kucuk boyut bandinda yogunlasiyor |

Yani "model kotu calisiyor" demek yetmiyor; **precision mi recall mi
bozuldugu, arizanin turunu soyluyor.** Bir teshis sisteminin dayanabilecegi
ilk saglam zemin bu.

### 2.2 En buyuk etki bir veri hatasi degil, bir ayar hatasi

Olculen en buyuk performans kaybi bozuk etiketlerden degil, modelin
**egitildiginden farkli bir cozunurlukte calistirilmasindan** geldi:

- Egitim cozunurlugu 768'de: mAP50 **0.920**
- Ayni model 512'de calistirilinca: mAP50 **0.602**

Hicbir veri bozulmasi buna yaklasmadi (en yakini -0.032). Pratik ders:
**sahada bir modeli yanlis cozunurlukte calistirmak, veri kalitesindeki
bircok sorundan daha pahaliya mal olur.**

### 2.3 Standart raporlama bir arizayi tamamen gizleyebiliyor

En ogretici bulgu. E1 senaryosunda modeli 1000 goruntuyle 200 tur egittik ve
klasik bir **ezberleme** (asiri uyum) elde ettik: egitim hatasi 8 kat dustu,
dogrulama hatasi yukseldi.

Buna ragmen, olagan raporlama biciminde (en iyi kontrol noktasi) model
**tamamen saglikli gorundu**: referansa gore mAP50 farki yalnizca -0.001.

Sebep: model zaten egitilmis bir modelden basladigi icin **en iyi sonucu 1.
turda** verdi; sonraki 199 tur onu yalnizca bozdu ve otomatik secim o ilk
turu kaydetti. Ariza yalnizca **egitim egrisinde** ve son kontrol noktasinda
gorunuyor (mAP50 -0.088).

> **Ders:** ezberlemeyi metrikten degil egrinin sekli'nden anlarsiniz.
> Yalnizca son skora bakan bir denetim bunu kaciririr.

### 2.4 "Fark var" demeden once gurultuyu olcmek gerekiyor

Ayni veriyle, ayni ayarlarla, **yalnizca rastgelelik tohumu degistirilerek**
egitilen iki model arasinda bile fark cikiyor:

- precision farki: 0.018
- recall farki: 0.011
- `insan` sinifi AP farki: **0.031**

Bu, bazi senaryolarin "etkisinden" buyuk. Nitekim bir senaryomuz (D6b) bu
olcum sonrasi **bulgu olmaktan cikti**: farki, saf rastgelelikten ayirt
edilemiyor.

> **Ders:** kontrol kosusu olmadan yapilan her "etki" iddiasi risklidir.

### 2.5 Ajan: sorun uydurmuyor, ama sorunun adini koymakta zayif

11 kosuluk denemede ajanin davranisi:

| Durum | Sonuc |
|---|---|
| Hicbir bozulma yokken (2 kosu) | **2/2 dogru** — sorun uydurmadi |
| Bozulma var, tespit edilebilir (6 kosu) | 3.5/6 dogru nedeni buldu |

Baskin hata turu **uydurmak degil**, yanlis nedene atfetmek ve kacirmak.
Pratikte bu tersinden iyi bir profil: uydurulmus bir teshis bosa is
yaptirir, yanlis adlandirilmis bir teshis en azindan dogru yere baktirir.

Bir de olumsuz sonuc: ajana **arac vermek** (kendi kanitini secmesine izin
vermek) teshis dogrulugunu olculebilir sekilde artirmadi (0.500 vs 0.444;
dokuz kosuda yarim kosu fark).

---

## 3. Neyi HENUZ soyleyemiyoruz

Bu, sunumun en onemli bolumu.

Projenin asil sorusu "bir LLM bozulmayi teshis edebilir mi?" idi.
**Bu soruyu cevaplayacak orneklem henuz yok:** tek model, kosu basina tek
deneme, 11 kosu ve yalnizca 2 saf kontrol. Olculen skor bir nokta
tahminidir; tekrar olmadigi icin guven araligi hesaplanamaz.

Ajanin "sorun uydurmama" orani icin verebilecegimiz aralik
**%0 ile %66** arasi. Isaret olumlu, kanit zayif.

Eksigi kapatan olcum belli ve suan **kosuyor**: saglikli modelin farkli
rastgelelik tohumlariyla tekrarlanmasi. Ayni kosular hem gurultu tahminini
hem kontrol grubunu buyutuyor.

---

## 4. Temel kavramlar

### Bbox nedir?

Bbox, nesneyi cevreleyen dikdortgen kutudur. Dataset'teki her etiket satiri
bir bbox'i temsil eder. "4.014 bbox" dedigimizde 4.014 fotograf degil,
fotograflardaki toplam 4.014 etiketli nesne kastedilir.

Bir fotograf birden fazla bbox icerebilir; goruntu sayisi ile bbox sayisi
ayni degildir.

### Dataset neden train, val ve test diye ayrilir?

- **Train:** Modelin ogrendigi fotograflar.
- **Val:** Egitim sirasinda ayarlari ve en iyi agirliklari secmek icin
  kullanilan kontrol grubu.
- **Test:** Tum secimler bittikten sonra gercek performansi olcmek icin
  saklanan son sinav grubu.

Test kullanilirsa ekip modeli farkinda olmadan test sonucuna gore ayarlar ve
test bagimsiz bir sinav olmaktan cikar. Bu nedenle bu projede **test final
asamasina kadar yasaktir** ve bugune kadar hic kullanilmadi.

### val_diagnostic neden var?

Mevcut val klasorunde ayni kaynaktan gelen benzer veya augment edilmis
goruntuler bulunabilir; bu, olcumu yanli hale getirir. `val_diagnostic` bunu
azaltmak icin hazirlanmis **kilitli** bir kumedir:

- Olusturulduktan sonra degistirilmez.
- Her model ayni set uzerinde olculur.
- Test seti degildir.

1.056 goruntu ve 4.014 bbox icerir.

### Metrikler nasil okunur?

**Precision:** modelin "bu nesne var" dedigi tahminlerin ne kadari dogru.
Dusukse model gereksiz kutu ciziyor demektir.

**Recall:** gercekte var olan nesnelerin ne kadari yakalandi. Dusukse model
nesne kaciriyor demektir.

**mAP:** sinif dogrulugu ile kutunun nesneyle ortusmesini birlikte olcen
ozet metrik.

mAP tek basina yeterli degildir. Bu projede her zaman sinif bazli AP, recall
ve **bbox sayisi** ile birlikte raporlanir: UAP mAP50 degeri 0.995 olsa bile
bu yalnizca 15 bbox'a dayaniyorsa belirsizlik yuksektir.

---

## 5. Neden senaryolar?

"Model neden kotu calisiyor?" sorusu gercek hayatta cevaplanamaz, cunku
ayni anda birden fazla sey yanlis olabilir. Biz tersini yapiyoruz: saglikli
bir referans egitip, sonra **her seferinde tek bir seyi** kasitli olarak
bozuyoruz.

Boylece aradaki fark tek bir nedene baglanabilir. Bunun calismasi icin
egitim ayarlarinin sabit kalmasi sarttir; proje bu ayarlari tek bir dosyada
tutar ve her kosunun ondan saptigi noktalari kaydeder.

Senaryolar iki aileye ayrilir:

- **D serisi** veriyi bozar, egitim ayarlarini sabit tutar.
- **E serisi** veriyi temiz tutar, egitim/calistirma ayarlarini bozar.

Ayrica **kontrol kosullari** vardir: hicbir sey bozulmaz, yalnizca
rastgelelik tohumu degisir. Bunlar "gurultu ne kadar" sorusunu cevaplar.

---

## 6. Ajanin rolu ve sinirlari

Ajan modelin yerine gecmez ve yeni gercekler uretmez. Gorevi:

1. Anonimlestirilmis kosu metriklerini okumak.
2. Saglikli referansla karsilastirmak.
3. Hangi sinifin, hangi boyutun, hangi kaynagin etkilendigini bulmak.
4. Arizanin turunu siniflandirmak ve **sayisal kanitla** aciklamak.
5. Kanit yetersizse bunu soylemek.

Ajan **kasitli olarak korlestirilmistir**: hangi kosunun hangi senaryo
oldugunu bilmez, kosular `kosu_01`, `kosu_02`... diye sunulur. Kaynak
gruplari bile takma adla verilir. Boylece cevabi tahmin edemez, kanittan
cikarmak zorunda kalir.

Gercek bir ajan cevabi (kontrol kosusu icin, dogru cevap "sorun yok"):

~~~text
Teshis: saglikli_model_performansi
Kanit : mAP50 0.9214, baseline 0.9200'e cok yakin (fark +0.0015)
        16-32 px kucuk nesne recall'u 0.8707 ile referansin (0.8344) uzerinde
Guven : yuksek
Sinir : UAI (17 bbox) ve UAP (15 bbox) ornek sayilari 20'nin altinda,
        bu siniflardaki oranlar genellenemez
~~~

---

## 7. Mentor icin kisa anlatim

Bu proje, termal drone nesne tespit modelinin **kontrollu kosullarda neden
bozuldugunu** olcen bir teshis altyapisidir. Once saglikli bir referans
egitilir; sonra tek bir veri veya egitim problemi kasitli olarak olusturulur;
ayni model bu kosulda tekrar egitilir ve **kilitli** bir dogrulama setinde
olculur. Sonuclar sinif, boyut ve kaynak bazinda raporlanir. Son asamada bir
LLM bu kanitlari kullanarak arizanin nedenini aciklamaya calisir.

Yaklasimin farki, daha yuksek mAP aramak yerine modelin **hangi kosullarda
guvenilirligini kaybettigini** olcmesi ve her iddiayi **kontrol kosusuyla**
sinamasidir.

Projenin bugune kadarki en degerli ciktisi tek bir sayi degil, uc gozlem:

1. Ariza turu metrik imzasindan okunabiliyor.
2. Standart raporlama (en iyi kontrol noktasi) bir arizayi tamamen
   gizleyebiliyor.
3. Kontrol kosusu olmadan yapilan "etki" iddialari, saf rastgelelikten
   ayirt edilemeyebiliyor.

Ve bir durustluk notu: ajanin teshis basarisi hakkinda kesin konusmak icin
gereken tekrar sayisina **henuz ulasilmadi**.

# Sunum Konsolu

Tamamlanmış ölçüm çıktılarını gezilebilir hale getiren Streamlit uygulaması.
Eğitim veya test çalıştırmaz; yalnızca `reports/`, `experiments/` ve
`results.csv` içindeki mevcut sonuçları okur. Tek istisna **Ajan**
bölümündeki açıkça işaretlenmiş "canlı çalıştır" düğmesidir.

## Kurulum ve çalıştırma

```bash
python -m pip install -r requirements-demo.txt
python -m streamlit run demo/app.py
```

Tema `.streamlit/config.toml` içinde **koyu** olarak sabitlenmiştir.
Görünümü izleyicinin işletim sistemi temasına bırakmak sunumda risklidir.
Renkler `stil.py` içindeki sözleşmeyle birebir aynı tutulur; ayrışırsa
Streamlit'in kendi bileşenleri (seçici, sekme, tablo) sayfadan kopuk
görünür. `tests/test_tema_ve_grafik.py` bunu korur.

Sol menüdeki sıra yalnızca bir öneridir. **Adım adım sunum akışı yoktur** —
"sunumu başlat / ileri / geri" gibi bir sihirbaz eklenmedi; gelen soruya
göre istenen bölüme doğrudan geçilir.

## Yapı

Konsol **sıra dayatmaz**. Bölümler birbirinden bağımsızdır; gelen soruya göre
istenen bölüme atlanır. Her bölüm kendi modülünde durur, böylece birini
değiştirmek diğerlerine dokunmayı gerektirmez.

```text
demo/
  app.py            yalnizca yonlendirme, tema ve kenar cubugu (~90 satir)
  stil.py           gorsel dil: renk SOZLESMESI, kart/rozet/KPI bilesenleri
  grafik.py         butun grafiklerin tek kaynagi (tema, hover, renk)
  data_loader.py    rapor okuma + ajan katmani (st.cache_data ile onbellekli)
  veri_seti.py      veri seti ve model kunyesi turetmeleri
  gorseller.py      etiketli ornek bulma + kutu cizme (+ tasinabilir set)
  ajan_katmani.py   canli ajan on kontrolu, hata siniflandirmasi
  sistem_durumu.py  kenar cubugundaki "ne var ne yok" paneli
  assets/ornekler/  taze bir klonda galeri bos kalmasin diye kucuk ornek seti
  bolumler/
    genel_bakis.py     proje, surec semasi, etki haritasi, uc ana bulgu
    veri_ve_model.py   veri seti, etiketli ornekler, saglikli referans model
    senaryolar.py      filtreli kart izgarasi + kosu basina tam analiz
    karsilastirma.py   genel tablo, coklu karsilastirma, gurultu, checkpoint
    hata_analizi.py    saglikli vs senaryo, kare basina otomatik aciklama
    ajan.py            kor teshis: kayitli / canli
    sonuclar.py        hipotezler, bilimsel sonuclar, sinirlamalar
```

## Üç tasarım kararı

**Senaryo özeti türetilir, yazılmaz.** Sayfanın beş bileşeninden dördü kaynak
dosyalardan gelir (`teshis/degerlendirme/senaryo_ozeti.py`); elle yazılan tek
alan senaryonun ne ölçtüğüdür (`senaryolar/anlatim.yaml`). Ekran, bulgunun ne
kadar sağlam olduğunu kendisi söyler: *güçlü* / *zayıf* / *gürültü içinde*.
Bu üç derece **yalnızca** kendi ölçeğinde hem referansı hem gürültü eşiği
olan koşulara verilir; diğerleri *kontrol koşusu*, *referans*, *eşlenik
ölçüm*, *eşik yok* veya *karşılaştırılamaz* olarak işaretlenir.

Önceki sürümde bu bilgi `app.py` içinde 24 girdilik elle tutulan bir sözlükte
duruyordu ve geride kalıyordu: D6a, D6b, v00n ve D1n eklendiğinde demo onları
sessizce eksik gösterdi.

**Ajan bölümünde kayıtlı mod varsayılandır.** Canlı çağrı asli parça değil,
isteğe bağlıdır. Gerekçe ölçüldü: ücretsiz katman 20 istek/gün ve 5 istek/dk
ile sınırlı, geliştirme sırasında hem kota tükendi hem 503 alındı. Kayıtlı mod
API harcamaz, her zaman çalışır ve **daha denetlenebilirdir** — ajanın gördüğü
kanıt yerelde yeniden üretilir, çünkü araçlar deterministiktir.

**Sınırlar birinci sınıf içeriktir.** "Neyi henüz söyleyemiyoruz" gizlenmez;
gürültü tabanı ölçüldükten sonra geri çekilen iddialar açıkça yazar. O sayı
da elle tutulmuyor: karşılaştırma sayfası hem eşik büyümesi tablosunu hem
zayıflayan iddialar listesini defterden türetir. Elle yazıldığı dönemde
"beş" diyordu ve E1 ile E2 atlanmıştı.

**Karşılaştırma ölçeği görünürdür.** Her koşu yalnızca dört kimlik alanı
(başlangıç modeli, değerlendirme kümesi, çıkarım çözünürlüğü, checkpoint)
kendisiyle aynı olan sağlıklı referansla karşılaştırılır
(`teshis/degerlendirme/karsilastirilabilirlik.py`). Tek referans kullanıldığı
dönemde `v00_saglikli last_pt` — hiçbir bozulma içermeyen bir koşu — "güçlü
bozulma kanıtı" olarak etiketleniyordu.

## Görsel dil

Sakin ve akademik: kırık beyaz zemin, koyu gri metin, tek vurgu rengi.
Emoji, animasyon ve pazarlama dili yoktur. Renk anlamları bütün sayfalarda
sabittir (`stil.py`):

| Renk | Anlamı |
|---|---|
| vurgu | incelenen koşu |
| nötr | referans (v00) — her grafikte aynı renk |
| uyarı | gürültü içinde kalan / zayıf kanıt |
| olumlu | gürültü eşiğini belirgin aşan kanıt |

Sayılar her zaman "değer + referansa fark" olarak verilir; her grafiğin
altında tek cümlelik okuma notu bulunur.

## Testler

```bash
python -m pytest tests/test_demo_konsol.py tests/test_demo_veri.py
```

`test_demo_konsol.py` altı bölümü ve açılır listedeki **her seçeneği**
headless render eder — sunum sırasında bir bölümün çökmesi en kötü
senaryodur. Senaryo listesi demonun kendi kaynağından alınır; `results.csv`
okumak yetmiyordu, demo ayrıca bir "Baseline" satırı ekliyor.

`test_demo_veri.py` veri katmanının sözleşmesini korur: sabit kodlu yol
haritalarının geri gelmemesi ve galeri anahtarlarının senaryo adlarıyla
eşleşmesi.

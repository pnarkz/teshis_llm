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

Tema `.streamlit/config.toml` içinde **açık** olarak sabitlenmiştir.
Görünümü izleyicinin işletim sistemi temasına bırakmak sunumda risklidir:
koyu temada metin renkleri çakışıyor ve içerik neredeyse görünmez oluyordu.

## Yapı

Konsol **sıra dayatmaz**. Bölümler birbirinden bağımsızdır; gelen soruya göre
istenen bölüme atlanır. Her bölüm kendi modülünde durur, böylece birini
değiştirmek diğerlerine dokunmayı gerektirmez.

```text
demo/
  app.py            yalnizca yonlendirme ve stil (~80 satir)
  stil.py           ortak gorsel dil; renk anlamlari her sayfada ayni
  data_loader.py    rapor okuma + ajan katmani
  bolumler/
    genel_bakis.py     proje, durum, uc ana bulgu
    tasarim.py         kontrollu deney kurgusu + NEYI SOYLEYEMIYORUZ
    senaryolar.py      kosu basina deney ozeti ve kanit
    karsilastirma.py   capraz tablo + gurultu tabani
    hata_analizi.py    siralanmis hata ornekleri
    ajan.py            kor teshis: kayitli / canli
```

## Üç tasarım kararı

**Senaryo özeti türetilir, yazılmaz.** Sayfanın beş bileşeninden dördü kaynak
dosyalardan gelir (`teshis/degerlendirme/senaryo_ozeti.py`); elle yazılan tek
alan senaryonun ne ölçtüğüdür (`senaryolar/anlatim.yaml`). Ekran, bulgunun ne
kadar sağlam olduğunu kendisi söyler: *güçlü* / *zayıf* / *gürültü içinde*.

Önceki sürümde bu bilgi `app.py` içinde 24 girdilik elle tutulan bir sözlükte
duruyordu ve geride kalıyordu: D6a, D6b, v00n ve D1n eklendiğinde demo onları
sessizce eksik gösterdi.

**Ajan bölümünde kayıtlı mod varsayılandır.** Canlı çağrı asli parça değil,
isteğe bağlıdır. Gerekçe ölçüldü: ücretsiz katman 20 istek/gün ve 5 istek/dk
ile sınırlı, geliştirme sırasında hem kota tükendi hem 503 alındı. Kayıtlı mod
API harcamaz, her zaman çalışır ve **daha denetlenebilirdir** — ajanın gördüğü
kanıt yerelde yeniden üretilir, çünkü araçlar deterministiktir.

**Sınırlar birinci sınıf içeriktir.** "Neyi henüz söyleyemiyoruz" gizlenmez;
gürültü tabanı ölçüldükten sonra beş iddianın geri çekildiği açıkça yazar.

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

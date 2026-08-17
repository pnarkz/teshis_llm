# Veri Surumleri

Her klasor bir bozulma kosusunun tam ciktisidir.

```text
v00_saglikli/
  images/{train,val,test}/
  labels/{train,val,test}/
  manifest.json
```

Yeni surumler `surum_uret.py` ile uretilir. Orijinal `dataset/` hicbir zaman
degistirilmez. `manifest.json` ajan girdisine dahil edilmez.

# Classification Kismi Icin Kisa Rehber

Bu odevde hedefimiz `is_podium` kolonunu tahmin etmek. Yani her surucu icin soru su:

`Bu pilot bu yarista podyuma girecek mi?`

## 1. Problem tipi

Bu bir `binary classification` problemidir.

- `1` = pilot podyuma girdi
- `0` = pilot podyuma girmedi

## 2. Hedef degisken

Modelde `y` olarak kullanacagin kolon:

`is_podium`

## 3. Leakage neden onemli?

Eger modelde yaristan sonra ortaya cikan bilgileri kullanirsak sonuc gercekci olmaz.

Ornek olarak su kolonlar classification icin kullanilmamali:

- `personal_best_lap_ms`
- `pit_stop_count`
- `avg_pit_dur_ms`
- `gap_to_fastest_ms`

Bunlar yaristan sonra veya yaris sirasinda netlesen bilgiler oldugu icin model "gelecegi gormus" gibi davranir.

## 4. Neden zaman bazli split yaptik?

Rastgele train-test split yapmak spor verisinde risklidir. Cunku 2024 yarislari train ve test icine karisabilir. Bu da gercek hayati yansitmaz.

Bu yuzden:

- ilk yarislar `train`
- sonraki blok `validation`
- en son yarislar `test`

seklinde ayrildi.

Bu daha dogru bir deney duzeni verir.

## 5. Kullandigimiz model

Ana model:

`XGBoostClassifier`

Secme nedeni:

- tabular veride guclu performans verir
- sinif dengesizligini iyi yonetir
- feature importance cikarmak kolaydir

## 6. Hangi metrikleri raporlayabilirsin?

Temel metrikler:

- `F1 Score`
- `Precision`
- `Recall`
- `ROC-AUC`

Projeye daha uygun ek metrikler:

- `Precision@3`
- `Recall@3`
- `Exact Top-3 Race Rate`

Buradaki mantik su: her yarista modelin en yuksek olasilik verdigi ilk 3 pilota bakiyoruz. Gercek podyumla ne kadar ortusuyor diye olcuyoruz.

## 7. Dosyalari nasil calistiracaksin?

PowerShell icinde:

```powershell
py -3.9 classification_podium.py
```

Olusan ciktilar:

- `classification_metrics.json`
- `classification_test_predictions.csv`
- `classification_feature_importance.csv`

## 8. Sunumda nasil anlatabilirsin?

Kisa bir anlatim:

`Bu calismada hedef degisken olarak is_podium kullanildi. Veri sizintisini engellemek icin yaristan sonra olusan kolonlar modele verilmedi. Veri zaman sirali oldugu icin train-validation-test ayirimi yaris bazinda kronolojik olarak yapildi. Siniflandirma modeli olarak XGBoost secildi. Model performansi F1, ROC-AUC ve race-level Precision@3 ile degerlendirildi.`

## 9. En onemli fikir

Bu projede sadece "skor yuksek cikti" demek yetmez. En kritik nokta:

`Model, yaristan once bilinebilecek bilgilerle podyumu tahmin etmeli.`

Eger bunu sunumda vurgularsan classification kismini guclu anlatmis olursun.

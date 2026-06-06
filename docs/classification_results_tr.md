# Classification Sonuclari

Bu dosya, `classification_podium.py` script'inin mevcut calisma sonucunu ozetler.

## Kisa yorum

Model, podium tahmininde guclu bir ayrim gucu gosteriyor:

- `ROC-AUC = 0.9166`
- `F1 Score = 0.6068`
- `Precision = 0.5682`
- `Recall = 0.6510`
- `Precision@3 = 0.5990`

Bu ne demek?

- Model, podiuma girecek pilotlari genel olarak iyi ayirabiliyor.
- Pozitif sinif dengesiz oldugu icin sadece accuracy'ye bakmak dogru degil.
- Bu yuzden F1 ve race-level `Precision@3` daha anlamli.

## Veri bolunmesi

Veri rastgele degil, kronolojik olarak ayrildi:

- `300` yaris train
- `64` yaris validation
- `64` yaris test

Bu yontem, gelecekteki yarislari gecmisteki verilerle tahmin etme mantigina daha uygun.

## En etkili ozellikler

Ilk calismada en guclu sinyaller:

- `quali_position`
- `grid`
- `drv_pre_position`
- `con_pre_position`

Yani model en cok:

- pilotun siralama performansi
- grid pozisyonu
- pilotun onceki puan/pozisyon durumu
- takimin onceki durumu

gibi degiskenlerden faydalaniyor.

## Sunumda kullanabilecegin cumle

`Podium tahmini icin is_podium hedef degiskeni kullanildi. Veri sizintisini onlemek amaciyla yaristan sonra olusan kolonlar modele dahil edilmedi. Veri zaman sirasina gore train-validation-test olarak ayrildi ve XGBoost ile binary classification modeli kuruldu. Model test setinde 0.9166 ROC-AUC ve 0.6068 F1 skoru elde etti. Yaris bazli Precision@3 sonucu ise yaklasik 0.60 olarak bulundu.`

## Dikkat etmen gereken nokta

En onemli savunma noktasi su:

`Biz sadece iyi skor ureten bir model degil, yaristan once bilinebilen bilgilerle anlamli tahmin yapan bir model kurmaya calistik.`

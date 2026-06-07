# 🏎️ Formula 1 Applied ML Project - Tam Sunum Metni (Speech)

**Sunum Süresi:** Yaklaşık 10-12 Dakika  
**Gereksinimler:** Slaytlar ve Arkada açık olan Streamlit Demo ekranı.

---

## 1. BÖLÜM: Projenin Amacı ve Veri Toplama
🗣️ **Konuşmacı:** Yiğit Enes Görgülü  
📺 **Ekranda:** Proje İsmi, Ekip Üyeleri ve Motivasyon Slaytı

> "Değerli hocalarım, değerli arkadaşlarım, hoş geldiniz. Biz takım olarak 'Applied Machine Learning' dersi projesi kapsamında **Formula 1 Yarış Tahminleri** üzerine bir yapay zeka hattı (pipeline) geliştirdik. Formula 1, saniyelerin binde birinin ve inanılmaz miktarda değişkenin (hava durumu, pilot formu, takım stratejisi) sonucu belirlediği çok dinamik bir spor. 
> 
> Amacımız, makine öğrenmesi algoritmalarını kullanarak yarış öncesinde iki temel soruyu cevaplamaktı: Birincisi, hangi pilot ilk 3'e girip podyuma çıkacak? İkincisi, o pilotun yarış içindeki en hızlı tur süresi saniye cinsinden ne olacak?
>
> Veri seti olarak internetteki en büyük F1 veritabanı olan Ergast API'yi temel aldık. Ek olarak, eksik olan hava durumu ve seans detaylarını tamamlamak için özel Python web scraper'lar (veri kazıyıcılar) yazdık. Böylece elimizde 2003 yılından günümüze kadar gelen, on binlerce satırlık çok zengin ve kompleks bir ham veri seti oluştu.
>
> Şimdi bu ham veriyi modellerimizin anlayabileceği bir formata nasıl çevirdiğimizi anlatması için sözü Görkem'e bırakıyorum."

---

## 2. BÖLÜM: Veri İşleme ve Feature Engineering (Özellik Çıkarımı)
🗣️ **Konuşmacı:** Görkem Özden  
📺 **Ekranda:** Data Preprocessing, Feature Engineering Akış Şeması

> "Teşekkürler Yiğit. Ham veriyi aldığımızda en büyük problemimiz şuydu: Modeller geçmişi kendiliğinden bilemezler. Bir pilotun o an formda olup olmadığını veya o pistte daha önce başarılı olup olmadığını modelin anlayabilmesi için **Feature Engineering (Özellik Çıkarımı)** aşamasına büyük önem verdik.
>
> Sadece pilot isimleri veya pist id'leri yerine; pilotların son 3 yarıştaki form durumları, kariyer podyum yüzdeleri ve takımların o pistteki tarihsel dominasyon oranları gibi hareketli (rolling) istatistikler ürettik. 
> Ayrıca modelin geleceği (yani yarış sonucunu) önceden görüp ezberlememesi (Data Leakage) için, yarış sırasında belli olan pit-stop sayısı veya aradaki fark gibi kolonları eğitim verisinden tamamen sildik. Verimizi kronolojik olarak böldük ve 2022 sonrasını sadece Test seti olarak ayırdık. 
> 
> Hazırladığımız bu temiz veriyi, sınıflandırma algoritmalarıyla podyum tahmini yapmak üzere Ahsen'e aktardık."

---

## 3. BÖLÜM: Classification - Podyum Tahmini (Sizin Kısmınız)
🗣️ **Konuşmacı:** Ahsen Pehlivan  
📺 **Ekranda:** Classification ROC Eğrisi, Confusion Matrix ve Feature Importance Grafikleri

> "Teşekkürler Görkem. Ben projemizin 'Sınıflandırma (Classification)' kısmını üstlendim. Hedefimiz ikili bir sınıflandırmaydı (Binary Classification): *Pilot podyuma çıkacak mı (1), çıkmayacak mı (0)?* 
>
> Bu aşamada karşılaştığımız en büyük zorluk **Sınıf Dengesizliği (Class Imbalance)** problemiydi. Her yarışta 20 pilot yarışıyor ancak sadece 3'ü podyuma çıkıyor. Yani verimizin büyük çoğunluğu '0'lardan oluşuyordu. Bunu çözmek için modellerimizde (XGBoost ve LightGBM) 'scale_pos_weight' parametrelerini ayarladık.
> 
> Başarı kriteri olarak sadece 'Accuracy' oranına bakmak yanıltıcı olacağı için **ROC-AUC** ve **Precision@3** (En iyi 3 adayımızın gerçekten podyuma çıkma oranı) metriklerine odaklandık. Ekranda gördüğünüz ROC eğrimizde **0.92** gibi muazzam bir AUC skoruna ulaştık. Feature Importance grafiğimize baktığımızda ise, podyumu belirleyen en önemli faktörlerin beklendiği üzere Sıralama Turları (Qualifying Position) ve Grid sırası olduğunu, ancak pilotun geçmiş formunun da çok kritik bir rol oynadığını tespit ettik.
>
> Modelimizin podyuma çıkacakları tahmin etme gücünü gördük. Peki süreleri ne kadar doğru tahmin edebiliyoruz? Bunu anlatması için sözü Tayfun'a bırakıyorum."

---

## 4. BÖLÜM: Regression - En Hızlı Tur Süresi Tahmini
🗣️ **Konuşmacı:** Tayfun Özgür  
📺 **Ekranda:** Regression Residual Plot, Actual vs Predicted Grafiği

> "Teşekkürler Ahsen. Ben de projenin Regresyon (Regression) tarafıyla ilgilendim. Buradaki hedefimiz, pilotun yarış içinde atacağı 'En Hızlı Tur' süresini milisaniye cinsinden tahmin etmekti. 
> 
> Ancak doğrudan süreyi tahmin etmeye çalışmak modelleri yanılttı. Bu yüzden farklı bir strateji izledik: Modellerimizden süreyi sıfırdan bulmasını değil; pilotun sıralama turlarındaki derecesinin **üzerine ne kadar süre ekleyeceğini (Residual/Fark)** tahmin etmesini istedik. Daha sonra bu farkı, sıralama süresiyle toplayarak gerçek tahmine ulaştık.
>
> Modellerimizi (Random Forest ve LightGBM) eğitip kıyasladık. Sonuç olarak LightGBM modeli bize en düşük hata payını (Mean Absolute Error) verdi. Ekranda gördüğünüz Gerçek vs Tahmin edilen tur süreleri grafiğindeki noktaların çapraz çizgi etrafında yoğunlaşması, modelimizin gerçek tur sürelerini sadece saniyelerin küsuratıyla şaştığını ve çok isabetli çalıştığını gösteriyor."

---

## 5. BÖLÜM: Canlı Demo (Web Uygulaması Üzerinden)
🗣️ **Konuşmacı:** Ortak (Tavsiyem bu kısmı Ahsen veya Tayfun'un sunmasıdır)  
📺 **Ekranda:** Streamlit Uygulaması (`streamlit run app.py` ekranı)

> "Son olarak, geliştirdiğimiz bu karmaşık makine öğrenmesi pipeline'ını gerçek hayatta kullanılabilir bir ürüne dönüştürdük. Ekranda gördüğünüz bu arayüz, bizim yazdığımız ve arka planda az önce anlattığımız eğitilmiş `.joblib` modellerini anlık olarak çalıştıran bir **Canlı Demo** ekranıdır.
> 
> *(Bu sırada arayüzden 2022'den bir yarış ve örneğin Charles Leclerc seçilir)*
> 
> Seçtiğimiz yarışa ve pilota ait güncel veriler saniyeler içinde modellerimize gönderiliyor. 'Tahmin Et' butonuna bastığımızda, Classification modelimiz pilotun podyuma çıkıp çıkamayacağını canlı olarak hesaplıyor. Hemen altında ise Regression modelimiz pilotun o gün atacağı en hızlı tur süresini tahmin ediyor. Gördüğünüz gibi modellerimiz önceden hesaplanmış verileri okumuyor, tamamen o an gönderilen veriyi işleyerek gerçek bir yapay zeka asistanı gibi çalışıyor."
> 
> "Bizi dinlediğiniz için takımım adına teşekkür ederim. Sorularınız varsa cevaplamaktan memnuniyet duyarız."

---
*İpuçları:* Sunumdan bir gün önce Discord'da veya yan yana gelip bu metni sesli olarak provalayın. Geçişlerin pürüzsüz olması hocalara takım olarak uyum içinde çalıştığınız mesajını verir. Başarılar! 🚀

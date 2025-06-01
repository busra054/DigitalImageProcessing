# DigitalImageProcessing
Geliştirilen görüntü işleme arayüzündeki tasarımda öncelikle load image butonuna basılarak kendi bilgisayarınızda
kayıtlı olan bir resim seçilir. Test edilebilmesi için proje dizinine gerekli resimler eklenmiştir.
Ardından aşağıda bulunan butonlara basılarak görüntü üzerinde işlemler yapılır. 
Bu butonların kullanımı şu şekildedir:

-> RGB,HSV,GrayScale,Blur, Edge Detect, Histogram butonları üzerinde de yazdığı gibi bu işlemleri arka planda 
yüklenen resim üzerinde gerçekleştirerek işlenmiş resmi yine resmin bulunduğu label içinde yayınlar.

-> Büyüt, Küçült fonksiyonları ise resmi girilen bir yüzdelik oranına göre büyüterek ya da küçülterek proje dizini içerisindeki processedImage klasörü 
içine kaydeder. Bu işlem yaklaşık 15 saniye kadar sürebilir beklenmelidir. Döndür butonu ise belli bir açı değeri alarak resmi döndürür, Zoom in ve zoom out butonları ise resmi 2x 
kadar büyütüp küçültürler ve klasöre kaydederler. 


Sağ tarafta bulunan butonlar ise:
Resimde kontrast arttırmak amacı ile standart, shifted, steep sigmoid fonksiyonlarını resme uygulayıp üzerindeki label içerisinde göstermeyi
sağlar.
Sinus-S-Curve ise s curve fonksiyonuna sinus fonksiyonu uygulanarak tarafımızca yazılmış ve görüntüye uygulanabilen yöntemdir. 
Hough transform ise görüntüdeki çizgileri tespit etmek ve verilecek bir yüz resminde gözleri tespit etme işimi gerçekleştirir. 
Detect lines on highway yoldaki çizgileri, Detect Human eye ise insan gözünü tespit etmek içindir.

Fakat kod içerisinde verilmiş bazı yollar kendi bilgisayarımızda bulunan yollardır bu değiştirilebilir. 

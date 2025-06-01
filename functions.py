from PyQt5 import QtWidgets, QtGui, QtCore
from odev1UI import Ui_Form
import cv2
import numpy as np  #eski ödev için eklendi 
import os
import datetime

from odev2 import ImageProcessor
from Final import Final  # final ödevi için kullanıldı.

class ImageProcessingApp(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.original_image = None
        self.processed_image = None
        self.current_image = None 
        self.final = Final()
        self.loadImageButton.clicked.connect(self.load_image) # resmi yükle
        self.GrayscaleConverterButton.clicked.connect(self.convert_to_grayscale) # butona basınca grayscale e çeviriyor.
        self.HSVConverterButton.clicked.connect(self.convert_to_hsv) # hsv ye çeviriyor
        self.EdgeDetectButton.clicked.connect(self.detect_edges)# kenarları tespit ediyor
        self.BlurAddButton.clicked.connect(self.apply_blur) # blur ekliyor
        self.HistogramButton.clicked.connect(self.show_histogram) # histogram çiziyor
        self.RgbConverterButton.clicked.connect(self.convert_to_rgb) # rgb ye çeviriyor

        self.output_dir = "/home/busra/Downloads/221229054_BüşraÖnal/processedImages" # bu yola işlenen resimleri ekliyor.
        os.makedirs(self.output_dir, exist_ok=True)

        # ödev2 için eklenen buton bağlantıları 
        self.buyut_PushButton.clicked.connect(self.buyut) # verilen yüzdelik değerine göre butona baınca büyütüp dosya yoluna kaydediyor işlenmiş resmi
        self.kucult_PushButton.clicked.connect(self.kucult)# verilen yüzdelik değerine göre butona baınca küçültüp dosya yoluna kaydediyor işlenmiş resmi
        self.dondurPushButton.clicked.connect(self.dondur)# verilen açı değerine göre butona baınca döndürüp dosya yoluna kaydediyor işlenmiş resmi
        self.zoomIn_PushButton_2.clicked.connect(self.zoom_in)# 2x zoom in yapıp dosya yoluna kaydediyor işlenmiş resmi
        self.zoomOut_PushButton.clicked.connect(self.zoom_out)# 2x zoom out yapıp dosya yoluna kaydediyor işlenmiş resmi



        self.standartSigmoidPushButton.clicked.connect(self.apply_standard_sigmoid)
        self.shiftedSigmoid_PushButton.clicked.connect(self.apply_shifted_sigmoid)
        self.steepSigmoid_PushButton.clicked.connect(self.apply_steep_sigmoid)
        self.sinusSigmoid_PushButton.clicked.connect(self.apply_sinusoidal_curve)
        self.DetectLines_PushButton.clicked.connect(self.detect_lines)
        self.DetectHumanEyePushButton.clicked.connect(self.detect_eyes)
        self.DeblurringPushButton.clicked.connect(self.deblur_image)

        self.CountAndExtractButton.clicked.connect(self.count_and_extract)
        
    def apply_standard_sigmoid(self):
        if self.current_image is None:
            QtWidgets.QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim yükleyin!")
            return
            
        processed = self.final.standard_sigmoid(self.current_image)
        self.display_image(processed, self.NewImage)

    def apply_shifted_sigmoid(self):
        if self.current_image is None:
            QtWidgets.QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim yükleyin!")
            return
            
        processed = self.final.shifted_sigmoid(self.current_image)
        self.display_image(processed, self.NewImage)

    def apply_steep_sigmoid(self):
        if self.current_image is None:
            QtWidgets.QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim yükleyin!")
            return
            
        processed = self.final.steep_sigmoid(self.current_image)
        self.display_image(processed, self.NewImage)

    def apply_sinusoidal_curve(self):
        if self.current_image is None:
            QtWidgets.QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim yükleyin!")
            return
            
        processed = self.final.sinusoidal_curve(self.current_image)
        self.display_image(processed, self.NewImage)

    def detect_lines(self):
        if self.current_image is None:
            QtWidgets.QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim yükleyin!")
            return
            
        processed = self.final.detect_lines(self.current_image)
        self.display_image(processed, self.NewImage)

    def detect_eyes(self):
        if self.current_image is None:
            QtWidgets.QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim yükleyin!")
            return
            
        processed = self.final.detect_eyes(self.current_image)
        self.display_image(processed, self.NewImage)

    def deblur_image(self):
        if self.current_image is None:
            QtWidgets.QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim yükleyin!")
            return
            
        processed = self.final.deblur_image(self.current_image)
        self.display_image(processed, self.NewImage)

    def count_and_extract(self):
        if self.current_image is None:
            QtWidgets.QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim yükleyin!")
            return

        # ExcelResult klasörü
        excel_parent = "/home/busra/Desktop/Görüntü İşleme 1. Basamak/221229054_BüşraÖnal (1)/221229054_BüşraÖnal/ExcelResult"

        processed, excel_file = self.final.count_and_extract(self.current_image, excel_parent)
        self.display_image(processed, self.NewImage)

        if excel_file:
            QtWidgets.QMessageBox.information(
                self,
                "Bilgi",
                f"Nesne analizi tamamlandı!\nExcel dosyası kaydedildi:\n{excel_file}",
            )
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Uyarı",
                "Koyu yeşil bölge bulunamadı veya çok küçük objeler vardı!",
            )


    def buyut(self): # % yüzdelik derecesi verildikten sonra bu değere göre büyütüyor.
        try:
            self._validate_current_image()
            text = self.buyutmeLineEdit.text().strip()
            if not text:
                raise ValueError("Lütfen bir yüzde değeri giriniz.")
            pct = float(text)
            if pct <= 0:
                raise ValueError("Yüzde pozitif olmalıdır.")
            factor = pct / 100.0

            # Büyütme sınırlarını belirle
            MAX_PCT = 300.0  # maksimum %300
            if pct > MAX_PCT:
                raise ValueError(f"Yüzde en fazla {MAX_PCT}% olabilir.")

            # Yeni boyutları hesapla ve sınır kontrolü yap
            max_width, max_height = 8000, 8000
            h, w = self.current_image.shape[:2]
            new_width = int(w * factor)
            new_height = int(h * factor)
            if new_width > max_width or new_height > max_height:
                raise ValueError(f"Oluşacak görüntü çok büyük: {new_width}x{new_height} piksel.")

            processor = ImageProcessor(self.current_image)
            processed = processor.resize(factor, 'bilinear')
            if processed is None:
                raise ValueError("Büyütme işlemi başarısız oldu.")

            save_path = self.save_processed_image(processed)
            QtWidgets.QMessageBox.information(
                self, "Başarılı",
                f"Resim %{pct} büyütüldü!\nKaydedilen: {save_path}"
            )

        except ValueError as ve:
            QtWidgets.QMessageBox.warning(self, "Uyarı", str(ve))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hata", str(e))

    def kucult(self): # verilen yüzdelik değerine göre küçültme yapıyor.
        try:
            self._validate_current_image()
            text = self.kucultmeLineEdit.text().strip()
            if not text:
                raise ValueError("Lütfen bir yüzde değeri giriniz.")
            pct = float(text)
            if pct <= 0:
                raise ValueError("Yüzde pozitif olmalıdır.")
            factor = pct / 100.0

            # Küçültme sınırlarını belirle
            MIN_PCT = 10.0  # minimum %10
            if pct < MIN_PCT:
                raise ValueError(f"Yüzde en az {MIN_PCT}% olabilir.")

            processor = ImageProcessor(self.current_image)
            processed = processor.resize(factor, 'average')
            if processed is None:
                raise ValueError("Küçültme işlemi başarısız oldu.")

            save_path = self.save_processed_image(processed)
            QtWidgets.QMessageBox.information(
                self, "Başarılı",
                f"Resim %{pct} küçültüldü!\nKaydedilen: {save_path}"
            )

        except ValueError as ve:
            QtWidgets.QMessageBox.warning(self, "Uyarı", str(ve))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hata", str(e))

    def dondur(self): # açı değeri alıp döndürme yapıyor. Biraz uzun sürüyor işle 20-40 sanniye kadar ebklemek gerekiyor butona bastıktan sonra.
        try:
            self._validate_current_image()
            text = self.DondurmeAcisiLineEdit.text().strip()
            if not text:
                raise ValueError("Lütfen bir açı değeri giriniz.")
            angle = float(text)

            processor = ImageProcessor(self.current_image)
            processed = processor.rotate(angle, 'bilinear')
            if processed is None:
                raise ValueError("Döndürme işlemi başarısız oldu.")

            save_path = self.save_processed_image(processed)
            QtWidgets.QMessageBox.information(
                self, "Başarılı",
                f"Resim {angle}° döndürüldü!\nKaydedilen: {save_path}"
            )

        except ValueError as ve:
            QtWidgets.QMessageBox.warning(self, "Uyarı", str(ve))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hata", str(e))

    def zoom_in(self): # değer almadan zoom in deyince 2x büyütüyor.Biraz uzun sürüyor işle 20-40 sanniye kadar ebklemek gerekiyor butona bastıktan sonra.
        try:
            self._validate_current_image()
            factor = 2.0
            processor = ImageProcessor(self.current_image)
            processed = processor.zoom_in(factor)
            if processed is None:
                raise ValueError("Zoom In işlemi başarısız oldu.")

            save_path = self.save_processed_image(processed)
            QtWidgets.QMessageBox.information(
                self, "Başarılı",
                f"Zoom In (2x) uygulandı!\nKaydedilen: {save_path}"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hata", str(e))

    def zoom_out(self): # 2x resmi zoom out yapıyor. biraz uzun sürüyor.
        try:
            self._validate_current_image()
            factor = 2.0
            processor = ImageProcessor(self.current_image)
            processed = processor.zoom_out(factor)
            if processed is None:
                raise ValueError("Zoom Out işlemi başarısız oldu.")

            save_path = self.save_processed_image(processed)
            QtWidgets.QMessageBox.information(
                self, "Başarılı",
                f"Zoom Out (2x) uygulandı!\nKaydedilen: {save_path}"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hata", str(e))

    def _validate_current_image(self):
        if self.current_image is None:
            raise ValueError("Lütfen önce bir resim yükleyin!")

    def save_processed_image(self, image): # işlemler yapıldıktan sonra resmi kaydediyor.
        if image is None:
            raise ValueError("İşlenmiş görüntü boş (None).")
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"processed_{timestamp}.png"
        save_path = os.path.join(self.output_dir, filename)
        cv_image = ImageProcessor.python_array_to_cv2(image)
        cv2.imwrite(save_path, cv_image)
        return save_path

    def update_image(self):
        if self.current_image:
            # Python listesini OpenCV formatına çeviriyor
            processor = ImageProcessor(self.current_image)
            cv2_image = processor.python_array_to_cv2(self.current_image)
            self.display_image(cv2_image, self.ProcessedImageButton)
        
    def load_image(self): # seçilen resmi labele yükleme yapıyor.
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Resim Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.lineEditImagePath.setText(file_path)
            self.original_image = cv2.imread(file_path)
            self.current_image = self.original_image  
            self.display_image(self.original_image, self.ProcessedImageButton)       
 
    def display_image(self, image, label): # resmi göstermek için kullanılıyor
        if isinstance(image, list):
            # Python listesini numpy array'e çevir
            processor = ImageProcessor(self.current_image)
            image = processor.python_array_to_cv2(image)
            
        if len(image.shape) == 2: 
            qt_image = QtGui.QImage(
                image.data, 
                image.shape[1], 
                image.shape[0], 
                image.strides[0], 
                QtGui.QImage.Format_Grayscale8
            )
        else:  
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(
                rgb_image.data, 
                w, 
                h, 
                bytes_per_line, 
                QtGui.QImage.Format_RGB888
            )
            
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap.scaled(
            label.width(), 
            label.height(), 
            QtCore.Qt.KeepAspectRatio
        ))
        
    def convert_to_grayscale(self):
        if self.original_image is not None:
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.display_image(self.processed_image, self.ProcessedImageButton)
            
    def convert_to_hsv(self):
        if self.original_image is not None:
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
            self.display_image(self.processed_image, self.ProcessedImageButton)
            
    def detect_edges(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.processed_image = cv2.Canny(gray, 100, 200)
            self.display_image(self.processed_image, self.ProcessedImageButton)
            
    def apply_blur(self):
        if self.original_image is not None:
            self.processed_image = cv2.GaussianBlur(self.original_image, (15,15), 0)
            self.display_image(self.processed_image, self.ProcessedImageButton)
            
    def show_histogram(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            
            canvas = np.zeros((300, 256, 3), dtype=np.uint8)
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            
            for i in range(1, 256):
                cv2.line(canvas, 
                        (i-1, 300 - int(hist[i-1])),
                        (i, 300 - int(hist[i])),
                        (0, 255, 0), 1)
            
            self.display_image(canvas, self.ProcessedImageButton)
            
    def convert_to_rgb(self):
        if self.original_image is not None:
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.processed_image, self.ProcessedImageButton)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
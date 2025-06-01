import cv2
import numpy as np
import math
import datetime
import os
import pandas as pd

class Final:
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        
    def apply_contrast_enhancement(self, image, func_type):
        if image is None:
            return None
            
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        normalized = gray.astype(np.float32) / 255.0

        if func_type == "standard":
            enhanced = 1 / (1 + np.exp(-10 * (normalized - 0.5)))
        elif func_type == "shifted":
            enhanced = 1 / (1 + np.exp(-8 * (normalized - 0.3)))
        elif func_type == "steep":
            enhanced = 1 / (1 + np.exp(-15 * (normalized - 0.5)))
        elif func_type == "sinusoidal":
            enhanced = 0.5 * np.sin(np.pi * (normalized - 0.5)) + 0.5

        enhanced = (enhanced * 255).astype(np.uint8)
        return enhanced

    def standard_sigmoid(self, image):
        return self.apply_contrast_enhancement(image, "standard")

    def shifted_sigmoid(self, image):
        return self.apply_contrast_enhancement(image, "shifted")

    def steep_sigmoid(self, image):
        return self.apply_contrast_enhancement(image, "steep")

    def sinusoidal_curve(self, image):
        return self.apply_contrast_enhancement(image, "sinusoidal")


    def detect_lines(self, image):
        """
        1) Griye çevir ve Gaussian Blur uygula
        2) Canny kenar algılama
        3) Morpholojik closing (dilate -> erode) ile çizgilere kalınlık ekleme
        4) cv2.HoughLinesP ile çizgileri al, parametreleri ayarla
        5) Bulunan çizgileri orijinal resme çiz
        """

        if image is None:
            return None

        # 1. Adım: Griye çevir + Gaussian Blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Adım: Canny kenar algılama 
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        # 3. Adım: Morpholojik closing (dilate sonra erode)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 4. Adım: HoughLinesP ile parametreleri sıkılaştırarak çizgi tespiti
        lines = cv2.HoughLinesP(
            closed,
            rho=1,
            theta=np.pi/180,
            threshold=80,
            minLineLength=100,
            maxLineGap=15
        )

        # Orijinal renkli kopya
        result = image.copy()

        # 5. Adım: Bulunan çizgileri çiz
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Kırmızı renkli kalın çizgi
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return result

    def detect_eyes(self, image):
        """
        Haar Cascade kullanarak insan gözü tespiti:
        1) Griye çevir
        2) Göz Cascade ile gözleri al
        3) Göz çevresine daireler ve nokta çiz
        """

        if image is None:
            return None

        # 1. Adım: Griye çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Adım: Haar cascade ile göz tespiti
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        result = image.copy()

        # 3. Adım: Bulunan her göz için dikdörtgen ve merkez nokta çiz
        for (ex, ey, ew, eh) in eyes:
            center = (ex + ew // 2, ey + eh // 2)
            radius = int(round((ew + eh) * 0.25))
            # Gözün etrafına yeşil daire
            cv2.circle(result, center, radius, (0, 255, 0), 2)
            # Göz merkezine kırmızı nokta
            cv2.circle(result, center, 2, (0, 0, 255), 3)

        return result

    def deblur_image(self, image):
        """
        Basit Unsharp Masking tekniği ile hareketli blur'u azaltır,
        """
        if image is None:
            return None

        # 1) Hafif bir Gaussian Blur al (sigmaX,sigmaY = 3.0) 
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3.0, sigmaY=3.0)

        # 2) Mask = Orijinal - Blurlanmış
        mask = cv2.subtract(image, blurred)

        # 3) Unsharp Masking: Orijinal + strength * Mask
        strength = 1.2  
        sharpened = cv2.addWeighted(image, 1.0, mask, strength, 0)

        # 4) Geri döndür
        return sharpened
    

    def count_and_extract(self, image, output_dir):
        
      #  1) Gri/BGR resmi HSV'ye çevirir.
      #  2) 'Parlak/orta yeşil' ile 'koyu yeşil' bölgeler için iki ayrı HSV aralığı tanımlar.
       # 3) Bu iki maskeyi birleştirir.
    #    4) Mask üzerinde morfolojik açma (opening) uygular.
     #   5) Konturları bulur, alanı <100 px² olanları atlar.
     #   6) Her kontur için bounding box, merkez, w, h, diagonal, energy, entropy, mean, median hesaplar.
    #    7) Sonuç resmine dikdörtgen ve indeks numarası ekler.
     #   8) Eğer en az bir obje bulunduysa OutputDir (örneğin ExcelResult) içinde Excel dosyası oluşturur.
        

        if image is None:
            return None, None

        # 1) HSV'ye çevir
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 2) 'Parlak/orta yeşil' için HSV aralığı (V yüksek)
        lower_bright = np.array([35, 100, 100])
        upper_bright = np.array([85, 255, 255])
        mask_bright = cv2.inRange(hsv, lower_bright, upper_bright)

        #    'Koyu yeşil' için HSV aralığı (V düşük)
        lower_dark = np.array([35, 100,  20])
        upper_dark = np.array([85, 255, 100])
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

        # 3) İki maskeyi birleştir
        mask = cv2.bitwise_or(mask_bright, mask_dark)

        # 4) Morfolojik açma (Opening: önce erode, sonra dilate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)


        # 5) Konturları bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = image.copy()
        table_data = []

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 100:  # 100 px² altındaki gürültüleri atla
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w // 2
            center_y = y + h // 2
            diagonal = math.sqrt(w**2 + h**2)

            roi = image[y : y + h, x : x + w]
            if roi.size == 0:
                continue

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Energy hesaplama
            energy = np.sum((gray_roi.astype(float) ** 2)) / (w * h)

            # Entropy hesaplama
            hist = cv2.calcHist([gray_roi], [0], None, [256], [0, 256])
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))

            mean_val = np.mean(gray_roi)
            median_val = np.median(gray_roi)

            table_data.append([
                i + 1,                     # No
                f"{center_x},{center_y}",  # Center
                w,                         # Length
                h,                         # Width
                diagonal,                  # Diagonal
                energy,                    # Energy
                entropy,                   # Entropy
                mean_val,                  # Mean
                median_val                 # Median
            ])

            # 6) Sonuç resmine dikdörtgen ve numara ekle
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                result,
                str(i + 1),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        # 7) Eğer en az bir obje bulunduysa, ExcelResult klasörüne kaydet
        excel_file = None
        if table_data:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            excel_file = os.path.join(output_dir, f"objects_{timestamp}.xlsx")

            df = pd.DataFrame(
                table_data,
                columns=[
                    "No",
                    "Center",
                    "Length",
                    "Width",
                    "Diagonal",
                    "Energy",
                    "Entropy",
                    "Mean",
                    "Median",
                ],
            )
            df.to_excel(excel_file, index=False)

        return result, excel_file
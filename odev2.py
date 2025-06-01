import math
import cv2
import numpy as np  # burada numpy sadece opencv formatında python-array formatına çevirmek için kullanılıyor.

class ImageProcessor:
    def __init__(self, image_array):
        self.image = image_array
        self.height = len(image_array)
        self.width = len(image_array[0]) if self.height > 0 else 0
        
    def rotate(self, angle, interpolation='bilinear'):
        """Görüntüyü belirtilen açıyla döndürür"""
        radians = math.radians(angle)
        cos_a = math.cos(radians)
        sin_a = math.sin(radians)
        
        # Yeni boyutları hesapla
        new_w = int(abs(self.width * cos_a) + abs(self.height * sin_a))
        new_h = int(abs(self.width * sin_a) + abs(self.height * cos_a))
        
        rotated = [[(0, 0, 0) for _ in range(new_w)] for _ in range(new_h)]
        
        # Merkezler
        cx, cy = self.width / 2.0, self.height / 2.0
        ncx, ncy = new_w / 2.0, new_h / 2.0
        
        for y in range(new_h):
            for x in range(new_w):
                # Ters dönüşüm koordinatları
                x_src = (x - ncx) * cos_a + (y - ncy) * sin_a + cx
                y_src = -(x - ncx) * sin_a + (y - ncy) * cos_a + cy
                
                # Kaynak koordinatlar geçerliyse interpolasyon uygula
                if 0 <= x_src < self.width and 0 <= y_src < self.height:
                    if interpolation == 'nearest':
                        rotated[y][x] = self.nearest_interpolation(x_src, y_src)
                    else:
                        rotated[y][x] = self.bilinear_interpolation(x_src, y_src)
        
        return rotated

    def resize(self, scale_factor, interpolation='bilinear'):
        """Görüntüyü yeniden boyutlandırır"""
        new_w = int(self.width * scale_factor)
        new_h = int(self.height * scale_factor)
        resized = [[(0, 0, 0) for _ in range(new_w)] for _ in range(new_h)]
        
        for y in range(new_h):
            for x in range(new_w):
                x_src = x / scale_factor
                y_src = y / scale_factor
                
                if interpolation == 'nearest':
                    resized[y][x] = self.nearest_interpolation(x_src, y_src)
                elif interpolation == 'bilinear':
                    resized[y][x] = self.bilinear_interpolation(x_src, y_src)
                else:
                    resized[y][x] = self.average_interpolation(x_src, y_src, scale_factor)
        return resized

    def nearest_interpolation(self, x, y):
        """En yakın komşu interpolasyonu"""
        xr = int(round(x))
        yr = int(round(y))
        # sınırları aşmaması için clamp
        xr = max(0, min(xr, self.width - 1))
        yr = max(0, min(yr, self.height - 1))
        return self.image[yr][xr]

    def bilinear_interpolation(self, x, y):
        """Bilinear interpolasyon"""
        # koordinatları clamp
        x = max(0.0, min(x, self.width - 1))
        y = max(0.0, min(y, self.height - 1))
        x1 = int(math.floor(x))
        y1 = int(math.floor(y))
        x2 = min(x1 + 1, self.width - 1)
        y2 = min(y1 + 1, self.height - 1)
        
        wx = x - x1
        wy = y - y1
        c11 = self.image[y1][x1]
        c12 = self.image[y2][x1]
        c21 = self.image[y1][x2]
        c22 = self.image[y2][x2]
        
        top = tuple((1 - wx) * a + wx * b for a, b in zip(c11, c21))
        bot = tuple((1 - wx) * a + wx * b for a, b in zip(c12, c22))
        return tuple(int((1 - wy) * a + wy * b) for a, b in zip(top, bot))

    def average_interpolation(self, x, y, scale_factor):
        """Ortalama interpolasyon (küçültme için)"""
        size = max(1, int(1 / scale_factor))
        x0 = int(max(0, min(x * scale_factor, self.width - 1)))
        y0 = int(max(0, min(y * scale_factor, self.height - 1)))
        total = [0, 0, 0]
        count = 0
        for dy in range(size):
            for dx in range(size):
                xi = x0 + dx
                yi = y0 + dy
                if 0 <= xi < self.width and 0 <= yi < self.height:
                    pix = self.image[yi][xi]
                    total[0] += pix[0]
                    total[1] += pix[1]
                    total[2] += pix[2]
                    count += 1
        return tuple(t // count for t in total) if count > 0 else (0, 0, 0)

    @staticmethod
    def cv2_to_python_array(cv2_image):
        """OpenCV görüntüsünü Python list formatına dönüştürür"""
        if cv2_image is None or cv2_image.size == 0:
            return []
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        return [[tuple(rgb[y, x]) for x in range(w)] for y in range(h)]

    @staticmethod
    def python_array_to_cv2(python_array):
        """Python listesini OpenCV formatına dönüştürür"""
        if not python_array or not python_array[0]:
            return np.empty((0, 0, 3), dtype=np.uint8)
        h = len(python_array)
        w = len(python_array[0])
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(h):
            for i in range(w):
                r, g, b = python_array[j][i][:3]
                arr[j, i] = [b, g, r]
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def zoom_in(self, factor=2.0, interpolation='bilinear'):
        """Görüntüyü yakınlaştırır"""
        return self.resize(factor, interpolation)

    def zoom_out(self, factor=2.0, interpolation='bilinear'):
        """Görüntüyü uzaklaştırır"""
        return self.resize(1 / factor, interpolation)

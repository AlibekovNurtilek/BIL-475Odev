import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image = cv2.imread('./img/images.png')
# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Применение размытия для уменьшения шума
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Применение пороговой фильтрации
ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Нахождение контуров
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=3)

# Нахождение фона и объекта
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_bg = cv2.dilate(dilated, kernel, iterations=3)

# Приведение типов данных к uint8
sure_fg = np.uint8(sure_fg)
sure_bg = np.uint8(sure_bg)

# Обнаружение неизвестных регионов
unknown = cv2.subtract(sure_bg, sure_fg)

# Помечаем регионы
markers = np.zeros_like(gray, dtype=np.int32)  # Создание меток с типом данных int32
markers[sure_fg == 255] = 1
markers[sure_bg == 255] = 2
markers[unknown == 255] = 0

# Применяем водную сегментацию
cv2.watershed(image, markers)

# Цвет границ (красный)
image[markers == -1] = [0, 0, 255]  # Обводим границы

# Отображение результатов
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
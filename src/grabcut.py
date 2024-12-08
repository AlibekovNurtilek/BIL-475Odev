import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image = cv2.imread('./img/image4.jpg')

# Проверка, что изображение загружено
if image is None:
    print("Ошибка загрузки изображения.")
    exit()

# Создаем маску для GrabCut (значения по умолчанию: 0 - фон, 1 - передний план)
mask = np.zeros(image.shape[:2], np.uint8)

# Определяем область для GrabCut (x, y, width, height)
rect = (50, 50, image.shape[1]-50, image.shape[0]-50)  # Примерная область, которую нужно выделить

# Используем алгоритм GrabCut для сегментации
cv2.grabCut(image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

# Алгоритм с дополнительными итерациями
for i in range(5):  # Увеличиваем количество итераций
    cv2.grabCut(image, mask, rect, None, None, 1, cv2.GC_EVAL)

# Обновляем маску: пиксели, которые соответствуют фону, получают значение 0, остальные — 1
mask2 = np.copy(mask)
mask2[mask == 2] = 0
mask2[mask == 0] = 0
mask2[mask == 1] = 1
mask2[mask == 3] = 1

# Создаем результат с удаленным фоном
result = image * (mask2[:, :, np.newaxis].astype(np.uint8))

# Отображаем результаты
plt.figure(figsize=(12, 6))

# Оригинальное изображение
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Маска GrabCut
plt.subplot(1, 3, 2)
plt.imshow(mask2, cmap='gray')
plt.title('GrabCut Mask')
plt.axis('off')

# Изображение с удаленным фоном
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Foreground Extracted')
plt.axis('off')

plt.show()

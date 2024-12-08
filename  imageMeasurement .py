import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image = cv2.imread('./img/image4.jpg')
# Преобразование в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение размытия для уменьшения шума
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Пороговая фильтрация для выделения объектов
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# Нахождение контуров
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Копия изображения для отображения результатов
output_image = image.copy()

# Инициализация списка для хранения измерений
measurements = []

# Обработка каждого контура
for contour in contours:
    # Если контур имеет ненулевую площадь
    if cv2.contourArea(contour) > 100:
        # Площадь объекта
        area = cv2.contourArea(contour)
        # Периметр объекта
        perimeter = cv2.arcLength(contour, True)

        # Добавление измерений в список
        measurements.append({'area': area, 'perimeter': perimeter})

        # Рисуем контур на изображении
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

        # Добавляем текст на изображение с измерениями
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(output_image, f"Area: {area:.2f}, Perimeter: {perimeter:.2f}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Отображаем результаты
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Вывод измерений в консоль
for i, measurement in enumerate(measurements):
    print(f"Object {i+1}: Area = {measurement['area']:.2f}, Perimeter = {measurement['perimeter']:.2f}")
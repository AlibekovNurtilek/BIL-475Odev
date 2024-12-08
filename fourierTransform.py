import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image = cv2.imread('./img/image4.jpg', cv2.IMREAD_GRAYSCALE)

# Проверка, что изображение загружено
if image is None:
    print("Ошибка загрузки изображения.")
    exit()

# Применение дискретного Фурье-преобразования
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

# Сдвиг нулевой частоты в центр спектра
dft_shift = np.fft.fftshift(dft)

# Расчет амплитудного спектра
magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

# Преобразование амплитудного спектра в логарифмическую шкалу
magnitude_spectrum = np.log(magnitude_spectrum + 1)

# Восстановление изображения из частотной области (обратное Фурье-преобразование)
dft_ishift = np.fft.ifftshift(dft_shift)  # Обратный сдвиг
img_back = cv2.idft(dft_ishift)  # Обратное Фурье-преобразование
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # Амплитуда изображения

# Генерация синусоид с низкой и высокой частотой
rows, cols = image.shape
x = np.linspace(0, 2 * np.pi, cols)
y = np.linspace(0, 2 * np.pi, rows)
X, Y = np.meshgrid(x, y)

# Генерация синусоид с разными частотами
sin_low = np.sin(2 * np.pi * 0.05 * X)  # низкая частота (медленные изменения)
sin_high = np.sin(2 * np.pi * 0.5 * X)  # высокая частота (быстрые изменения)

# Комбинированная синусоида
combined = sin_low + sin_high

# Отображение результатов
plt.figure(figsize=(15, 6))

# Оригинальное изображение
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Амплитудный спектр
plt.subplot(1, 4, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

# Восстановленное изображение
plt.subplot(1, 4, 3)
plt.imshow(img_back, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')

# Синусоиды
plt.subplot(1, 4, 4)
plt.imshow(combined, cmap='gray')
plt.title('Combined Sine Waves')
plt.axis('off')

plt.show()

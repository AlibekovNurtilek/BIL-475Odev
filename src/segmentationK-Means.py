import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_kmeans(image_path, k=3):
    """
    Сегментация изображения с использованием K-means.
    :param image_path: Путь к изображению.
    :param k: Количество кластеров.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка: Невозможно загрузить изображение. Проверьте путь.")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Конвертация в RGB
    
    # Уменьшение размера изображения
    scale_percent = 50  # Процент от оригинального размера
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height))

    # Преобразование изображения в 2D массив
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Преобразование центров в целые числа (цвета кластеров)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    
    # Показ результатов
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title(f"Segmented Image (K={k})")
    plt.imshow(segmented_image)
    plt.axis("off")
    plt.show()

# Пример использования
segment_kmeans("../img/image1.jpg", k=4)

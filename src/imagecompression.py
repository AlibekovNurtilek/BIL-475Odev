from PIL import Image
import cv2

def compress_image_lossless(input_path, output_path):
    """
    Сжимает изображение без потерь, сохраняя качество.
    """
    with Image.open(input_path) as img:
        img.save(output_path, format='PNG', optimize=True)
    print(f"Сохранено без потерь: {output_path}")


def compress_image_lossy(input_path, output_path, quality=30):
    """
    Сжимает изображение с потерями, уменьшая качество.
    :param quality: Уровень качества (0-100). Чем ниже, тем больше потерь.
    """
    image = cv2.imread(input_path)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    cv2.imwrite(output_path, image, encode_params)
    print(f"Сохранено с потерями: {output_path}, качество: {quality}")


compress_image_lossy("./img/image1.jpg", "./img/output_lossy.jpg", quality=30)
compress_image_lossless("./img/image1.jpg", "./img/output_lossess.jpg")

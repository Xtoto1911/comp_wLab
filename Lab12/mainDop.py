import sys
from pathlib import Path
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

DIR = Path(sys.argv[0]).parent.resolve()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TFHUB_CACHE_DIR'] = str(DIR)
os.environ['TFHUB_DOWNLOAD_PROGRESS'] = '1'

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

def crop_center(image):
    """Обрезает изображение до квадратной формы"""
    _number, height, width, _chans = image.shape
    if width > height:
        x = (width - height) // 2
        y = 0
        size = height
    else:
        x = 0
        y = (height - width) // 2
        size = width
    image = tf.image.crop_to_bounding_box(
        image,  # изображение(я)
        y, x,  # левый верхний угол
        size, size  # размеры прямоугольника
    )
    return image

def load_image(impath: str, size=(256, 256)):
    data = np.fromfile(impath, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError()
    img = img[np.newaxis, ...]  # добавляем новую ось массива
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = crop_center(img)
    img = tf.image.resize(img, size, preserve_aspect_ratio=True)
    return img

content_mask = input('Введите маску: ')  # маска для поиска файлов с содержимым
style_image = input('Введите путь к стилю: ')
style_size = (256, 256)  # авторы советуют уменьшить стиль до 256 * 256
style = load_image(style_image, style_size)
# сглаживаем изображение стиля
style = tf.nn.avg_pool(style, ksize=[3, 3], strides=[1, 1], padding='SAME')

# Находим изображения контента с помощью pathlib
content_images = list(DIR.glob(content_mask))

for content_path in content_images:
    content = load_image(content_path)
    outputs = hub_module(content, style)
    result = outputs[0]  # outputs - список выходов
    result_image = np.squeeze(result)

    # Сохраняем стилизованное изображение
    stylized_path = DIR / f'stylized_{content_path.name}'
    cv2.imwrite(str(stylized_path), (result_image * 255).astype(np.uint8))

import sys
from pathlib import Path
import os

DIR = Path(sys.argv[0]).parent.resolve()
# детализация журнала работы tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# путь для кэширования моделей tensorflow-hub
os.environ['TFHUB_CACHE_DIR'] = str(DIR)
# показывать ли прогресс загрузки модели
os.environ['TFHUB_DOWNLOAD_PROGRESS'] = '1'
import tensorflow
import tensorflow_hub as tfhub
import numpy
import cv2


hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
# https://www.kagle.com/code/kerneler/arbitrary-image-stylization
hub_module = tfhub.load(hub_handle)
# описание основной модели
sig = hub_module.signatures['serving_default']
print('Входы модели: ')
for name, value in sig.structured_input_signature[1].items():
    print(f'    {name}: {value}')
print('Выходы модели: ')
for name, value in sig.structured_outputs.items():
    print(f'    {name}: {value}')


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
    image = tensorflow.image.crop_to_bounding_box(
        image,  # изображение(я)
        y, x,  # левый верхний угол
        size, size  # размеры прямоугольника
    )
    return image


def load_image(impath: str, size=(256, 256)):
    data = numpy.fromfile(impath, numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError()
    img = img[numpy.newaxis, ...]  # добавляем новую ось массива
    img = img.astype(numpy.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = crop_center(img)
    img = tensorflow.image.resize(img, size, preserve_aspect_ratio=True)
    return img


content_image = 'content.jpg'
content_size = (512, 512)
content = load_image(content_image, content_size)

style_image = 'style.jpg'
style_size = (256, 256)  # авторы советуют уменьшить стиль до 256 * 256
style = load_image(style_image, style_size)
# сглаживаем изображение стиля
style = tensorflow.nn.avg_pool(style, ksize=[3, 3], strides=[1, 1], padding='SAME')
print('Content', content.shape)
print('Style', style.shape)
outputs = hub_module(content, style)
result = outputs[0]  # outputs - список выходов
print('Result', result.shape)
# numpy.squeeze() удаляет из массива оси с размером 1
result_image = numpy.squeeze(result)
content_image = numpy.squeeze(content)
style_image = numpy.squeeze(style)

cv2.imshow('Content', content_image)
cv2.imshow('Style', style_image)
cv2.imshow('Result', result_image)
cv2.waitKey()


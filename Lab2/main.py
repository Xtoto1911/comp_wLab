import numpy
import cv2


def load_image(fname: str) -> numpy.ndarray:
    data = numpy.fromfile(fname, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Invalid image format")
    return img


threshold = int(input('Введите значение порога (0-255): '))
image = load_image('lena.png')
gray = cv2.cvtColor(  # преобразование цвета
    image,  # исходное изображение
    cv2.COLOR_BGR2GRAY  # из чего и во что преобразуем
)
_, binary = cv2.threshold(  # пороговое преобразование
    gray,  # серое изображение
    threshold,  # значение порога
    255,  # значение "белых" пикселей
    cv2.THRESH_BINARY  # простое пороговое преобр.
)
cv2.imshow('Gray', gray)
cv2.imshow('Binary', binary)
cv2.waitKey()

import numpy
import cv2

ESC = 0x0000001B
UP = 0x00260000
DOWN = 0x00280000
LEFT = 0x00250000
RIGHT = 0x00270000


def load_image(fname: str) -> numpy.ndarray:
    data = numpy.fromfile(fname, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Invalid image format")
    return img


image = load_image('lena.png')
gray = cv2.cvtColor(  # преобразование цвета
    image,  # исходное изображение
    cv2.COLOR_BGR2GRAY  # из чего и во что преобразуем
)
N = int(input("Введите размер окна (нечетное число): "))
C = int(input("Введите константу С: "))

binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    N,
    C
)

cv2.imshow('Gray', gray)
cv2.imshow('Binary', binary)
cv2.waitKeyEx()



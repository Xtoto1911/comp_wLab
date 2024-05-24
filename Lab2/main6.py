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


image = load_image('contrast.png')
hsv = cv2.cvtColor(  # преобразование цвета
    image,  # исходное изображение
    cv2.COLOR_BGR2HSV_FULL # из чего и во что преобразуем
)
saturation = hsv[..., 1]

threshold, sat_binary = cv2.threshold(
    saturation,
    0,
    255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU
)
color_image = numpy.copy(image)
gray_image = numpy.copy(image)
color_image[sat_binary > 0] = [127,127,127]
gray_image[sat_binary == 0] = [127,127,127]
cv2.imshow('Color', color_image)
cv2.imshow('Gray', gray_image)
cv2.waitKeyEx()

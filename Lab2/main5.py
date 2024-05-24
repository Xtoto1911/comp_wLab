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

def update_threshold(value, step):
    new_val = value + step
    return numpy.maximum(0, numpy.minimum(255, new_val))


image = load_image('contrast.png')
hsv = cv2.cvtColor(  # преобразование цвета
    image,  # исходное изображение
    cv2.COLOR_BGR2HSV_FULL # из чего и во что преобразуем
)

saturation = hsv[..., 1]
while True:
    threshold, sat_binary = cv2.threshold(
    saturation,
    0,
    255,
    cv2.THRESH_TOZERO | cv2.THRESH_OTSU
    )
    hsv[...,1] = sat_binary
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)
    cv2.imshow('Result', result)

    key = cv2.waitKeyEx()
    if key == ESC:
        break
    elif key == UP:
        hsv[..., 1] = update_threshold(hsv[..., 1], 5)
    elif key == DOWN:
        hsv[..., 1] = update_threshold(hsv[..., 1], -5)
cv2.destroyAllWindows()



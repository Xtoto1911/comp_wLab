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
threshold, binary = cv2.threshold(  # пороговое преобразование
        gray,  # серое изображение
        0,  # значение порога
        255,  # значение "белых" пикселей
        cv2.THRESH_BINARY | cv2.THRESH_OTSU  # простое пороговое преобр.
    )
cv2.imshow('Gray', gray)
while True:
    print(threshold)
    _, binary = cv2.threshold(  # пороговое преобразование
        gray,  # серое изображение
        threshold,  # значение порога
        255,  # значение "белых" пикселей
        cv2.THRESH_BINARY  # простое пороговое преобр.
    )
    cv2.imshow('Binary', binary)
    key = cv2.waitKey()
    if key == ord('w') and threshold < 255:
        threshold += 1
    elif key == ord('s') and threshold > 0:
        threshold -= 1
    elif key == 27:
        break


import numpy
import cv2
import typing as tp
import sys

from numpy.matlib import zeros


def load_image(fname: str) -> numpy.ndarray:
    data = numpy.fromfile(fname, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Invalid image format")
    return img


class Clicker:
    def __init__(self, name: str, image: numpy.ndarray):
        self._wndname = name
        self._image = image
        self.clicks: tp.List[tp.Tuple[int, int]] = list()
        self.marker_size = 5
        self.marker_color = (60, 220, 20)
        cv2.namedWindow(self._wndname, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self._wndname, self._mouse_event)

    def _mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.clicks.append((x, y))
        elif event == cv2.EVENT_RBUTTONUP:
            if self.clicks:
                del self.clicks[-1]
        else:
            return
        self.update()

    def update(self):
        copy = self._image.copy()
        for x, y in self.clicks:
            cv2.circle(copy, (x, y), self.marker_size, self.marker_color, -1)
        cv2.imshow(self._wndname, copy)

    def close(self):
        cv2.destroyWindow(self._wndname)

    def __enter__(self):  # вход в блок with
        self.update()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # выход из блока with
        self.close()


poster = load_image('lena.png')
image = load_image('times-square.jpg')
with Clicker('Image', image) as wnd:
    while len(wnd.clicks) < 4:
        key = cv2.waitKey(100)
        if key == 27:  # Esc
            sys.exit(0)
    print(wnd.clicks)
    pts = numpy.array(wnd.clicks, dtype=numpy.float32)
# определяем размер извлекаемого изображения
height, width = poster.shape[:2]
srcpoints = numpy.array([
    (0, 0),
    (width - 1, 0),
    (width - 1, height - 1),
    (0, height - 1)
], dtype=numpy.float32)
matrix = cv2.getPerspectiveTransform(  # расчёт перспективного преобразования
    srcpoints,  # точки-прообразы ("откуда")
    pts)  # точки-образы ("куда")
warped = cv2.warpPerspective(  # применяет перспективу к изображению
    poster,  # исходное изображение
    matrix,  # матрица преобразования
    (image.shape[1], image.shape[0])  # размер получачемого изображения
)
# формируем логическую маску для переноса пикселей
# функции рисования умеют работать только с uint8
mask = numpy.zeros(image.shape[:2], dtype=numpy.uint8)
cv2.fillPoly(  # рисует залитый многоугольник
    mask,  # на каком изображении рисуем
    pts.reshape(1, 4, 2).astype(numpy.int32),  # координаты
    1,  # "цвет" многоугольника
)
mask.dtype = numpy.bool_
image[mask] = warped[mask]
cv2.imshow('Image', image)
cv2.waitKey()

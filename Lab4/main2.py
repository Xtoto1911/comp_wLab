import numpy
import cv2
import typing as tp
import sys




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


image = load_image('times-square.jpg')
with Clicker('Image', image) as wnd:
    while len(wnd.clicks) < 4:
        key = cv2.waitKey(100)
        if key == 27:  # Esc
            sys.exit(0)
    print(wnd.clicks)
    pts = numpy.array(wnd.clicks, dtype=numpy.float32)
# определяем размер извлекаемого изображения
width = int(max(
    numpy.linalg.norm(pts[1] - pts[0]),  # длина верхней стороны
    numpy.linalg.norm(pts[3] - pts[2])  # длина нижней стороны
))
height = int(max(
    numpy.linalg.norm(pts[3] - pts[0]),  # длина левой стороны
    numpy.linalg.norm(pts[2] - pts[1])  # длина правой стороны
))
dstpoints = numpy.array([
    (0, 0),
    (width - 1, 0),
    (width - 1, height - 1),
    (0, height - 1)
], dtype=numpy.float32)
matrix = cv2.getPerspectiveTransform(  # расчёт перспективного преобразования
    pts,  # точки-прообразы ("откуда")
    dstpoints)  # точки-образы ("куда")
warped = cv2.warpPerspective(  # применяет перспективу к изображению
    image,  # исходное изображение
    matrix,  # матрица преобразования
    (width, height)  # размер получачемого изображения
 )
cv2.imshow('Warped', warped)
cv2.waitKey()
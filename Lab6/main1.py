import numpy
import cv2


def load_image(fname: str) -> numpy.ndarray:
    data = numpy.fromfile(fname, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Invalid image format")
    return img


image = load_image('road.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.blur(  # размытие изображения
    gray,  # исходное изображение
    (3, 3)  # среднее по окну 3х3 пикселя
)
canny = cv2.Canny(  # поиск границ
    blurred,  # изображение
    50, 150  # пороговые значения
)
# параметры алгоритма Хафа
# порог аккумулятора - чем больше, тем более строгий отбор
threshold = 100
# минимальная длина прямой
minLineLength = 100
# максимально допустимый разрыв в прямой
maxLineGap = 100
while True:
    print(f'Thresh = {threshold} minL = {minLineLength} maxG = {maxLineGap}')
    lines = cv2.HoughLinesP(  # поиск отрезков прямых
        canny,  # где ищем прямые
        rho=1,  # шаг смещения прямой в пикселях
        theta=numpy.pi/180,  # шаг угла наклона в радианах
        threshold=threshold,
        minLineLength=minLineLength,
        maxLineGap=maxLineGap
    )
    copy = image.copy()
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(
            copy,
            (x1, y1), (x2, y2),
            (64, 255, 255),
            2
        )
    cv2.imshow('Result', copy)
    key = cv2.waitKey()
    if key == 27:  # Esc
        break
    elif key == ord('q'):
        threshold += 5
    elif key == ord('a'):
        threshold -= 5
    elif key == ord('w'):
        minLineLength += 5
    elif key == ord('s'):
        minLineLength -= 5
    elif key == ord('e'):
        maxLineGap += 5
    elif key == ord('d'):
        maxLineGap -= 5
    threshold = max(5, min(1000, threshold))
    minLineLength = max(5, min(1000, minLineLength))
    maxLineGap = max(0, min(100, maxLineGap))

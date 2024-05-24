import numpy
import cv2


def load_image(fname: str) -> numpy.ndarray:
    data = numpy.fromfile(fname, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Invalid image format")
    return img


image = load_image('coins.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.blur(  # размытие изображения
    gray,  # исходное изображение
    (3, 3)  # среднее по окну 3х3 пикселя
)
# параметры поиска окружностей
# минимальное число голосов
threshold = 80
# минимальный радиус окружности
minRadius = 40
# максимальный радиус окружности
maxRadius = 0  # не ограничен
# минимальная расстояние между центрами
midDist = 100

while True:
    print(f'T = {threshold}, D = {midDist}, R = {minRadius}')
    circles = cv2.HoughCircles(  # ищем окружности
        blurred,  # изображение
        method=cv2.HOUGH_GRADIENT,  # алгоритм поиска
        dp=1,  # шаг при поиске центра
        minDist=midDist,
        param1=50,  # для метода GRADIENT - порог фильтра Кэнни
        param2=threshold,  # для метода GRADIENT - порог аккумулятора
        minRadius=minRadius,
        maxRadius=maxRadius
    )
    copy = image.copy()
    if circles is not None:
        for x, y, r in circles[0, :].astype(numpy.int32):
            cv2.circle(copy, (x, y), r, (64, 255, 64), 2)
    cv2.imshow('Result', copy)
    key = cv2.waitKey()
    if key == 27:  # Esc
        break
    elif key == ord('q'):
        threshold += 5
    elif key == ord('a'):
        threshold -= 5
    elif key == ord('w'):
        midDist += 5
    elif key == ord('s'):
        midDist -= 5
    elif key == ord('e'):
        minRadius += 5
    elif key == ord('d'):
        minRadius -= 5
    threshold = max(5, min(1000, threshold))
    midDist = max(5, min(1000, midDist))
    minRadius = max(5, min(1000, minRadius))

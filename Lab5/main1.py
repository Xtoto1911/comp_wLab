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


sample = load_image("book1.jpg")
sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
# рассматриваем особенности с "силой" не менее 10% от наибольшей
quality_level = 0.1
# радиус, в котором "сильная" особенность подавляет "слабые"
min_distance = 20
while True:
    # ищем особенности
    sample_features = cv2.goodFeaturesToTrack(
        image=sample_gray,  # обрабатываемое изображение
        maxCorners=100,  # возвращать не более 100 особенностей
        qualityLevel=quality_level,  # насколько "сильные" особенности возвращать
        minDistance=min_distance,  # мин. расстояние между особенности
        useHarrisDetector=False  # используем детектор Ши-Томаси
    )
    copy = sample.copy()
    sf = sample_features[:, 0, :].astype(numpy.int32)
    for x, y in sf:
        cv2.circle(copy, (x, y), 5, (128, 255, 255), 1)
        cv2.circle(copy, (x, y), min_distance, (255, 128, 255), 1)
    cv2.imshow('Features', copy)
    key = cv2.waitKeyEx()
    if key == UP and quality_level < 0.95:
        quality_level += 0.05
    elif key == DOWN and quality_level > 0.05:
        quality_level -= 0.05
    elif key == LEFT and min_distance > 5:
        min_distance -= 5
    elif key == RIGHT and min_distance < 200:
        min_distance += 5
    elif key == ESC:
        break


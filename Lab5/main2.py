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


sift = cv2.SIFT.create()
matcher = cv2.BFMatcher(cv2.NORM_L2)

sample = load_image("book1.jpg")
sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
# ищем особенности на изображении
sample_pts, sample_descriptors = sift.detectAndCompute(
    sample_gray,  # изображение
    None  # у нас нет подсказки о расположении особенностей
)

scene = load_image('bookz.jpg')
scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
scene_pts, scene_descriptors = sift.detectAndCompute(
    scene_gray,  # изображение
    None  # у нас нет подсказки о расположении особенностей
)
# сравниваем особенности
matches = matcher.knnMatch(
    sample_descriptors,  # дескрипторы на образце
    scene_descriptors,  # дескрипторы на сцене
    k=2  # для каждой точки найти 2 самых похожих
)
# критерий Лёвэ - у точки должно быть одно хорошее совпадение
lowe_thresh = 0.8
while True:
    print(f'Lowe={lowe_thresh}')
    good_matches = []
    for m1, m2 in matches:  # перебираем сравнения
        if m1.distance < lowe_thresh * m2.distance:
            # m1 "Дальше" чем m2 на 10% и более
            good_matches.append(m1)  # считаем m1 хорошим сравнением

    # отображаем пары точек
    W = sample.shape[1] + scene.shape[1]
    H = max(sample.shape[0], scene.shape[0])
    result = numpy.zeros((H, W, 3), dtype=numpy.uint8)
    cv2.drawMatches(
        sample,  # изображение-образец
        sample_pts,  # координаты особенностей на образце
        scene,  # изображение-сцена
        scene_pts,  # координаты особенностей на сцене
        good_matches,  # список "хороших" совпадений
        result,  # куда поместить результат
        # рисуем только точки, у которых есть пара
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow('Matches', result)
    key = cv2.waitKeyEx()
    if key == UP:
        lowe_thresh += 0.01
    elif key == DOWN:
        lowe_thresh -= 0.01
    elif key == ESC:
        break
    lowe_thresh = min(0.99, max(0.01, lowe_thresh))

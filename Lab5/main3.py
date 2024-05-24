import numpy
import cv2


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
# координаты углов
corners_sample = numpy.array([
    (0, 0),
    (sample.shape[1] - 1, 0),
    (sample.shape[1] - 1, sample.shape[0] - 1),
    (0, sample.shape[0] - 1)
], numpy.float32)
corners_sample = corners_sample.reshape((-1, 1, 2))
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
good_matches = []
for m1, m2 in matches:  # перебираем сравнения
    if m1.distance < lowe_thresh * m2.distance:
        # m1 "Дальше" чем m2 на 10% и более
        good_matches.append(m1)  # считаем m1 хорошим сравнением

# определяем чётко узнаваемые точки
points_sample = []
points_scene = []
for m in good_matches:
    points_sample.append(sample_pts[m.queryIdx].pt)
    points_scene.append(scene_pts[m.trainIdx].pt)
points_sample = numpy.array(points_sample, numpy.float32)
points_sample = points_sample.reshape((-1, 1, 2))
points_scene = numpy.array(points_scene, numpy.float32)
points_scene = points_scene.reshape((-1, 1, 2))
# ищем проективное преобразование
matrix, ptmask = cv2.findHomography(
    points_sample,  # прообразы
    points_scene,  # образы
    cv2.RANSAC  # используем метод
)
# ищем координаты углов книги в сцене
corners_scene = cv2.perspectiveTransform(corners_sample, matrix)
# преобразуем массив координат в вид, пригодный для рисования
corners_scene = corners_scene.reshape((1, -1, 2)).astype(numpy.int32)
# рисуем рамку вокруг книги
cv2.polylines(  # рисуем многоугольники
    scene,  # изображение, на котором рисуем
    corners_scene,  # координаты углов
    True,  # замкнуть каждый многоугольник
    (64, 255, 64),  # цвет
    2  # толщина линии
)
cv2.imshow('Match', scene)
cv2.waitKey()

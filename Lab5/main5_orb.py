import numpy
import cv2
import time

def load_image(fname: str) -> numpy.ndarray:
    data = numpy.fromfile(fname, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Invalid image format")
    return img


orb = cv2.ORB.create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

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
sample_pts, sample_descriptors = orb.detectAndCompute(
    sample_gray,  # изображение
    None  # у нас нет подсказки о расположении особенностей
)

video_capture = cv2.VideoCapture('bookz.mp4')
time_search_points = []
time_matching_points = []
time_find_homography = []
while True: 
    res, frame = video_capture.read()
    if not res:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    timer = time.time()
    frame_pts, frame_descriptions = orb.detectAndCompute(
        frame_gray,
        None
    )
    time_search_points.append((time.time() - timer))
    # сравниваем особенности
    timer = time.time()
    matches = matcher.knnMatch(
        frame_descriptions,  # дескрипторы на образце
        sample_descriptors,  # дескрипторы на сцене
        k=2  # для каждой точки найти 2 самых похожих
    )
    time_matching_points.append((time.time() - timer))
    # критерий Лёвэ - у точки должно быть одно хорошее совпадение
    lowe_thresh = 0.8
    timer = time.time()
    good_matches = []
    for m1, m2 in matches:  # перебираем сравнения
        if m1.distance < lowe_thresh * m2.distance:
            # m1 "Дальше" чем m2 на 10% и более
            good_matches.append(m1)  # считаем m1 хорошим сравнением

    # определяем чётко узнаваемые точки
    points_sample = []
    points_frame = []
    for m in good_matches:
        points_sample.append(sample_pts[m.trainIdx].pt)
        points_frame.append(frame_pts[m.queryIdx].pt)
    points_sample = numpy.array(points_sample, numpy.float32)
    points_sample = points_sample.reshape((-1, 1, 2))
    points_frame = numpy.array(points_frame, numpy.float32)
    points_frame = points_frame.reshape((-1, 1, 2))
    # ищем проективное преобразование
    matrix, ptmask = cv2.findHomography(
        points_sample,  # прообразы
        points_frame,  # образы
        cv2.RANSAC  # используем метод
    )
    time_find_homography.append((time.time() - timer))
    # ищем координаты углов книги в сцене
    corners_scene = cv2.perspectiveTransform(corners_sample, matrix)
    # преобразуем массив координат в вид, пригодный для рисования
    corners_scene = corners_scene.reshape((1, -1, 2)).astype(numpy.int32)
    # рисуем рамку вокруг книги
    cv2.polylines(  # рисуем многоугольники
        frame,  # изображение, на котором рисуем
        corners_scene,  # координаты углов
        True,  # замкнуть каждый многоугольник
        (64, 255, 64),  # цвет
        2  # толщина линии
    )
    cv2.imshow('Match', frame)
    key = cv2.waitKey(1)
    if key == 27: 
        break
    
video_capture.release()
cv2.destroyAllWindows()

avg_time_search_points = sum(time_search_points) / len(time_search_points)
avg_time_matching_points = sum(time_matching_points) / len(time_matching_points)
avg_time_find_homography = sum(time_find_homography) / len(time_find_homography)

print("Среднее время поиска точек: {:.5f} сек.".format(avg_time_search_points))
print("Среднее время сопоставления точек: {:.5f} сек.".format(avg_time_matching_points))
print("Среднее время поиска гомографии: {:.5f} сек.".format(avg_time_find_homography))

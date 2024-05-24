import numpy
import cv2

# число внутренних углов где встречаются 4 клетки
PATTERN_SIZE = (7, 3)


def make_pattern(shape: tuple) -> numpy.ndarray:
    pts = []
    for r in range(1, PATTERN_SIZE[1]+1):
        for c in range(1, PATTERN_SIZE[0]+1):
            x = c * shape[1] // (PATTERN_SIZE[0] + 1)
            y = r * shape[0] // (PATTERN_SIZE[1] + 1)
            pts.append((x, y))
    pts = numpy.array(pts, dtype=numpy.float32)
    return pts.reshape((-1, 1, 2))


def load_image(fname: str) -> numpy.ndarray:
    data = numpy.fromfile(fname, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Invalid image format")
    return img


insert = load_image('times-square.jpg')
insert_corners = make_pattern(insert.shape)
insert_endpoints = numpy.array([
    (0, 0),
    (insert.shape[1] - 1, 0),
    (insert.shape[1] - 1, insert.shape[0] - 1),
    (0, insert.shape[0] - 1)
], dtype=numpy.float32).reshape((-1, 1, 2))
# cv2.drawChessboardCorners(insert, PATTERN_SIZE, insert_corners, True)
# cv2.imshow('Insert', insert)
video = cv2.VideoCapture('chessboard.mp4')
warped = None
try:
    while True:
        success, frame = video.read()  # читаем кадр из видео
        if not success:  # кадр прочитать не удалось
            print('Video has ended.')
            break
        if warped is None:
            warped = numpy.zeros(frame.shape, numpy.uint8)
        success, corners = cv2.findChessboardCorners(  # ищем шахматный шаблон
            frame,  # где ищем шаблон
            PATTERN_SIZE,  # размер шаблона
            flags=cv2.CALIB_CB_FILTER_QUADS
        )
        if success:  # нашли шаблон
            # если начальная точка правее конечной, массив "вверх ногами"
            if corners[0, 0, 0] > corners[-1, 0, 0]:
                corners = corners[::-1, ...]  # разворачиваем массив
            # cv2.drawChessboardCorners(frame, PATTERN_SIZE, corners, success)
            # ищем проективное преобразование
            matrix, ptmask = cv2.findHomography(
                insert_corners,  # точки прообразы ("откуда")
                corners,  # точки образы ("куда")
                cv2.RANSAC  # используем метод RANSAC
            )
            warped.fill(0)
            cv2.warpPerspective(  # проективное преобр. изображения
                insert,  # какое изображение преобразуем
                matrix,  # матрица преобразования
                (warped.shape[1], warped.shape[0]),  # размер
                dst=warped  # записать результат в массив warped
            )
            # cv2.imshow('Warped', warped)
            # ищем углы шахматной доски
            ends = cv2.perspectiveTransform(
                insert_endpoints,  # координаты для преобразования
                matrix  # матрица преобразования
            )
            ends = ends.reshape((1, -1, 2)).astype(numpy.int32)
            # делаем маску для переноса пикселей
            mask = numpy.zeros(frame.shape[:2], numpy.uint8)
            cv2.fillPoly(mask, ends, (1,))  # рисуем маску
            mask.dtype = numpy.bool_
            frame[mask] = warped[mask]
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(10)  # ждём нажатия с таймаутом
        if key == 27:
            print('Stopped by user.')
            break
finally:
    video.release()  # освобождает источник видео

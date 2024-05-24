import pipes

import numpy
import cv2

# число внутренних углов где встречаются 4 клетки
PATTERN_SIZE = (7, 3)


video = cv2.VideoCapture('chessboard.mp4')
try:
    while True:
        success, frame = video.read()  # читаем кадр из видео
        if not success:  # кадр прочитать не удалось
            print('Video has ended.')
            break
        success, corners = cv2.findChessboardCorners(  # ищем шахматный шаблон
            frame,  # где ищем шаблон
            PATTERN_SIZE,  # размер шаблона
            flags=cv2.CALIB_CB_FILTER_QUADS
        )
        if success:  # нашли шаблон
            # если начальная точка правее конечной, массив "вверх ногами"
            if corners[0, 0, 0] > corners[-1, 0, 0]:
                corners = corners[::-1, ...]  # разворачиваем массив
            cv2.drawChessboardCorners(frame, PATTERN_SIZE, corners, success)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(10)  # ждём нажатия с таймаутом
        if key == 27:
            print('Stopped by user.')
            break
finally:
    video.release()  # освобождает источник видео

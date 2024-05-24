import numpy
import cv2
import sys

video = cv2.VideoCapture('NewYork.mp4')
_, old_frame = video.read()

window = cv2.selectROI('Select area', old_frame)
x, y, w, h = window
if w == 0 or h == 0:
    sys.exit(0)
roi = old_frame[y:y+h, x: x+w]
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
# пороговое преобразование по нескольки каналам
mask = cv2.inRange(
    hsv_roi,
    # нижняя граница диапозонов
    numpy.array([0.0,60.0,32.0]),
    # верхняя граница диапозонов
    numpy.array([180.0,255.0,255.0]),
)
# какие цвета есть в области
histogram = cv2.calcHist(
    [hsv_roi],
    [0], # нулевой канал
    mask,
    [180], # сколько корзин в гистограмме
    [0,180]
)

cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)

term_crit = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    10, 1
)

while True:
    if cv2.waitKey(10) == 27:
        break
    success, frame = video.read()
    if not success:
        break
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject(
        [hsv],
        [0],
        histogram,
        [0, 180],
        1
    )
    _new_area, window = cv2.meanShift(dst,window,term_crit)
    x,y,w,h = window
    cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 255, 255), 2)
    cv2.imshow('Frame', frame)
    cv2.imshow('Back proj', dst)
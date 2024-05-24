import numpy
import cv2

video = cv2.VideoCapture('NewYork.mp4')
#  поиска особенностей
feature_params = {
    'maxCorners': 100,# макс. число точек
    'qualityLevel': 0.3,# сила точки не менее 30проц от макс
    'minDistance': 30# минимальное расстояние между точками
}
# параметры расчета оптического потока
lk_params = {
    'winSize': (15,15), # размер окрестности в пикселях
    'maxLevel': 2, # уравней в пирамиде изображений
    'criteria': ( # алгоритм итеративный - нужны параметры остановы
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10, 0.03
    )
}

success, old_frame = video.read()
if not success:
    raise IOError('Video')

old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
old_features = cv2.goodFeaturesToTrack(
    old_gray,
    **feature_params
)
while(True):
    success, frame = video.read()
    if not success:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # расчитаем оптический поток
    moved_features, good, _err = cv2.calcOpticalFlowPyrLK(
        prevImg=old_gray,#предыдущий кадр
        nextImg=gray,#следующий
        prevPts=old_features,
        nextPts=None,
        **lk_params
    )

    # найти точки, для которох известны позиция до и после
    good_old = old_features[good != 0].astype(numpy.int32)
    good_new = moved_features[good != 0].astype(numpy.int32)

    for p1, p2 in zip(good_old,good_new):
        delta = (p1 - p2) * 5
        cv2.line(
            frame,
            p2,
            p2 + delta,
            (128,255,255),
            1
        )
        cv2.circle(frame, p2, 2, (128,255,255),-1)
    cv2.imshow('Flow',frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

    old_features = cv2.goodFeaturesToTrack(
        gray, **feature_params
    )
    old_frame , old_gray = frame, gray

video.release()
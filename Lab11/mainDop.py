import numpy
import cv2

video = cv2.VideoCapture('v2HewYork.mp4')
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
    raise IOError('Video...')

box = cv2.selectROI(old_frame, False)
cv2.destroyAllWindows()
x,y,w,h = box
corners = numpy.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], 
                      dtype=numpy.float32)
old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)

while True:
    success, frame = video.read()
    if not success:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # расчитаем оптический поток
    moved_features, good, _err = cv2.calcOpticalFlowPyrLK(
        prevImg=old_gray,#предыдущий кадр
        nextImg=gray,#следующий
        prevPts=corners.reshape(-1,1,2),
        nextPts=None,
        **lk_params
    )
    new_corns = moved_features.reshape(-1,2)
    if not numpy.all(good):
        print('Не все углы в кадре')
        break
    cv2.polylines(frame,[new_corns.astype(numpy.int32)],
                  isClosed=True,
                  color=(128,128,255),
                  thickness=3)
    cv2.imshow('Vivod',frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

    corners , old_gray = new_corns, gray

video.release()
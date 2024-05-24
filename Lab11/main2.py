import numpy
import cv2

video = cv2.VideoCapture('sunset.mp4')

# реализация метода смеси гауссиан
back_sub = cv2.createBackgroundSubtractorMOG2(
    detectShadows= False
)
kernel_open = numpy.array([
    [0,1,0],
    [1,1,1],
    [0,1,0],
], numpy.uint8)
kernel_close = numpy.ones((5,5), numpy.uint8)
while True:
    if cv2. waitKey(20) == 27:
        break
    success, frame = video.read()
    if not success:
        break
    mask = back_sub.apply(
        frame,
        # 0 не обновлять
        # 1 забывать историю на каждом кадре
        learningRate= 0.010
    )
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel_close)
    # ищем блобы белых пикселей на максе
    _, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask,
        ltype=cv2.CV_16U
    )
    for stat in stats[1:].astype(numpy.int32):
        x, y, w, h, area = stat
        if area > 20:
            cv2.rectangle(
                frame,
                (x,y),
                (x + w,y + h),
                (128, 255, 255),
                2
            )
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
video.release()
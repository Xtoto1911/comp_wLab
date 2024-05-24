import numpy
import cv2


def load_image(fname: str) -> numpy.ndarray:
    data = numpy.fromfile(fname, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Invalid image format")
    return img


# сцена где ищем фигуру
image = load_image('sils.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_image = cv2.blur(gray_image, (3, 3))
# образец искомой фигуры
sample = load_image('sil3.png')
gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
blur_sample = cv2.blur(gray_sample, (3, 3))

angle_t = 50
scale_t = 50
pos_t = 150

ght = cv2.createGeneralizedHoughGuil()
ght.setMaxBufferSize(1000)
ght.setLevels(360)
ght.setXi(45)
# параметры поиска ориентации
ght.setMinAngle(0)
ght.setMaxAngle(360)
ght.setAngleEpsilon(5)
ght.setAngleStep(5)
ght.setAngleThresh(angle_t)
# параметры поиска масштаба
ght.setMinScale(1.0)
ght.setMaxScale(1.01)
ght.setScaleStep(0.1)
ght.setScaleThresh(scale_t)
# параметры поиска позиции
ght.setMinDist(50)  # минимальное расстояние
ght.setDp(2)  # шаг при поиске позиции
ght.setPosThresh(pos_t)
ght.setTemplate(blur_sample)  # что ищем

positions, votes = ght.detect(blur_image)
copy = image.copy()
if positions is not None:
    color = (64, 255, 64)
    R = max(gray_sample.shape) // 2
    for obj in positions[0]:
        x, y = obj[:2].astype(numpy.int32)
        scale = obj[2]
        angle = int(obj[3])  # угол наклона в градусах
        r = int(R * scale)
        c = numpy.cos(numpy.radians(angle-90))
        s = numpy.sin(numpy.radians(angle-90))
        cv2.circle(copy, (x, y), 2, color, -1)
        cv2.circle(copy, (x, y), r, color, 2)
        cv2.line(copy, (x, y), (x + int(r * c), y + int(r * s)), color, 2)
cv2.imshow('Result', copy)
cv2.waitKey()

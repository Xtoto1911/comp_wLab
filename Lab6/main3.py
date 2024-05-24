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
ght = cv2.createGeneralizedHoughBallard()
ght.setLevels(360)
ght.setMinDist(50)  # минимальное расстояние
ght.setDp(2)  # шаг при поиске позиции
ght.setVotesThreshold(50)  # необходимое число голосов
ght.setTemplate(gray_sample)  # что ищем

positions, votes = ght.detect(gray_image)
copy = image.copy()
if positions is not None:
    color = (64, 255, 64)
    R = max(gray_sample.shape) // 2
    for obj in positions[0]:
        x, y = obj[:2].astype(numpy.int32)
        cv2.circle(copy, (x, y), 2, color, -1)
        cv2.circle(copy, (x, y), R, color, 2)
cv2.imshow('Result', copy)
cv2.waitKey()

import cv2
import numpy
from pathlib import Path
from cap_from_youtube import cap_from_youtube

ESC = 0x0000001B
UP = 0x00260000
DOWN = 0x00280000
LEFT = 0x00250000
RIGHT = 0x00270000


def load_image(fname: str) -> numpy.ndarray:
    data = numpy.fromfile(fname, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Invalid image format")
    return img


CASCADE = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(str(CASCADE))
if cascade.empty():
    raise IOError('Каскад не найден или повреждён')

# video = cv2.VideoCapture('face.mp4')
url = 'https://www.youtube.com/watch?v=pR_KraxE7MA'
quality = '480p'
video = cap_from_youtube(url, quality)
# во сколько раз уменьшать изображение на каждой попытке
scale_per_stage = 1.3
# сколько соседних окон тоже должно дать отклик
min_neighbours = 5

while True:
    success, image = video.read()
    if not success:
        print('Видео закончилось')
        break
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(min_neighbours)
    rois = cascade.detectMultiScale(  # поиск объекта
        gray,  # анализируем изображение
        scaleFactor=scale_per_stage,  # для пирамиды Гаусса
        minNeighbors=min_neighbours  # для дедупликации
    )
    copy = image.copy()
    for x, y, w, h in rois:
        cv2.rectangle(copy,
                      (x, y), (x+w, y+h),
                      (64, 255, 255), 2
                      )
    cv2.imshow('Face', copy)
    key = cv2.waitKeyEx(10)
    if key == UP:
        min_neighbours += 1
    elif key == DOWN:
        min_neighbours -= 1
    elif key == ESC:
        break
    min_neighbours = max(0, min(20, min_neighbours))

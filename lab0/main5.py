import numpy
import cv2


def load_image(fpath: str) -> numpy.ndarray:
    data = numpy.fromfile(fpath, dtype=numpy.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError('Invalid file format')
    return image


def save_image(fpath: str, image: numpy.ndarray) -> None:
    ret, data = cv2.imencode('.png', image)
    if not ret:
        raise IOError('Invalid image data')
    with open(fpath, 'wb') as dst:
        dst.write(data)


channels = ['blue', 'green', 'red']
img = load_image('lena.png')
x, y, w, h = cv2.selectROI('Select region', img)
print(x, y, w, h)
if (w != 0) and (h != 0):
    fragment = img[y:y+h, x:x+w, :]
    filename = input("Введите имя файла для сохранения: ")
    save_image(f'{filename}.png', fragment)
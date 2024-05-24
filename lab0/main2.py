import numpy
import cv2


def load_image(fpath: str) -> numpy.ndarray:
    data = numpy.fromfile(fpath, dtype=numpy.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError('Invalid file format')
    return image


img = load_image('лена.png')
h = img.shape[0]
w = img.shape[1]
ch = img.shape[2]
print(f'Изображение имеет размер {w}x{h} и {ch} канала')

import numpy
import cv2


def load_image(fpath: str) -> numpy.ndarray:
    data = numpy.fromfile(fpath, dtype=numpy.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError('Invalid file format')
    return image


img = load_image('лена.png')
cv2.imshow('Picture of Lena', img)
while cv2.waitKey() not in (27, None):
    pass

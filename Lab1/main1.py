import numpy
import cv2

def load_image(fname: str) -> numpy.ndarray:
    data = numpy.fromfile(fname, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Invalid image format")
    return img

image = load_image('lena_2.png')
chnames = ['blue', 'green', 'red']
for i, chname in enumerate(chnames):
    part = image[..., i]
    # линейный идекс минимума и максимума
    imin = part.argmin()
    imax = part.argmax()
    #unravel_index() пересчитывает линейный индекс с обычный
    minpos = numpy.unravel_index(imin,part.shape)
    maxpos = numpy.unravel_index(imax, part.shape)
    print(chname)
    print(f'   Min at {minpos} = {part[minpos]}')
    print(f'   Max at {maxpos} = {part[maxpos]}')
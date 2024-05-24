import numpy
import cv2


def load_image(fname: str) -> numpy.ndarray:
    data = numpy.fromfile(fname, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Invalid image format")
    return img



image = load_image('lena_2.png')
copy = image.copy()#копия изображения для вывода на экран
chnames = ['blue', 'green', 'red']
for i, chname in enumerate(chnames):
    part = image[..., i]
    # линейный идекс минимума и максимума
    imin = part.argmin()
    imax = part.argmax()
    # unravel_index() пересчитывает линейный индекс с обычный
    minpos = numpy.unravel_index(imin,part.shape)
    maxpos = numpy.unravel_index(imax, part.shape)
    print(chname)
    print(f'   Min at {minpos} = {part[minpos]}')
    print(f'   Max at {maxpos} = {part[maxpos]}')
    color = [0, 0, 0]
    color[i] = 255
    cv2.line(# рисуем прямую
        copy, # изображение
        pt1 = minpos[::-1], # начальная точка(X,Y)
        pt2 = maxpos[::-1], # конечная точка(X,Y)
        color = color, # цвет линии(B,G,R)
        thickness = 2 # толщина в пикселях
    )
    cv2.circle(
        copy,
        center = maxpos[::-1],
        radius = 5,
        color = color,
        thickness = -1,# если -1, то закрашенный круг
    )
cv2.imshow('MinMax',copy)
cv2.waitKey()
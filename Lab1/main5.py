import numpy
import cv2


def load_image(fname: str) -> numpy.ndarray:
    data = numpy.fromfile(fname, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Invalid image format")
    return img


k_str = input('Введите множитель: ')
k = float(k_str)
image = load_image('lena_2.png')
copy = image.copy() # копия изображения для вывода на экран
print(image.dtype)
chnames = ['blue', 'green', 'red']
imin = image.argmin()
imax = image.argmax()
avarage = image.mean()
minpos = numpy.unravel_index(imin, image.shape)
maxpos = numpy.unravel_index(imax, image.shape)
minval = image[minpos] # минимальное значение любого канала
maxval = image[maxpos] # максимальное значение любого канала 1
print(f'Диапазон изображения {minval}...{maxval}')
float_image = image.astype(numpy.float32) # копия с типом float
# это выражение будет создавать копии массива
# float_image = (float_image - minval) / (maxval - minval) * 255
# мы сделаем без создания копий, изменяя массив на месте
float_image -= avarage
float_image *= k
float_image += avarage
float_image.clip(min = 0, max = 255, out=float_image)
copy = float_image.astype(numpy.uint8)
imin = copy.argmin()
imax = copy.argmax()
minpos = numpy.unravel_index(imin, copy.shape)
maxpos = numpy.unravel_index(imax, copy.shape)
minval = copy[minpos] # минимальное значение любого канала
maxval = copy[maxpos] # максимальное значение любого канала
print(f'Диапазон изображения {minval}...{maxval}')

# создаем массив, заполненный нулями
w = image.shape[1]
combo = numpy.zeros(
    (image.shape[0], w * 2, 3),
    dtype=numpy.uint8
)
combo[:, :w, :] = image
combo[:, w:, :] = copy
cv2.putText( # Выводит текст изображения
    img=combo, # куда выводить текст
    text='До',
    org=(0, combo.shape[0] - 20), # точка отчета
    fontFace=cv2.FONT_HERSHEY_COMPLEX,
    fontScale=1.0, # маштаб шрифта
    color=(255,255,255) # цвет текста
)

cv2.putText( # Выводит текст изображения
    img=combo, # куда выводить текст
    text='После',
    org=(w, combo.shape[0] - 20), # точка отчета
    fontFace=cv2.FONT_HERSHEY_COMPLEX,
    fontScale=1.0, # маштаб шрифта
    color=(255,255,255) # цвет текста
)

cv2.imshow('Before and after',combo)
cv2.waitKey()
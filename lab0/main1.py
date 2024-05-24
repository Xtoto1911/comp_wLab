import numpy
import cv2

img = cv2.imread('lena.png')
if img is None:
    print('Не удалось загрузить файл')
else:
    h = img.shape[0]
    w = img.shape[1]
    ch = img.shape[2]
    print(f'Изображение имеет размер {w}x{h} и {ch} канала')

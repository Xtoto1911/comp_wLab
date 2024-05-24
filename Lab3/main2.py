import numpy
import  cv2

def load_image(fpath: str) -> numpy.ndarray:
    data = numpy.fromfile(fpath, dtype=numpy.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError('Invalid file format')
    return image


def view_fourier(f: numpy.ndarray)-> numpy.ndarray:
    res = numpy.log(numpy.abs(f))
    min, max = res.min(), res.max()
    res-=min
    res*=255.0/(max-min)
    return res.astype(numpy.uint8)


image = load_image('bridge.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('Image', gray)

fourier = numpy.fft.fft2(gray)  # быстрое преобр. Фурье
fourier = numpy.fft.fftshift(fourier)  # приводим к каноническому виду
print(f'Изображение: {gray.shape}, {gray.dtype}')
print(f'Форма Фурье: {fourier.shape}, {fourier.dtype}')
amp = view_fourier(fourier)
print(f'Амплитуды: {amp.shape}, {amp.dtype}')
print(f'     {amp.min()}... {amp.max()}')
cv2.imshow('Амплитуды', amp)

unshift = numpy.fft.ifftshift(fourier)
restored = numpy.fft.ifft2(unshift)
print(f'Востоновленное: {restored.shape}, {restored.dtype}')
result = numpy.abs(restored).astype(numpy.uint8)
cv2.imshow('Result', result)
cv2.waitKey()
import numpy
import  cv2

ESC   = 0x0000001B
UP    = 0x00260000
DOWN  = 0x00280000
LEFT  = 0x00250000
RIGHT = 0x00270000


def load_image(fpath: str) -> numpy.ndarray:
    data = numpy.fromfile(fpath, dtype=numpy.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError('Invalid file format')
    return image


def view_fourier(f: numpy.ndarray)-> numpy.ndarray:
    res = numpy.log(numpy.abs(f))
    min, max = res.min(), res.max()
    res -= min
    res *= 255.0/(max-min)
    return res.astype(numpy.uint8)


def round_mask(shape: type, larger_radius: int, smaller_radius: int) -> numpy.ndarray:
    xr = numpy.arange(0, shape[1]) - shape[1] // 2
    yr = numpy.arange(0, shape[0]) - shape[0] // 2
    sq = xr[numpy.newaxis, :]**2 + yr[:, numpy.newaxis]**2
    mask_big = sq <= larger_radius**2
    mask_small = sq <= smaller_radius**2
    return mask_big & ~mask_small




image = load_image('bridge.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image', gray)


fourier = numpy.fft.fft2(gray)  # быстрое преобр. Фурье
fourier = numpy.fft.fftshift(fourier)
larger_radius = gray.shape[0] // 8
smaller_radius = larger_radius // 2
sost = True
while True:
    mask = round_mask(gray.shape, larger_radius, smaller_radius)
    f = fourier.copy()
    if not sost:
        mask = ~mask
    f[mask] = 1e-10
    amp = view_fourier(f)
    cv2.imshow('Амплитуды', amp)
    unshift = numpy.fft.ifftshift(f)
    restored = numpy.fft.ifft2(unshift)
    result = numpy.abs(restored).astype(numpy.uint8)
    cv2.imshow('Result', result)
    res = cv2.waitKeyEx(0)
    if res == ESC:
        break
    elif res == 32:
        sost = not sost
    elif res == UP:
        larger_radius += 1
    elif res == DOWN:
        larger_radius -= 1
    elif res == LEFT:
        smaller_radius -= 1
    elif res == RIGHT:
        smaller_radius += 1


cv2.destroyAllWindows()


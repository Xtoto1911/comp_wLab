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
    res -= min
    res *= 255.0/(max-min)
    return res.astype(numpy.uint8)


def round_mask(shape: type, radius: int) -> numpy.ndarray:
    xr = numpy.arange(0, shape[1]) - shape[1]//2
    yr = numpy.arange(0, shape[0]) - shape[0]//2
    d = xr[numpy.newaxis, :] ** 2 + yr[:, numpy.newaxis]**2
    return d < radius ** 2




image = load_image('bridge.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image', gray)


fourier = numpy.fft.fft2(gray)  # быстрое преобр. Фурье
fourier = numpy.fft.fftshift(fourier)
k = 8
sost = True
while True:
    radius = gray.shape[0]//k
    mask = round_mask(gray.shape, radius)
    f = fourier.copy()
    if(sost):
        f[~mask] = 1e-10
    else:
        f[mask] = 1e-10
    amp = view_fourier(f)
    cv2.imshow('Амплитуды', amp)
    unshift = numpy.fft.ifftshift(f)
    restored = numpy.fft.ifft2(unshift)
    result = numpy.abs(restored).astype(numpy.uint8)
    cv2.imshow('Result', result)
    res = cv2.waitKeyEx()
    if res == 27:
        break
    if res == 0x00260000 and k <= gray.shape[1]:
        k += 1
    if res == 0x00280000 and k > 1:
        k -= 1
    if res == 32:
        sost = not sost

cv2.destroyAllWindows()


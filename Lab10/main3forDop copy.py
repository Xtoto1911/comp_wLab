import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import sys
import numpy
import cv2


class DrawingWindow:
    def __init__(self, name: str, size: int):
        self.name = name
        self.image = numpy.zeros((size, size),
                                 numpy.uint8)
        self.w = int(2 * size / 28)
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self._mouse)
        self._prev = None

    def update(self):
        cv2.imshow(self.name, self.image)

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONUP:
            self.image.fill(0)
            self._prev = None
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._prev = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._prev = None
        elif event == cv2.EVENT_MOUSEMOVE and self._prev:
            cv2.line(self.image,
                     self._prev, (x, y),
                     (255,), self.w)
            self._prev = (x, y)
        else:
            return
        self.update()

    def getMy(self) -> numpy.ndarray:
        output = cv2.resize(
            self.image,
            (28, 28),
            interpolation=cv2.INTER_AREA
        )
        output = output / 255.0
        output = output.reshape(1, 28, 28,1)
        return output
    
    def get(self) -> numpy.ndarray:
        output = cv2.resize(
            self.image,
            (28, 28),
            interpolation=cv2.INTER_AREA
        )
        output = output / 255.0
        output = output.reshape(1, 28, 28)
        return output


trained_modelMy = tf.keras.models.load_model('digitsDop.h5')
probability_modelMy = tf.keras.Sequential([
    trained_modelMy,
    tf.keras.layers.Softmax()  # слой softmax для нормализации
])
trained_model = tf.keras.models.load_model('digits.h5')
probability_model = tf.keras.Sequential([
    trained_model,
    tf.keras.layers.Softmax()  # слой softmax для нормализации
])
wnd = DrawingWindow('Draw a digit', 400)
wnd.update()
while True:
    key = cv2.waitKey()
    if key == 27:
        break
    elif key == 32:
        inp = wnd.get()
        inpMy = wnd.getMy()
        result = probability_model.predict(inp)
        answer = result[0].argmax()
        probability = result[0][answer]
        print(f'{answer} - {probability:.1%}')
        result = probability_modelMy.predict(inpMy)
        answer = result[0].argmax()
        probability = result[0][answer]
        print(f'{answer} - {probability:.1%}')


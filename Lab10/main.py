

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import sys
from pathlib import Path


DIR = Path(sys.argv[0]).parent.resolve()
MNIST = DIR / 'mnist.npz'
dataset = tf.keras.datasets.mnist.load_data(
    str(MNIST))
(train_images, train_labels), (test_images, test_labels) = dataset

print(f'Training images:  {train_images.shape} {train_images.dtype}')
print(f'    {train_images.min()}  {train_images.max()}')
print(f'Training labels:  {train_labels.shape} {train_labels.dtype}')
print(f'    {train_labels.min()}  {train_labels.max()}')
# нормализуем датасет
train_images = train_images / 255.0
test_images = test_images / 255.0

# формируем модель - структуру нейронной сети
model = tf.keras.Sequential([
    # простая сеть без обратных связей
    # вытягиваем матрицу пикселей в одну строку
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # скрытый 'плотный' слой на 32 нейрона
    tf.keras.layers.Dense(32, activation='relu'),
    # выходной слой для классификации
    tf.keras.layers.Dense(10)  # у нас 10 классов
])

model.compile(
    # оптимизатор - алгоритм обучения сети
    optimizer='adam',  # даём имя или объект алгоритма
    # функция потерь сети - определяет направление обучения
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    # метрики качества работы сети
    metrics=['accuracy']  # доля успешных классификаций
)
# обучение модели
model.fit(
    train_images,  # массив входных данных
    train_labels,  # массив 'правильных ответов'
    epochs=10,  # сколько эпох обучать сеть
    verbose=2  # детализация журнала работы
)
# проверка качества работы сети на контрольной выборке
test_loss, test_accuracy = model.evaluate(
    test_images,  # проверочные 'задания'
    test_labels,  # проверочные 'ответы'
    verbose=2
)
print(f'Accuracy: {test_accuracy:.1%}')
# сохраняем сеть
model.save(str(
    DIR / 'digits.h5'
))


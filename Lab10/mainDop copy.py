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

# нормализуем изображения
train_images = train_images / 255.0
test_images = test_images / 255.0

# Создание модели
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Компиляция модели
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

# Создание экземпляра класса ранней остановки
early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor='val_accuracy', 
      min_delta=0.00001, 
      patience=3
)

# Обучение модели
history = model.fit(
    train_images,
    train_labels,
    epochs=20,
    callbacks=[early_stopping],
    validation_data=(test_images, test_labels)
)

# Сохранение модели
model.save(str(
    DIR /'digitsDop.h5'
))
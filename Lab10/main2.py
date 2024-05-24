import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_datasets as tfds

import sys
from pathlib import Path
import matplotlib.pyplot as plt


AUTOTUNE = tf.data.experimental.AUTOTUNE

DIR = Path(sys.argv[0]).parent.resolve()
MNIST = DIR / 'mnist'
MNIST.mkdir(exist_ok=True)

# загружаем датасет MNIST с репозитория
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',  # имя датасета на tensorflow hub
    data_dir=str(MNIST),  # куда скачиваем
    split=['train', 'test'],  # нужно отдельно тренировочную и контрольную часть
    as_supervised=True,  # задание и ответы отдельно
    with_info=True  # добавить метаданные - описание
)
print(ds_info.splits['train'])


# нормализуем изображения в диапазон 0...1
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# конвейер для обучающей выборки
def train_pipeline(dataset):
    # нормализуем входные данные
    dataset = dataset.map(
        normalize_img,  # функция нормализации
        num_parallel_calls=AUTOTUNE
    )
    # кэшируем результат
    dataset = dataset.cache()
    # перемешиваем примеры в случайном порядке
    dataset = dataset.shuffle(
        ds_info.splits['train'].num_examples
    )
    # разбиваем датасет на порции
    dataset = dataset.batch(128)
    # предварительная выборка данных
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


# конвейер для контрольной выборки
def eval_pipeline(dataset):
    # нормализуем входные данные
    dataset = dataset.map(
        normalize_img,  # функция нормализации
        num_parallel_calls=AUTOTUNE
    )
    # разбиваем датасет на порции
    dataset = dataset.batch(128)
    # кэшируем результат
    dataset = dataset.cache()
    # предварительная выборка данных
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


# применяем конвейеры
ds_train = train_pipeline(ds_train)
ds_test = eval_pipeline(ds_test)

def train_model(layer, learning_rate):
    # формируем модель - структуру нейронной сети
    model = tf.keras.Sequential([
      # простая сеть без обратных связей
      # вытягиваем матрицу пикселей в одну строку
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      # скрытый 'плотный' слой на 32 нейрона
      *[tf.keras.layers.Dense(32, activation='relu') for _ in range(layer)],
      # выходной слой для классификации
      tf.keras.layers.Dense(10)  # у нас 10 классов
    ])

    model.compile(
      # оптимизатор - алгоритм обучения сети
      optimizer=tf.keras.optimizers.Adam(
          learning_rate=learning_rate  # скорость обучения сети
      ),  # даём имя или объект алгоритма
      # функция потерь сети - определяет направление обучения
      loss=tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True
      ),
      # метрики качества работы сети
      metrics=['accuracy']  # доля успешных классификаций
    )

    # обучение сети
    history = model.fit(
        ds_train,  # датасет для обучения
        validation_data=ds_test,  # датасет для проверки
        epochs=10,
        verbose=2
    )
    return history
    

def show_history(histories, key='accuracy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, 
                       history.history['val_accuracy'],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, 
                 history.history[key], 
                 color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.grid(True)
    plt.show()


history_64_1e2 = train_model(layer=1, learning_rate=1e-2)
history_64_1e3 = train_model(layer=1, learning_rate=1e-3)
history_64_1e4 = train_model(layer=1, learning_rate=1e-4)

# Обучение модели с двумя слоями по 32 нейрона
history_32_1e2 = train_model(layer=2, learning_rate=1e-2)
history_32_1e3 = train_model(layer=2, learning_rate=1e-3)
history_32_1e4 = train_model(layer=2, learning_rate=1e-4)

# Построение графиков
show_history([('64_1e-2', history_64_1e2),
              ('64_1e-3', history_64_1e3),
              ('64_1e-4', history_64_1e4),
              ('32_32_1e-2', history_32_1e2),
              ('32_32_1e-3', history_32_1e2),
              ('32_32_1e-4', history_32_1e2)])
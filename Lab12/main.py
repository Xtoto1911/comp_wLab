import sys
from pathlib import Path
import os

DIR = Path(sys.argv[0]).parent.resolve()
# детализация журнала работы tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# путь для кэширования моделей tensorflow-hub
os.environ['TFHUB_CACHE_DIR'] = str(DIR)
# показывать ли прогресс загрузки модели
os.environ['TFHUB_DOWNLOAD_PROGRESS'] = '1'
import tensorflow
import tensorflow_hub as tfhub
import numpy
import cv2


hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
# https://www.kagle.com/code/kerneler/arbitrary-image-stylization
hub_module = tfhub.load(hub_handle)
# описание основной модели
sig = hub_module.signatures['serving_default']
print('Входы модели: ')
for name, value in sig.structured_input_signature[1].items():
    print(f'    {name}: {value}')
print('Выходы модели: ')
for name, value in sig.structured_outputs.items():
    print(f'    {name}: {value}')


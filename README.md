# <div align="center">Оптическое распознавание номера автомобиля </div>
## <div align="center"> Описание проекта </div>
![Alt text](resources%2Fimages%2FArhitectur_EfficientNetV2L__OCR_car_plate.png)

За основу были взяты модель [EfficientNetV2L](https://www.tensorflow.org/versions/r2.12/api_docs/python/tf/keras/applications/efficientnet_v2/EfficientNetV2L), набор данных [nomeroff.net](https://nomeroff.net.ua/) и библиотека Tensorflow-2.14. Данные были разбиты на подвыборки для обучения, тестирования и валидации:

|Типп выборки данных | Количество изображений |
| :----------------: | :--------------------: |
| train | 37775 |
| val   | 4891  |
| test  | 2845  |

Разметка представленна как имя изображения соостветствующего номера заглавгыми буквами на английском. В номерах с двузначным номером региона вместо третьей цифры в конце номера должен быть добавлен символ "-".

Примеры изображений номеров:

|![Alt text](resources%2Fimages%2FA129XY196.png)|![Alt text](resources%2Fimages%2FK211PA69.png)|
| ------------------------------------------ | ------------------------------------------ |
|![Alt text](resources%2Fimages%2FE353TA46.png)|![Alt text](resources%2Fimages%2FP895HE96.png)|

## <div align="center"> Настройка среды </div>

Для настройки среды используйте ```requirements_ocr.txt```:
```commandline
pip install -r requirements_ocr.txt
```
Так же, можно запустить среду в ```Docker```. В таком случае будет использоваться ```requirements_ocr_for_docker.txt```.
Команда для сборки докера:
```commandline
docker build -t ocr .
```
Для работы в контейнере докера нужно подключить к контейнеру папку с данными ```data``` и папку для сохранения логов
и чекпоинтов модели ```logs```, а также папку с конфигурационным файлом ```Config.ini```. Перед запуском контейнера отредактируйте ```Config.ini ``` под ваш проект.
Пример запуска докер контейнера:
```commandline
docker run -v "$(pwd)"/../data/ocr:/ocr/data -v "$(pwd)"/logs:/ocr/logs -v /ocr/logs:"$(pwd)"/logs -v "$(pwd)"/Config:/ocr/Config --shm-size 16gb -it --rm --runtime=nvidia --gpus '"device=0"' ocr
```

## <div align="center"> Обучение </div>
Для запуска обучения отредактируйте раздел ```Train``` в ```Config.ini``` под свой проект и запустите скрипт ```train.py``` в своей среде или докер контейнере:
```commandline
python train.py
```
По умолчанию файлы логов обучения записываются в папку ```logs```. Логи сохраняются в формате Tensorboard. Пример команды для запуска Tensorboard:
```commandline
tensorboard --logdir logs --port 6015
```

## <div align="center"> Тестирование </div>
Для запуска тестирования отредактируйте раздел ```Test``` в ```Config.ini``` под свой проект и запустите скрипт ```test.py``` в своей среде или докер контейнере:
```commandline
python test.py
```
Результаты тестирования сохраняются в папке с моделью.


## <div align="center">  Результаты тестирования обученной модели </div>
Модель обучалась на протяжении 200 эпох с размером батча 32 и размером цветных изображений 200 х 50 пикселей. Значение шага обучения изменялось от 1e-4 до 0 по закону косинусного распада (CosineDecay) с прогревом (warmup) в 5 эпох.  
Результаты тестирования на тестовой выборке данных:

| Название метрики | Доля, %|
| :--------------- | :-----: |
| Доля распознанных символов | 98,4 |
| Доля распознанных номеров | 94,1 |

Параметры модели:

| Название параметра | Значение|
| :--------------- | :-----: |
| Размер в формате ONNX |  133.3 MB|
| Количество параметров | 33,5 M |
| Время инфернса* | 33,9 мc. |

\* Время инференса замерялось на ноутбуке с RTX 3060 Laptop и Intel Core i9-12900H, при batch_size=1 и размере цветного изображения 200 х 50.


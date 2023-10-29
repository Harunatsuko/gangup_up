# Решение команды GANg up! для хакатона Цифровой Прорыв 2023. Сезон: ИИ.

Кейс Разработка системы видео-аналитики подсчета ТБО. 
Решение представляет собой автоматизированную систему подсчета твёрдых бытовых отдходов.

### 1. Описание
В ходе решения кейса было разработано рабочее место оператора для возможности контроля человеком процесса разделения ТБО и своевременной реакции на возможные отклонения в работе.

Для создание подобной системы были решены следующие задачи:
- проведена работа по обогащению исходного датасета с помощью алгоритмов аугментации;
- обучена модель YOLOv5 для детекции объектов;
- создан алгоритм трекинга объектов на видео;
- разработана линейная модель для предсказания местоположения конкретного объекта на видео. Более того, разработанный нами трекер может обнаруживать остановку конвейера.
- разработан интерфейс, с помощью которого оператор станции переработки ТБО может отслеживать процесс работы модели с экрана своего рабочего ПК.

Стек решения: Python, YOLOv5, openCV, Django, Vue, Celery, Redis.

### 2. Структура репозитория

Решение состоит из следующих компонентов: 

- `backend`: Директория с кодом фронтенда интерфейса.
- `frontend`: Директория с кодом фронтенда интерфейса.
- `model`: Директория с кодом обучения модели детекции объектов.
- `synth_data`: Директория с кодом для генерации синтетическихданных данных.
- `tracking`: Директория с кодом алгоритма трекинга объектов на видео.

```
├── README.md
├── __init__.py
├── backend
├── docker-compose.yml
├── frontend
├── model
├── synth_data
└── tracking
```

### 2. Входные и выходные данные пайплайна

Пайплайну на вход должны приходить данные вида `[rgb_frame: PILImage, ms_frame: PILImage, frame_idx]`.

Пайплайн возвращает список вида: `[[obj_id: int, class_label: int, x: int, y:int, w:int, h:int], …]`
и список фреймов с отрисованными боксами.


### 3. Запуск алгоритма трекинга

Пример использования скрипта `tracking/tracking.py`:
```
traker = Traker('yolov5_v6_640_640.onnx')
for i, framep in enumerate(frames_pths):
     frame_rgb = Image.open(framep)
     frame_ms = Image.open(framep.replace('frames_rgb',
                                 'frames_ms').replace('png','tif'))
     objs, rgb_img, gray_imgs = traker.track(frame_rgb,frame_ms,i, framep)
```



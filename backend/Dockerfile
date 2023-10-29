# образ на основе которого создаём контейнер
FROM python:3.8

# рабочая директория внутри проекта
WORKDIR /usr/src/app

# устанавливаем зависимости
RUN apt-get update
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade cython
RUN pip install --upgrade pip setuptools wheel
RUN pip install psycopg2
COPY ./requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
# копируем содержимое текущей папки в контейнер
COPY . .
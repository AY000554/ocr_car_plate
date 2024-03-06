FROM tensorflow/tensorflow:2.14.0-gpu

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 graphviz -y

WORKDIR /ocr_car_plate

COPY requirements_ocr_for_docker.txt /ocr_car_plate/ 

RUN pip install -r requirements_ocr_for_docker.txt

COPY *.py /ocr_car_plate/

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Dockerfile para echar a andar un contenedor especifico

FROM tensorflow/tensorflow

RUN pip install numpy
RUN pip install tensorflow
RUN pip install tensorflow_hub
RUN pip install matplotlib
RUN pip install scipy

ADD Emisiones_1.py .

WORKDIR $PWD

CMD ["python", "./Emisiones_1.py"]

FROM tensorflow/tensorflow

WORKDIR /app
ADD . /app
RUN pip install tensorflow_datasets && pip install matplotlib
CMD cd /app && python main.py
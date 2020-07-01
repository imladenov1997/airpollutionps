FROM python:3.6
ADD . /server-side
WORKDIR /server-side
EXPOSE 5000
RUN python -m pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.13.1-py3-none-any.whl
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
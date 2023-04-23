FROM python:3.8-buster

ADD . /opt/ml_in_app
WORKDIR /opt/ml_in_app

RUN pip install -r requirements.txt
CMD ["python", "app.py"]
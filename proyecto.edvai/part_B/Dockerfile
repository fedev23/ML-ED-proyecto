FROM python:3.9


WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

CMD ["uvicorn", "modulos.main:app", "--host=127.0.0.1", "--port=7860"]
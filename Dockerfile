FROM python:3.8

WORKDIR /fastapi-app

RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

CMD ["uvicorn"  , "app.main_api:app", "--host", "0.0.0.0", "--port=8000"]
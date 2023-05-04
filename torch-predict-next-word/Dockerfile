FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app

#RUN pip install --no-cache-dir torch flask gunicorn
RUN pip install torch flask

EXPOSE 8000

CMD ["python", "main.py"]
#CMD ["gunicorn", "wsgi:app"]
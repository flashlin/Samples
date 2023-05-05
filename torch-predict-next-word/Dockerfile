FROM python:3.9-slim-buster

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir torch flask gunicorn

EXPOSE 8000
ENV PY_ENVIRONMENT="Production"
CMD ["python", "main.py"]
#CMD ["gunicorn", "wsgi:app"]
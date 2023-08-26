FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/

ENV REPLICATE_API_TOKEN=""

# Heroku uses PORT, Azure App Services uses WEBSITES_PORT, Fly.io uses 8080 by default
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 80"]

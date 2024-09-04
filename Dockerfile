FROM python:3.11

RUN apt-get update && apt-get install -y \
    libpq-dev gcc --no-install-recommends

WORKDIR /app

COPY requirements.txt .

RUN  pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_RUN_PORT 6000

COPY . .

CMD ["python", "app.py"]
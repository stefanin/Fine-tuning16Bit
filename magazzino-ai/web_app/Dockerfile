FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia l'intera applicazione, inclusa la cartella 'templates'
COPY . .

CMD ["python", "app.py"]
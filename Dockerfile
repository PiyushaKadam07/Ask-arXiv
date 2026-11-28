FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model while building, so a cold start doesn't need the internet
ENV FASTEMBED_CACHE=/app/models
COPY search.py .
RUN python -c "from search import fastembed_model; fastembed_model()"

COPY . .

ENV PORT=8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
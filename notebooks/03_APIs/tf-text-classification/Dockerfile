FROM python:3.6-slim

# Install dependencies
# Do this first for caching
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy
COPY embeddings embeddings
COPY experiments experiments
COPY text_classification text_classification
COPY app.py app.py
COPY config.py config.py
COPY logging.json logging.json
COPY utilities.py utilities.py

# Export ports
EXPOSE 5000

# Start app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
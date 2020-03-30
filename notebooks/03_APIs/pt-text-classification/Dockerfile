FROM python:3.6-slim

# Install dependencies
# Do this first for caching
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy
COPY embeddings embeddings
COPY experiments experiments
COPY text_classification text_classification
COPY logging.json logging.json

# Export ports
EXPOSE 5000

# Start app
CMD ["uvicorn", "text_classification.app:app", "--host", "0.0.0.0", "--port", "5000"]
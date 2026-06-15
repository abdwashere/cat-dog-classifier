FROM python:3.10-slim

WORKDIR /app

# Install torch separately first (so it stays cached)
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2

# Install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY models/ ./models/
COPY params.yaml .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
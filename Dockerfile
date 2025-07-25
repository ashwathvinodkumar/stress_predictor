FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for building wheels
RUN apt-get update && \
    apt-get install -y gcc g++ build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY main.py .
COPY tamil_nadu_stress_model.pkl .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

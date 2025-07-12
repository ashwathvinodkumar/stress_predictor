FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY main.py .
COPY tamil_nadu_stress_model.pkl .

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files, models, and demo sample only (keep image compact)
COPY app/ ./app/
COPY models/ ./models/
COPY outputs/ ./outputs/
COPY data/sample/ ./data/sample/
COPY start.sh .

RUN chmod +x start.sh

# Expose ports: 8000 (FastAPI), 7860 (Streamlit / Hugging Face Spaces default)
EXPOSE 8000 7860

CMD ["./start.sh"]

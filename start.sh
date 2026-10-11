#!/bin/bash
# Startup script for single container deployment (Hugging Face Spaces & Docker)

echo "Starting FastAPI backend on port 8000..."
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 &

# Wait for backend to become healthy
echo "Waiting for FastAPI backend to initialize..."
while ! curl -s http://localhost:8000/health | grep -q "healthy"; do
    sleep 1
done
echo "Backend is live and healthy!"

echo "Starting Streamlit frontend on port 7860..."
export BACKEND_URL="http://localhost:8000"
streamlit run app/frontend/app.py --server.port 7860 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false

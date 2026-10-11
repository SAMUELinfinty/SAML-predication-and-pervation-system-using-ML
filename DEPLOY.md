# Deployment & Execution Guide

This document describes the three verified deployment pathways for the **Explainable ML Pipeline for AML Fraud Detection & Business Risk Management**:

---

## (A) Local Execution (Development & Testing)

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ is installed:
```bash
# Clone or navigate to the repository
cd "e:/Devlopment/Have fun in projects/SAML predication and pervation system using ML"

# Install dependencies
pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution
Run the complete pipeline from raw data to trained models and outputs:
```bash
# Full pipeline run
python run_all.py

# Or fast test run on a 5% sample:
python run_all.py --sample
```

### 3. Start Backend & Frontend
In Terminal 1 (FastAPI Backend):
```bash
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Test health check at: `http://localhost:8000/health` (Interactive Swagger docs: `http://localhost:8000/docs`).

In Terminal 2 (Streamlit UI):
```bash
streamlit run app/frontend/app.py --server.port 7860
```
Open web browser at `http://localhost:7860`.

---

## (B) Docker Containerization

### Single-Container Deployment (Recommended for Cloud / Spaces)
Build and run the unified image containing both FastAPI and Streamlit:
```bash
# Build Docker image
docker build -t aml-fraud-detector .

# Run container exposing ports 8000 and 7860
docker run -p 8000:8000 -p 7860:7860 aml-fraud-detector
```

### Multi-Container Deployment via Docker Compose
To run backend and frontend in isolated communicating services:
```bash
docker-compose up --build
```
Access UI at `http://localhost:7860` and API at `http://localhost:8000`.

---

## (C) Hugging Face Spaces Deployment

Hugging Face Spaces supports standalone Docker containers seamlessly.

### 1. Spaces Metadata (README.md Front-Matter)
When creating a new Space on [Hugging Face](https://huggingface.co/new-space), select **Docker** SDK.
Your `README.md` must begin with the following front-matter:

```yaml
---
title: Explainable AML Fraud Detection & Risk BI
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---
```

### 2. Push Instructions
Push your repo to Hugging Face Spaces:
```bash
# Add Hugging Face Space remote
git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>

# Push code and models (ensure Git LFS is configured for ensemble.joblib if needed)
git push space main
```
The Space will automatically build `Dockerfile`, launch `start.sh`, bind Streamlit to `0.0.0.0:7860`, and serve the live application.

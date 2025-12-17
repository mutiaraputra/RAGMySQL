#!/bin/bash

# ============================================
# KONFIGURASI - Ganti dengan nilai Anda
# ============================================
PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"  # Set via: export GCP_PROJECT_ID=xxx
REGION="asia-southeast2"
SERVICE_NAME="rag-chatbot"

# Validasi PROJECT_ID
if [ "$PROJECT_ID" = "your-gcp-project-id" ]; then
    echo "ERROR: Please set GCP_PROJECT_ID environment variable"
    echo "Usage: export GCP_PROJECT_ID=your-actual-project-id && ./deploy.sh"
    exit 1
fi

gcloud services enable cloudbuild.googleapis.com run.googleapis.com secretmanager.googleapis.com

gcloud builds submit --config cloudbuild.yaml

gcloud run services update $SERVICE_NAME \
    --region=$REGION \
    --update-secrets=PINECONE__API_KEY=pinecone-api-key:latest \
    --update-secrets=PINECONE__INDEX_URL=pinecone-index-url:latest \
    --update-secrets=GEMINI__API_KEY=gemini-api-key:latest \
    --set-env-vars="APP__EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2" \
    --set-env-vars="APP__EMBEDDING_DIMENSION=384" \
    --allow-unauthenticated
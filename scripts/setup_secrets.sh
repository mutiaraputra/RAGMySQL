#!/bin/bash

PROJECT_ID="your-gcp-project-id"

# Create secrets
gcloud secrets create json-endpoint-url --data-file=- <<< "$JSON_ENDPOINT__URL"
gcloud secrets create pinecone-api-key --data-file=- <<< "$PINECONE__API_KEY"
gcloud secrets create pinecone-index-url --data-file=- <<< "$PINECONE__INDEX_URL"
gcloud secrets create gemini-api-key --data-file=- <<< "$GEMINI__API_KEY"

# Grant access to Cloud Run service account
SERVICE_ACCOUNT="your-service-account@$PROJECT_ID.iam.gserviceaccount.com"

gcloud secrets add-iam-policy-binding json-endpoint-url \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding pinecone-api-key \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding pinecone-index-url \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding gemini-api-key \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"
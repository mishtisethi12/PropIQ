#!/usr/bin/env bash
# ==============================================================================
# PropIQ - Google Cloud Run Automated Deployment Script
# Provisions Artifact Registry, builds and pushes image, and deploys to Cloud Run.
# ==============================================================================
set -euo pipefail

# Configuration Defaults (override via environment)
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo 'your-gcp-project-id')}"
REGION="${GCP_REGION:-us-central1}"
REPO_NAME="propiq-repo"
SERVICE_NAME="propiq-app"
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"

echo "========================================================"
echo "DEPLOYING PROPIQ TO GOOGLE CLOUD RUN"
echo "Project ID: ${PROJECT_ID}"
echo "Region:     ${REGION}"
echo "========================================================"

# 1. Enable Required GCP APIs
echo "==> Enabling GCP Cloud Services..."
gcloud services enable \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    sqladmin.googleapis.com \
    secretmanager.googleapis.com \
    --project="${PROJECT_ID}"

# 2. Create Artifact Registry Docker Repository (if not existing)
echo "==> Provisioning Artifact Registry repository..."
if ! gcloud artifacts repositories describe "${REPO_NAME}" --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    gcloud artifacts repositories create "${REPO_NAME}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="PropIQ Docker Container Repository" \
        --project="${PROJECT_ID}"
    echo "[OK] Created repository ${REPO_NAME} in ${REGION}"
else
    echo "[OK] Repository ${REPO_NAME} already exists."
fi

# 3. Authenticate Docker with Google Artifact Registry
echo "==> Configuring Docker credentials..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# 4. Build and Push Docker Image
echo "==> Building and pushing Docker container image..."
docker build -t "${IMAGE_TAG}" -f Dockerfile .
docker push "${IMAGE_TAG}"
echo "[OK] Successfully pushed image: ${IMAGE_TAG}"

# 5. Deploy to Google Cloud Run
echo "==> Deploying container to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE_TAG}" \
    --region="${REGION}" \
    --platform=managed \
    --port=8501 \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=1 \
    --max-instances=10 \
    --allow-unauthenticated \
    --set-env-vars="STREAMLIT_SERVER_PORT=8501,STREAMLIT_SERVER_ADDRESS=0.0.0.0,STREAMLIT_SERVER_HEADLESS=true" \
    --project="${PROJECT_ID}"

# 6. Retrieve Service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --platform=managed --region="${REGION}" --format="value(status.url)" --project="${PROJECT_ID}")

echo "========================================================"
echo "PROPIQ CLOUD RUN DEPLOYMENT COMPLETE!"
echo "Service URL: ${SERVICE_URL}"
echo "========================================================"

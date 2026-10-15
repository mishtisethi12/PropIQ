# PropIQ: Production Cloud & Container Deployment Guide

This document outlines the end-to-end procedures for containerizing the PropIQ analytics platform and deploying the multi-container architecture across **Local Docker Compose**, **Amazon Web Services (AWS)**, and **Google Cloud Platform (GCP)**.

---

## 1. System Architecture Overview

```mermaid
flowchart TD
    subgraph Client_Access["Client Access Tier"]
        User["End User / Investment Analyst"]
        Browser["Modern Web Browser"]
    end

    subgraph Cloud_Ingress["Cloud Ingress & Networking"]
        ALB["Application Load Balancer / Cloud Run Ingress"]
        SSL["TLS / SSL Certificate (HTTPS:443)"]
    end

    subgraph Container_Runtime["Application Runtime Tier"]
        Streamlit["PropIQ Streamlit Web App (Port 8501)"]
        ML_Engine["ML Model Inference (Decision Forest, SVM, k-NN, NB)"]
        RAG_Engine["ChromaDB Vector Store (all-MiniLM-L6-v2)"]
    end

    subgraph Storage_Tier["Data Persistence Tier"]
        PostgreSQL["PostgreSQL 16 OLTP (Raw Listings & DDL)"]
        DuckDB["DuckDB OLAP (Star Schema & Partitioning)"]
        Parquet["Partitioned Parquet Lakehouse (City/Month)"]
    end

    User --> Browser
    Browser --> ALB
    ALB --> SSL
    SSL --> Streamlit
    Streamlit --> ML_Engine
    Streamlit --> RAG_Engine
    Streamlit --> PostgreSQL
    Streamlit --> DuckDB
    DuckDB --> Parquet
```

---

## 2. Local Multi-Container Deployment (One-Command Launch)

Both the Streamlit web platform and the PostgreSQL 16 database launch simultaneously with healthcheck dependency resolution using Docker Compose.

### Quick Start

1. **Verify Docker Engine**:
   Ensure Docker Desktop (or Docker Engine on Linux) is running:
   ```bash
   docker --version
   docker compose version
   ```

2. **Launch Platform**:
   From the project root directory, execute:
   ```bash
   docker compose up --build -d
   ```

3. **Verify Service Health**:
   ```bash
   docker compose ps
   ```
   Expected output:
   | Name | Service | Status | Ports |
   | :--- | :--- | :--- | :--- |
   | `propiq_postgres` | `postgres` | `Up (healthy)` | `0.0.0.0:5432->5432/tcp` |
   | `propiq_app` | `app` | `Up (healthy)` | `0.0.0.0:8501->8501/tcp` |

4. **Access the Interfaces**:
   - **Streamlit Analytics Dashboard**: [http://localhost:8501](http://localhost:8501)
   - **PostgreSQL Database**: `localhost:5432` (`propiq_db`, user: `postgres`, password: `postgrespassword`)

5. **Stop Platform**:
   ```bash
   docker compose down
   # Or remove volumes to reset database state:
   docker compose down -v
   ```

---

## 3. AWS Deployment Guide

We provide two deployment options for AWS:
- **Option A (Enterprise Serverless)**: Amazon ECS (Fargate) + Amazon RDS PostgreSQL + Application Load Balancer
- **Option B (Cost-Optimized VM)**: Amazon EC2 / Lightsail running Docker Compose

---

### Option A: Amazon ECS Fargate + RDS PostgreSQL (Recommended)

#### Step 1: Create Amazon ECR Repository and Push Image
```bash
# 1. Set environment variables
export AWS_REGION="us-east-1"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO="propiq"

# 2. Create ECR repository
aws ecr create-repository \
    --repository-name $ECR_REPO \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true

# 3. Authenticate Docker with ECR
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# 4. Build and tag image
docker build -t $ECR_REPO:latest -f Dockerfile .
docker tag $ECR_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

# 5. Push image to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
```

#### Step 2: Provision Amazon RDS PostgreSQL Instance
```bash
# Create RDS PostgreSQL DB Subnet Group and Instance
aws rds create-db-instance \
    --db-instance-identifier propiq-postgres-db \
    --db-instance-class db.t4g.micro \
    --engine postgres \
    --engine-version 16.2 \
    --master-username postgres \
    --master-user-password "YourStrongPassword123" \
    --allocated-storage 20 \
    --db-name propiq_db \
    --vpc-security-group-ids "sg-xxxxxxx" \
    --backup-retention-period 7 \
    --no-publicly-accessible
```

#### Step 3: Store Database Secret in AWS Secrets Manager
```bash
aws secretsmanager create-secret \
    --name propiq/postgres_url \
    --secret-string "postgresql+psycopg2://postgres:YourStrongPassword123@propiq-postgres-db.xxxxxx.us-east-1.rds.amazonaws.com:5432/propiq_db"

aws secretsmanager create-secret \
    --name propiq/gemini_api_key \
    --secret-string "YOUR_GEMINI_API_KEY_HERE"
```

#### Step 4: Register ECS Task Definition and Launch Service
```bash
# Substitute your AWS Account ID and Region in deploy/aws/ecs-task-definition.json:
sed -i "s/ACCOUNT_ID/$AWS_ACCOUNT_ID/g" deploy/aws/ecs-task-definition.json
sed -i "s/REGION/$AWS_REGION/g" deploy/aws/ecs-task-definition.json

# Register task definition
aws ecs register-task-definition \
    --cli-input-json file://deploy/aws/ecs-task-definition.json

# Create ECS Fargate Cluster
aws ecs create-cluster --cluster-name propiq-cluster

# Create Fargate Service attached to Application Load Balancer
aws ecs create-service \
    --cluster propiq-cluster \
    --service-name propiq-service \
    --task-definition propiq-app-task \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxx,subnet-yyyy],securityGroups=[sg-zzzz],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=propiq-streamlit,containerPort=8501"
```

---

### Option B: Amazon EC2 / Lightsail VM with Docker Compose

For rapid prototyping or single-instance evaluation ($10–$20/mo):

1. **Launch Ubuntu 22.04 LTS Instance** (`t3.medium` or `t4g.medium`, 2 vCPU, 4GB RAM).
2. **Assign Security Group**:
   - Inbound Port `8501`: Streamlit UI (or Port `443` via Caddy/Nginx reverse proxy)
   - Inbound Port `22`: SSH administrative access
3. **Run User Data Script**:
   Attach [`deploy/aws/ec2-userdata.sh`](file:///C:/Users/misht/.gemini/antigravity/scratch/PropIQ/deploy/aws/ec2-userdata.sh) at launch or run manually via SSH:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/your-org/PropIQ/master/deploy/aws/ec2-userdata.sh | sudo bash
   ```

---

## 4. GCP Deployment Guide

We provide two deployment options for Google Cloud:
- **Option A (Fully Managed Serverless)**: Google Cloud Run + Cloud SQL PostgreSQL
- **Option B (Compute Engine VM)**: GCE Ubuntu VM with Docker Compose

---

### Option A: Google Cloud Run + Cloud SQL (Recommended)

#### Step 1: Enable Google Cloud APIs
```bash
export GCP_PROJECT_ID="your-gcp-project-id"
export GCP_REGION="us-central1"

gcloud config set project $GCP_PROJECT_ID

gcloud services enable \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    sqladmin.googleapis.com \
    secretmanager.googleapis.com \
    vpcaccess.googleapis.com
```

#### Step 2: Provision Cloud SQL PostgreSQL 16
```bash
gcloud sql instances create propiq-postgres-instance \
    --database-version=POSTGRES_16 \
    --tier=db-custom-1-3840 \
    --region=$GCP_REGION \
    --root-password="YourStrongPassword123"

# Create database
gcloud sql databases create propiq_db --instance=propiq-postgres-instance
```

#### Step 3: Build & Push Container Image to Google Artifact Registry
```bash
# Create Artifact Registry repository
gcloud artifacts repositories create propiq-repo \
    --repository-format=docker \
    --location=$GCP_REGION \
    --description="PropIQ Docker Images"

# Configure Docker credentials
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev

# Build and Push using Cloud Build or Local Docker
gcloud builds submit --config=deploy/gcp/cloudbuild.yaml
```

#### Step 4: Deploy Container to Google Cloud Run
```bash
# Execute automated deployment script
bash deploy/gcp/deploy_cloud_run.sh
```

Or deploy manually via `gcloud`:
```bash
gcloud run deploy propiq-app \
    --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/propiq-repo/propiq-app:latest" \
    --region="$GCP_REGION" \
    --platform=managed \
    --port=8501 \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=1 \
    --max-instances=10 \
    --allow-unauthenticated \
    --add-cloudsql-instances="$GCP_PROJECT_ID:$GCP_REGION:propiq-postgres-instance" \
    --set-env-vars="POSTGRES_URL=postgresql+psycopg2://postgres:YourStrongPassword123@/propiq_db?host=/cloudsql/$GCP_PROJECT_ID:$GCP_REGION:propiq-postgres-instance,STREAMLIT_SERVER_PORT=8501,STREAMLIT_SERVER_ADDRESS=0.0.0.0,STREAMLIT_SERVER_HEADLESS=true"
```

---

## 5. Environment Variables & Secret Configuration

| Variable Name | Required | Default Local Value | Cloud Production Recommendation |
| :--- | :---: | :--- | :--- |
| `POSTGRES_URL` | Yes | `postgresql+psycopg2://postgres:postgrespassword@localhost:5432/propiq_db` | AWS Secrets Manager / GCP Secret Manager |
| `DATABASE_URL` | Optional | Alias to `POSTGRES_URL` | Cloud SQL socket or RDS endpoint |
| `GEMINI_API_KEY` | Optional | `""` (runs offline fallback synthesis) | Google AI Studio key stored in Secrets Manager |
| `STREAMLIT_SERVER_PORT` | Yes | `8501` | `8501` (mapped to 443 via ALB / Cloud Run ingress) |
| `STREAMLIT_SERVER_ADDRESS`| Yes | `0.0.0.0` | `0.0.0.0` (binds to all network interfaces) |
| `STREAMLIT_SERVER_HEADLESS`| Yes | `true` | `true` (disables interactive browser launching) |

---

## 6. Healthchecks, Monitoring & Rollbacks

- **Container Healthcheck**:
  Streamlit provides an internal healthcheck endpoint at `http://localhost:8501/_stcore/health`.
  Docker Compose and AWS/GCP load balancers use this endpoint to detect container unresponsiveness within 15 seconds.
- **Logging**:
  - AWS: Captured via AWS CloudWatch Logs at `/ecs/propiq-production`.
  - GCP: Ingested automatically into Google Cloud Logging (`projects/$PROJECT_ID/logs/run.googleapis.com`).
- **Rollback Procedure**:
  - AWS ECS: `aws ecs update-service --cluster propiq-cluster --service propiq-service --task-definition propiq-app-task:PREVIOUS_REVISION`
  - GCP Cloud Run: `gcloud run services update-traffic propiq-app --to-revisions=PREVIOUS_REVISION_NAME=100`

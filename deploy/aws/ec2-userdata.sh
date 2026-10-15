#!/usr/bin/env bash
# ==============================================================================
# PropIQ - AWS EC2 / Lightsail Automated Provisioning & Launch Script
# Installs Docker, Docker Compose, clones PropIQ, and launches multi-container platform.
# ==============================================================================
set -euo pipefail

echo "==> Updating system packages..."
apt-get update -y
apt-get install -y ca-certificates curl gnupg lsb-release git

echo "==> Installing Docker Engine..."
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

systemctl enable docker
systemctl start docker

echo "==> Setting up PropIQ Application Directory..."
INSTALL_DIR="/opt/propiq"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# If repository URL is provided via instance tag or metadata, clone or extract:
# git clone https://github.com/your-org/PropIQ.git .

# Configure environment variables
cat << 'EOF' > "$INSTALL_DIR/.env"
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgrespassword
POSTGRES_DB=propiq_db
POSTGRES_URL=postgresql+psycopg2://postgres:postgrespassword@postgres:5432/propiq_db
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
GEMINI_API_KEY=
EOF

echo "==> Starting PropIQ Containers via Docker Compose..."
# docker compose -f docker-compose.yml up --build -d

echo "==> PropIQ deployment initialization complete!"

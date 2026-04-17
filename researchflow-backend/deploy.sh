#!/bin/bash
# ResearchFlow — One-click production deployment
# Usage: bash deploy.sh
#
# Prerequisites:
#   - Ubuntu 22.04+ with Docker + Docker Compose installed
#   - .env file configured (see .env.example)
#   - Domain DNS pointed to this server (for HTTPS)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "  ResearchFlow Production Deployment"
echo "========================================="

# ── Check prerequisites ────────────────────────────────────────

if ! command -v docker &>/dev/null; then
    echo "[!] Docker not found. Installing..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "[!] Docker installed. Please log out and back in, then re-run this script."
    exit 1
fi

if ! docker compose version &>/dev/null; then
    echo "[!] Docker Compose not found."
    exit 1
fi

# ── Check .env ─────────────────────────────────────────────────

if [ ! -f .env ]; then
    echo "[!] .env file not found. Creating from template..."
    cp .env.example .env
    echo ""
    echo "  Please edit .env and set at minimum:"
    echo "    POSTGRES_PASSWORD=<a strong random password>"
    echo "    ANTHROPIC_API_KEY=<your key>"
    echo "    DOMAIN=<your domain, e.g. rf.example.com>"
    echo ""
    echo "  Then re-run: bash deploy.sh"
    exit 1
fi

source .env

if [ "$POSTGRES_PASSWORD" = "changeme" ]; then
    echo "[!] POSTGRES_PASSWORD is still 'changeme'. Please set a real password in .env"
    exit 1
fi

echo "[OK] .env loaded"
echo "  Domain: ${DOMAIN:-localhost}"
echo "  Storage: ${OBJECT_STORAGE_PROVIDER:-local}"
echo "  LLM: $([ -n "$ANTHROPIC_API_KEY" ] && echo 'Anthropic' || echo 'Mock')"

# ── Build and start ────────────────────────────────────────────

echo ""
echo "[*] Building Docker images..."
docker compose -f docker-compose.prod.yml build

echo ""
echo "[*] Starting services..."
docker compose -f docker-compose.prod.yml up -d

echo ""
echo "[*] Waiting for PostgreSQL to be ready..."
for i in $(seq 1 30); do
    if docker compose -f docker-compose.prod.yml exec -T postgres pg_isready -U rf -d researchflow &>/dev/null; then
        echo "[OK] PostgreSQL is ready"
        break
    fi
    sleep 2
done

# ── Run migrations ─────────────────────────────────────────────

echo ""
echo "[*] Running database migrations..."
docker compose -f docker-compose.prod.yml exec -T api alembic -c alembic/alembic.ini upgrade head

echo ""
echo "[*] Enabling pgvector extension..."
docker compose -f docker-compose.prod.yml exec -T postgres psql -U rf -d researchflow -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true

# ── Health check ───────────────────────────────────────────────

echo ""
echo "[*] Checking service health..."
sleep 3

API_STATUS=$(docker compose -f docker-compose.prod.yml exec -T api curl -s http://localhost:8000/api/v1/health 2>/dev/null || echo "failed")
echo "  API: $API_STATUS"

echo ""
echo "========================================="
echo "  Deployment complete!"
echo "========================================="
echo ""
echo "  Web UI:    https://${DOMAIN:-localhost}"
echo "  API Docs:  https://${DOMAIN:-localhost}/docs"
echo "  API:       https://${DOMAIN:-localhost}/api/v1/health"
echo ""
echo "  Useful commands:"
echo "    docker compose -f docker-compose.prod.yml logs -f        # View logs"
echo "    docker compose -f docker-compose.prod.yml ps             # Service status"
echo "    docker compose -f docker-compose.prod.yml restart api    # Restart API"
echo "    docker compose -f docker-compose.prod.yml down           # Stop all"
echo ""
echo "  To import existing data from paperAnalysis/:"
echo "    docker compose -f docker-compose.prod.yml exec api python -m migration.migrate_csv_to_db"
echo "    docker compose -f docker-compose.prod.yml exec api python -m migration.migrate_md_to_db"
echo ""

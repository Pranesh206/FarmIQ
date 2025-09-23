#!/bin/bash
# FarmIQ Deploy Script

set -e  # Exit on error

PLATFORM=$1
if [ -z "$PLATFORM" ]; then
    echo "Usage: ./scripts/deploy.sh <heroku|docker|compose>"
    exit 1
fi

echo "[DEPLOY] Starting deployment for $PLATFORM..."

# Load env
if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi

case $PLATFORM in
    "heroku")
        echo "[DEPLOY] Deploying to Heroku..."
        if ! command -v heroku &> /dev/null; then
            echo "Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli"
            exit 1
        fi
        git push heroku main
        heroku run python -c "from app import db; db.create_all()"  # Init DB
        heroku open
        ;;
    "docker")
        echo "[DEPLOY] Building and running Docker..."
        docker build -t farmiq .
        docker run -d -p 5000:5000 --env-file .env farmiq
        echo "[DEPLOY] App running at http://localhost:5000"
        curl http://localhost:5000/health  # Test
        ;;
    "compose")
        echo "[DEPLOY] Starting Docker Compose..."
        docker-compose up -d --build
        echo "[DEPLOY] App running at http://localhost:5000"
        curl http://localhost:5000/health
        ;;
    *)
        echo "Unsupported platform: $PLATFORM"
        exit 1
        ;;
esac

echo "[DEPLOY] Deployment complete!"
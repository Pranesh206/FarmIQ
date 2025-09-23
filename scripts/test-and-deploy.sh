#!/bin/bash
# Test and Deploy Script

echo "[DEPLOY] Running tests..."
pytest test_backend.py test_data.py -v || { echo "Tests failed!"; exit 1; }

echo "[DEPLOY] Tests passed. Deploying..."
./deploy.sh heroku  # Default to Heroku; change as needed

echo "[DEPLOY] Done!"
#!/bin/bash
# DB Migration Script (for Heroku release phase or local)

echo "[DEPLOY] Running DB migrations..."

# Activate virtualenv if needed (local)
# source venv/bin/activate  # Uncomment for local

python -c "
from app import app, db
from models.database import Farmer  # Adjust imports
with app.app_context():
    db
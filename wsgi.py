#!/usr/bin/env python
"""WSGI entrypoint for Gunicorn."""

import os
from dotenv import load_dotenv  # [DEPLOY] Load env vars
load_dotenv()

from app import app as application  # Import the Flask app

if __name__ == "__main__":
    application.run()
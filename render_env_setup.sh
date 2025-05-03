#!/bin/bash
# This script installs additional dependencies for Render.com

# Install gunicorn
pip install gunicorn

# Install psycopg2 binary (pre-compiled)
pip install psycopg2-binary

# Install any other system dependencies
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* 
#!/usr/bin/env bash
# render_build.sh
set -o errexit

echo "Setting environment variables for Render.com deployment..."

# Set environment variable to indicate this is running on Render
export RENDER=true
export ENV=production

echo "Install Python dependencies..."
pip install -r requirements.txt

echo "Build complete!" 
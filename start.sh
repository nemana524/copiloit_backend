#!/bin/bash
set -e

# Log start time and environment
echo "Starting application at $(date)"
echo "Environment: $ENV"

# Set environment variables
export RENDER=true
export ENV=production

# Wait a bit for database to be ready if needed
echo "Waiting for 3 seconds to ensure database is ready..."
sleep 3

# Start the application with proper settings for Render.com
echo "Starting Gunicorn with app:app..."
exec gunicorn -k uvicorn.workers.UvicornWorker \
  --access-logfile - \
  --error-logfile - \
  --log-level info \
  --timeout 120 \
  app:app \
  --bind 0.0.0.0:$PORT 
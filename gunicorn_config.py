# Gunicorn configuration for production deployment on Render.com
import os

# Worker configuration
workers = 4  # Adjust based on your plan's resources
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120

# Bind to the port provided by Render
port = os.environ.get("PORT", "8000")
bind = f"0.0.0.0:{port}"

# Log configuration
loglevel = "info"
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr

# Keep-alive settings
keepalive = 65 
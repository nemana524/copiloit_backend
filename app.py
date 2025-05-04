import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log environment
logger.info(f"Starting app.py in environment: {'production' if os.environ.get('ENV') == 'production' else 'development'}")
if os.environ.get("RENDER"):
    logger.info("Running on Render.com")

try:
    # Import the FastAPI app from main.py
    logger.info("Importing app from main.py")
    from main import app
    logger.info("Successfully imported app from main.py")
except Exception as e:
    logger.error(f"Error importing app from main.py: {str(e)}", exc_info=True)
    # Provide a simple app for error reporting if the main app fails to load
    from fastapi import FastAPI
    app = FastAPI()
    
    @app.get("/")
    def error_root():
        return {
            "error": "Application failed to start properly",
            "details": str(e),
            "status": "error"
        }

# This file is used by Render.com for deployment
# It imports the FastAPI app from main.py 
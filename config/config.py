import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file in development
load_dotenv()

# Log environment and variables for debugging (only key names, not values)
if os.environ.get("RENDER"):
    logger.info("Running on Render.com")
    logger.info(f"Environment variables available: {', '.join(k for k in os.environ.keys() if not k.startswith('_'))}")

SEED = 42

# Critical environment variables with defaults for development only
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logger.warning("DATABASE_URL not found in environment, using default for development")
    DATABASE_URL = "postgresql://user:password@localhost/dbname"
    
# Log database connection info (without credentials)
if DATABASE_URL:
    # Extract and log just the host part for debugging
    db_parts = DATABASE_URL.split("@")
    if len(db_parts) > 1:
        db_host = db_parts[1].split("/")[0]
        logger.info(f"Database host: {db_host}")
    else:
        logger.info("Using local database")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_jwt_secret_here")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "30"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "openai_index")
NEO4J_HOST = os.getenv("NEO4J_HOST", "bolt://localhost:7687")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# OpenAI model configuration
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_CHAT_MODEL_FULL = "gpt-4o"  # For more complex tasks
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_VISION_MODEL = "gpt-4o"  # Model with vision capabilities

props_schema = """
    `page_label` STRING,
    `file_name` STRING,
    `file_path` STRING,
    `file_type` STRING,
    `file_size` INT,
    `creation_date` STRING,
    `last_modified_date` STRING,
    `_node_content` STRING,
    `_node_type` STRING,
    `document_id` STRING,
    `doc_id` STRING,
    `ref_doc_id` STRING,
    `triplet_source_id` STRING
"""

VECTOR_DB_DIMENSION=1536
import asyncio
import nest_asyncio
nest_asyncio.apply()

import httpx
from typing import List, AsyncGenerator, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import logging

from config.config import SEED, OPENAI_API_KEY
from databases.database import get_db
from auth import get_current_user, UserOut
from routes.websockets import websocket_auth_dialogue, websocket_chat
from routes.repositories import router as repositories_router
from routes.files import router as files_router
from openai import OpenAI

# Configure logging
# Use the default event loop policy for Windows compatibility
# This replaces the previous uvloop.install() call

# logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

# Configuration
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o"]
CHECK_INTERVAL = 240  # 4 minutes
TIMEOUT = 200  # 200 seconds timeout for requests

def check_openai_model(client: OpenAI, model: str) -> Dict[str, Any]:
    """
    Perform a health check on a specific OpenAI model by attempting to use it.
    
    Args:
        client: OpenAI client
        model: Name of the model to check
    
    Returns:
        Dictionary with health check results
    """
    try:
        # Use the client directly - no need for await since OpenAI client is synchronous
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=5,
            timeout=TIMEOUT
        )
        
        # If we get this far, the model is working
        return {
            "model": model,
            "status": "healthy",
            "created_at": response.created,
            "done": True
        }
    
    except Exception as e:
        logger.error(f"Error checking model {model}: {e}")
        return {
            "model": model,
            "status": "unhealthy",
            "error": str(e)
        }

async def continuous_model_health_checks():
    """
    Continuously perform health checks on configured OpenAI models.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    while True:
        try:
            # Perform health checks for all models
            health_checks = []
            for model in OPENAI_MODELS:
                # Run the synchronous function in a thread pool and properly await it
                check_func = lambda: check_openai_model(client, model)
                result = await asyncio.get_event_loop().run_in_executor(None, check_func)
                health_checks.append(result)
            
            # Log health check results
            for check in health_checks:
                if check['status'] == 'healthy':
                    logger.info(f"Model {check['model']} is healthy")
                else:
                    logger.warning(f"Model health check failed: {check}")
        
        except Exception as e:
            logger.error(f"Unexpected error in health checks: {e}")
        
        # Wait before next round of checks
        await asyncio.sleep(CHECK_INTERVAL)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[dict, None]:
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup tasks
    health_check_task = asyncio.create_task(continuous_model_health_checks())
    
    try:
        yield {}
    finally:
        # Cleanup tasks on shutdown
        health_check_task.cancel()
        try:
            await health_check_task
        except asyncio.CancelledError:
            logger.info("Health check task cancelled successfully")

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(repositories_router)
app.include_router(files_router)

@app.get("/")
def read_root():
    return {"message": "Hello, World! This is your FastAPI backend with phone-based authentication."}

@app.get("/get_user", response_model=UserOut)
async def get_user_endpoint(token: str = Query(..., description="Authentication token")):
    """
    Retrieve user details based on the provided token.
    """
    logger.debug(f"Received token: {token}")
    try:
        user = await get_current_user(token)
        logger.debug(f"User retrieved: {user}")
        return UserOut(id=user.id, phone_number=user.phone_number)
    except Exception as e:
        logger.error(f"Error in get_user_endpoint: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid authentication credentials: {str(e)}")

@app.websocket("/ws/auth-dialogue")
async def websocket_auth_endpoint(websocket: WebSocket):
    await websocket_auth_dialogue(websocket)

@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if token is None:
        await websocket.accept()
        await websocket.send_json({"error": "No token provided"})
        await websocket.close(code=1008)
        return
    await websocket_chat(websocket, token)

if __name__ == "__main__":
    import uvicorn
    # Use standard asyncio loop instead of uvloop for Windows compatibility
    uvicorn.run(app, host="0.0.0.0", port=8000)
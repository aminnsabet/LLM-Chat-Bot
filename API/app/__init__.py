from fastapi import FastAPI
from .routers import db_functions, login_router, inference , vllm_build, OpenAI_API
from app.logging_config import setup_logger

# Setup logger
logger = setup_logger()

# Initialize FastAPI app
app = FastAPI()

# Include routers
app.include_router(login_router.router)
app.include_router(db_functions.router, prefix="/db_request", tags=["db_functions"])
app.include_router(vllm_build.router, prefix="/vllm_request", tags=["vllm_build"])
app.include_router(inference.router, prefix="/llm_request", tags=["inference"])
app.include_router(OpenAI_API.router, prefix="/chat", tags=["OPENAI_API"])

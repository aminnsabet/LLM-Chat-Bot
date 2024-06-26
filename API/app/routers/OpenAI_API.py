from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import time
from fastapi import APIRouter, Depends, HTTPException
from app.APIs import  ChatCompletionRequest
from app.depencencies.security import get_current_active_user
from app.database import User
import requests
import yaml
import pathlib
import yaml
import logging
from app.routers.utils.backend_vLLM import vLLM_Inference
from app.logging_config import setup_logger
current_path = pathlib.Path(__file__)

config_path = current_path.parent.parent.parent / 'cluster_conf.yaml'
# Environment and DB setup
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

logger = setup_logger()
def get_route_prefix_for_llm(llm_name):
    for llm in config['LLMs']:
        if llm['name'] == llm_name:
            return llm['route_prefix']
    return None


router = APIRouter()


@router.post("/completions")
async def chat_completions(request: ChatCompletionRequest,current_user: User = Depends(get_current_active_user)):
    if request.messages and request.messages[0].role == 'user':
        resp_content = "As a mock AI Assistant, I can only echo your last message: " + request.messages[-1].content
    else:
        resp_content = "As a mock AI Assistant, I can only echo your last message, but there were no messages!"
    
    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{
            "message": ChatMessage(role="assistant", content=resp_content)
        }]
    }

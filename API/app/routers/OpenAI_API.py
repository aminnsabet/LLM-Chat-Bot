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
from openai import OpenAI
from app.APIs import ChatMessage
from app.routers.utils.backend_database import Database
from starlette.responses import StreamingResponse
import json
import asyncio

database = Database()

router = APIRouter()
def engine_name_exists(engines_list, engine_name_to_check):
    for engine in engines_list['engines']:
        if engine['model_name'] == engine_name_to_check:
            return engine["inference_end_point"]
    return False

async def _resp_async_generator(client:OpenAI ,messages: List[ChatMessage], model: str, max_tokens: int, temperature: float):
   
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": m.role, "content": m.content} for m in messages],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True
    )

    for chunk in response:
        chunk_data = chunk.to_dict()
        yield f"data: {json.dumps(chunk_data)}\n\n"
        await asyncio.sleep(0.01)
    yield "data: [DONE]\n\n"

@router.post("/completions")
async def chat_completions(request: ChatCompletionRequest,current_user: User = Depends(get_current_active_user)):
    available_models = database.get_user_engines({"username": current_user.username})
    model_endpoint = engine_name_exists(available_models,request.model)
    if not model_endpoint:
        raise HTTPException(status_code=404, detail="Model not found")
    else: 
        endpoint_url = model_endpoint.replace("/completions", "")
        client = OpenAI(base_url=endpoint_url, api_key="fake-key")
        if request.messages:
            if request.stream:
                return StreamingResponse(
                    _resp_async_generator(
                        client=client,
                        messages=request.messages,
                        model=request.model,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature
                    ), media_type="application/x-ndjson"
                )
            else:
                response = client.chat.completions.create(
                    model=request.model,
                    messages=[{"role": m.role, "content": m.content} for m in request.messages],
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
                return response
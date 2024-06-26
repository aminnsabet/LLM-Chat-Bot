from fastapi import APIRouter, Depends, HTTPException
from app.APIs import  InferenceRequest
from app.depencencies.security import get_current_active_user
from app.database import User
import requests
import logging
from app.routers.utils.backend_vLLM import vLLM_Inference
from app.logging_config import setup_logger


llm = vLLM_Inference()
router = APIRouter()


@router.post("/")
async def create_inference(data: InferenceRequest, current_user: User = Depends(get_current_active_user)):
    logger.info("Received request by router: %s", data.dict())
    try:
        data.username = current_user.username
        response = await llm.InferenceCall(data)
        return {"username": current_user.username, "data":response}

    except requests.HTTPError as e:
        if response.status_code == 400:
            raise HTTPException(status_code=400, detail="Bad request to the other API service.")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to forward request to the other API service. Error: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

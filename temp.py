import os
import socket
import logging
import requests
import json
import ray
from fastapi import FastAPI, HTTPException, APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import Request
from pydantic import BaseModel
from ray import serve
from ray.serve.schema import ServeStatus, ApplicationStatusOverview, ApplicationStatus, DeploymentStatus
from typing import Optional, Dict, List
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from app.APIs import deploymentsRequest, PatchRequest
from app.routers.utils.database_utils import Database
from app.logging_config import setup_logger
import secrets
import string
import uuid
from app.database import get_db
from sqlalchemy.orm import Session

# Custom logging configuration
CUSTOM_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
            "datefmt": "%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": logging.DEBUG,
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": logging.DEBUG,
    },
    "loggers": {
        "vllm": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(CUSTOM_LOGGING_CONFIG)
logger = setup_logger()
router = APIRouter()

def generate_random_name(length=16):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for i in range(length))

def find_free_port(start=8500, end=8700):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    raise Exception("No free port found in the specified range.")

def validate_args(args: dict):
    errors = []
    
    # Validate autoscaling config
    autoscaling_config = args.get("autoscaling_config", {})
    for key, value in autoscaling_config.items():
        if value < 0:
            errors.append(f"Autoscaling config '{key}' must be non-negative.")
    
    # Validate HF Access Token
    hf_access_token = args.get("hf_access_token")
    if hf_access_token:
        headers = {"Authorization": f"Bearer {hf_access_token}"}
        response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
        if response.status_code != 200:
            errors.append("Invalid Hugging Face access token.")
    
    # Validate model source
    model_source = args.get("model_source")
    if model_source not in ["hugging_face", "nsclae"]:
        errors.append("Model source must be either 'hugging_face' or 'nsclae'.")
    
    # Validate seed
    seed = args.get("vllm_deploy_config", {}).get("seed")
    if seed is not None and seed < 0:
        errors.append("Seed must be non-negative.")
    
    return errors

def validate_huggingface_model(value: str):
    response = requests.head(f"https://huggingface.co/{value}")
    if response.status_code != 200:
        return False, "Invalid Hugging Face model name."
    logger.info(f"Model {value} exists on Hugging Face.")
    return True, None

def parse_vllm_args(cli_args: Dict[str, str]):
    parser = make_arg_parser()
    free_port = find_free_port()
    final_args = []
    for key, value in cli_args.items():
        if value is None or value == "string":
            continue
        arg_key = f"--{key.replace('_', '-')}"
        final_args.extend([arg_key, str(value)])
    final_args.extend(["--port", str(free_port)])
    logger.info(f"Final args for deployment: {final_args}")
    parsed_args = parser.parse_args(args=final_args)
    return parsed_args


@serve.deployment()
@serve.ingress(router)
class VLLMDeployment:
    def __init__(self, engine_args: AsyncEngineArgs, response_role: str, lora_modules: Optional[List[LoRAModulePath]] = None, chat_template: Optional[str] = None, hf_access_token_ref = None):
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.chat_template = chat_template
        if hf_access_token_ref:
            hf_access_token = ray.get(hf_access_token_ref)  
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_access_token
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.health = self.engine.check_health()

    @router.post("/v1/chat/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            served_model_names = self.engine_args.served_model_name or [self.engine_args.model]
            self.openai_serving_chat = OpenAIServingChat(
                self.engine, model_config, served_model_names, self.response_role, self.lora_modules, self.chat_template
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


    

def make_vllm_deployment(request: deploymentsRequest):
    try:
        validation_errors = validate_args(request.dict())
        if validation_errors:
            logger.error(f"Validation errors: {validation_errors}")
            return None, None, None, None, {"status": "error", "message": validation_errors}
        
        parsed_args = parse_vllm_args(request.vllm_deploy_config.dict())

        hf_access_token = request.hf_access_token
        hf_access_token_ref = ray.put(hf_access_token)  # Use ray.put to handle the token securely
        logger.info('HUGGING_FACE_HUB_TOKEN is set.')

        valid, error_message = validate_huggingface_model(request.vllm_deploy_config.model)
        if not valid:
            logger.error(error_message)
            return None, None, None, None, {"status": "error", "message": error_message}
        
        engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
        engine_args.worker_use_ray = True

        tp = engine_args.tensor_parallel_size
        pg_resources = [{"CPU": 1}]
        for i in range(tp):
            pg_resources.append({"CPU": 1, "GPU": 1})

        deployment_name = uuid.uuid4().hex
        random_prefix = "/" + deployment_name

        try:
            ray.init(namespace=request.project_id, ignore_reinit_error=True)
        except Exception:
            ray.shutdown()
            ray.init(namespace=request.project_id, ignore_reinit_error=True)

        deployment = VLLMDeployment.options(
            placement_group_bundles=pg_resources, 
            placement_group_strategy="STRICT_PACK", 
            autoscaling_config=request.autoscaling_config.dict(), 
            health_check_period_s=10, 
            health_check_timeout_s=10
        ).bind(
            engine_args, parsed_args.response_role, parsed_args.lora_modules, parsed_args.chat_template, hf_access_token_ref  # Pass the token reference
        )
        
        return deployment, deployment_name, parsed_args.port, random_prefix, {"status": "success", "message": "Deployment created successfully.", "parsed_args": vars(parsed_args)}

    except Exception as e:
        logger.error(f"Error creating deployment: {e}")
        return None, None, None, None, {"status": "error", "message": f"Error creating deployment: {e}"}





@router.post("/build")
def build_app(request: deploymentsRequest, db: Session = Depends(get_db)) -> JSONResponse:
    try:
        deployment, deployment_name, port, random_prefix, status = make_vllm_deployment(request)
        if status["status"] == "error":
            return JSONResponse(content={"status": "error", "message": status["message"]}, status_code=500)
        database = Database()
        if not deployment:
            raise Exception("Deployment creation failed.")

        deployment_name = "deploy_" + deployment_name
        model_args = status.get("parsed_args", {})
        payload = {
            "user_id": request.user_id,
            "autoscaling_config": json.dumps(request.autoscaling_config.dict()),
            "vllm_deploy_config": json.dumps(model_args),
            "model_source": request.model_source,
            "endpoint": "",
            "deployment_name": deployment_name,
            "status": "DEPLOYING",
        }

        db_resp = database.add_deployment_model_to_user(payload, db)
        
        if db_resp['message'] != "success":
            logger.error(f"Database error: {db_resp['error']}")
            return JSONResponse(content={"status": "error", "message": "Deployment failed."}, status_code=500)

        serve.run(deployment, name=deployment_name, route_prefix=random_prefix)

        if serve.status().applications[deployment_name].status == "RUNNING":
            db_resp = database.update_deployment_model({"user_id": request.user_id, "deployment_name": deployment_name, "status": "RUNNING", "endpoint": ""}, db)
            if db_resp['message'] == "success":
                logger.info(f"Application built successfully and running on port {port} with deployment of {deployment_name}")
                return JSONResponse(content={"status": "success", "message": f"Application built successfully and running on port {port} with deployment of {deployment_name}"}, status_code=200)
            else:
                ray.serve.delete(name=deployment_name, _blocking=True)
                logger.error(f"Database error and deployment deleted: {db_resp['error']}")
                return JSONResponse(content={"status": "error", "message": f"Database error: {db_resp['error']}"}, status_code=500)
        else:
            raise Exception("Deployment failed.")
    except Exception as e:
        if serve.status().applications.get(deployment_name) and serve.status().applications[deployment_name].status == "DEPLOY_FAILED":
            error_message = serve.status().applications[deployment_name].deployments["VLLMDeployment"].message
            logger.error(f"Deployment failed error message saved in database")
            database.add_deployment_error({"user_id": request.user_id, "deployment_name": deployment_name, "error_message": error_message, "error_code": 500}, db)
            ray.serve.delete(name=deployment_name, _blocking=True)
            database.update_deployment_model({"user_id": request.user_id, "deployment_name": deployment_name, "status": "DEAD", "endpoint": ""}, db)
            logger.error(f"Deployment deleted")
        return JSONResponse(content={"status": "error", "message": f"Error building the application: {e}"}, status_code=500)


@router.post("/delete")
def delete_app(request: PatchRequest, db: Session = Depends(get_db)) -> JSONResponse:
    try:
        database = Database()
        try:
            ray.init(namespace=request.project_id, ignore_reinit_error=True)
        except Exception:
            ray.shutdown()
            ray.init(namespace=request.project_id, ignore_reinit_error=True)
        ray.serve.delete(name=request.deployment_name, _blocking=True)
        database.update_deployment_model({"user_id": request.user_id, "deployment_name": request.deployment_name, "status": "DEAD", "endpoint": ""},  db)
        logger.info(f"Deleted the application: {request.deployment_name}")
        return JSONResponse(content={"status": "success", "message": "Application deleted successfully."}, status_code=200)
    except Exception as e:
        logger.error(f"Error deleting the application: {e}")
        return JSONResponse(content={"status": "error", "message": f"Error deleting the application: {e}"}, status_code=500)

@router.post("/status")
def get_status(request: PatchRequest) -> JSONResponse:
    try:
        try:
            ray.init(namespace="vllm", ignore_reinit_error=True)
        except Exception:
            ray.shutdown()
            ray.init(namespace="vllm", ignore_reinit_error=True)
        app_status = serve.status().applications.items()
        return JSONResponse(content={"status": "success", "applications": list(app_status)}, status_code=200)
    except Exception as e:
        logger.error(f"Error getting the application status: {e}")
        return JSONResponse(content={"status": "error", "message": f"Error getting the application status: {e}"}, status_code=500)


import os
import threading
import socket
import logging
from fastapi import FastAPI, HTTPException,APIRouter,Depends
from app.depencencies.security import get_current_active_user
from app.database import User
from pydantic import BaseModel, field_validator
import docker
import uvicorn
import traceback
import requests
from typing import Optional
import time
from app.logging_config import setup_logger
from app.models import VllmRequest
from app.routers.LLM.backend_database import Database
logger = setup_logger()


router = APIRouter()

class VllmRequest(VllmRequest):
    
    @field_validator('HUGGING_FACE_HUB_TOKEN')
    def validate_huggingface_token(cls, value):
        headers = {"Authorization": f"Bearer {value}"}
        response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
        if response.status_code != 200:
            raise ValueError("Invalid Hugging Face token.")
        return value

    @field_validator('MODEL')
    def validate_huggingface_model(cls, value):
        response = requests.head(f"https://huggingface.co/{value}")
        if response.status_code != 200:
            raise ValueError("Invalid Hugging Face model.")
        return value

    @field_validator('TOKENIZER')
    def validate_huggingface_tokenizer(cls, value):
        if value != 'auto':
            response = requests.head(f"https://huggingface.co/{value}")
            if response.status_code != 200:
                raise ValueError("Invalid Hugging Face tokenizer.")
        return value

    @field_validator('MAX_MODEL_LEN')
    def validate_max_model_len(cls, value):
        if value <= 0:
            raise ValueError("MAX_MODEL_LEN must be a positive integer.")
        return value

    @field_validator('TENSOR_PARALLEL_SIZE')
    def validate_tensor_parallel_size(cls, value):
        if value <= 0:
            raise ValueError("TENSOR_PARALLEL_SIZE must be a positive integer.")
        return value

    @field_validator('SEED')
    def validate_seed(cls, value):
        if value < 0:
            raise ValueError("SEED must be a non-negative integer.")
        return value

    @field_validator('QUANTIZATION')
    def validate_quantization(cls, value):
        valid_options = [
            "aqlm", "awq", "deepspeedfp", "fp8", "marlin",
            "gptq_marlin_24", "gptq_marlin", "gptq", "squeezellm",
            "compressed-tensors", "bitsandbytes","None"
        ]
        if value not in valid_options:
            raise ValueError(f"QUANTIZATION must be one of {valid_options}.")
        return value

def capture_logs(container, log_file_path):
    with open(log_file_path, 'a') as log_file:  # Open in append mode
        for line in container.logs(stream=True, stdout=True, stderr=True):
            decoded_line = line.strip().decode('utf-8')
            log_file.write(decoded_line + '\n')
            log_file.flush()  # Ensure each log line is written immediately

def find_free_port(start=8500, end=8700):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    raise Exception("No free port found in the specified range.")

def validate_huggingface_token(token: str):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
    if response.status_code == 200:
        return response.json().get("name")
    return None

def validate_huggingface_model(model: str):
    response = requests.head(f"https://huggingface.co/{model}")
    return response.status_code == 200

def check_vllm_health(endpoint: str, timeout: int = 60, interval: int = 5):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(endpoint)
            if response.status_code == 200:
                return True
        except requests.RequestException as e:
            logger.error(f"Health check error: {str(e)}")
        time.sleep(interval)
    return False

def update_user_vLLM(username,container_id,model_name,endpoint,quantized,max_model_length,seed):
    database = Database()
    response= database.add_engine_to_user({"username": username, "container_id": container_id, "model_name": model_name,"endpoint":endpoint, "quantized": quantized, "max_model_length": max_model_length, "seed": seed})
    return response

def delete_user_vLLM(username,container_id,engine_name=None):
    database = Database()
    response= database.remove_engine_from_user({"username": username, "container_id": container_id, "engine_name": engine_name})
    return response

def user_engine_info(username):
    database = Database()
    response= database.get_user_engines({"username": username})
    return response

@router.post("/start/")
def run_docker(engine_args: VllmRequest,user: User = Depends(get_current_active_user)):
    user_info = validate_huggingface_token(engine_args.HUGGING_FACE_HUB_TOKEN)
    if user_info is None:
        raise HTTPException(status_code=400, detail="Invalid Hugging Face token.")
    
    if not validate_huggingface_model(engine_args.MODEL):
        raise HTTPException(status_code=400, detail="Invalid Hugging Face model.")

    client = docker.from_env()

    home_directory = os.path.expanduser("~")
    volume_path = os.path.join(home_directory, ".cache/huggingface")
    volumes = {volume_path: {'bind': '/root/.cache/huggingface', 'mode': 'rw'}}
    env_vars = {'HUGGING_FACE_HUB_TOKEN': engine_args.HUGGING_FACE_HUB_TOKEN}

    log_file_path = "/home/amin_sabet/dev/LLM-Chat-Bot/vLLM/file.log"

    # Ensure the directory exists
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    try:
        
        free_port = find_free_port()
        ports = {8000:f'{free_port}/tcp'}  # Map the external port to the internal port 8000

        command = f"--model {engine_args.MODEL} --max-model-len {engine_args.MAX_MODEL_LEN}"
        if engine_args.QUANTIZATION:
            command += f" --quantization {engine_args.QUANTIZATION}"
        if engine_args.TOKENIZER and engine_args.TOKENIZER != 'auto':
            command += f" --tokenizer {engine_args.TOKENIZER}"

        container = client.containers.run(
            'vllm/vllm-openai:latest',
            command=command,
            runtime='nvidia',
            volumes=volumes,
            environment=env_vars,
            ports=ports,
            ipc_mode='host',
            detach=True
        )
        
        # Start a thread to capture logs asynchronously
        log_thread = threading.Thread(target=capture_logs, args=(container, log_file_path))
        log_thread.start()

        # Check container status
        container.reload()
        if container.status != "running":
            raise Exception("Container failed to start. Check the logs for more details.")

        health_check_endpoint = f"http://localhost:{free_port}/metrics"
        if check_vllm_health(health_check_endpoint):
            logger.info(f"Container started successfully. Port: {free_port}")
            logger.info(f"User info: {user_info}")
            logger.info(f"Container ID: {container.id}")
            endpoint = f"http://localhost:{free_port}/v1/completions"
            try:
                db_resp = update_user_vLLM( user.username,container.id,engine_args.MODEL,endpoint,engine_args.QUANTIZATION,engine_args.MAX_MODEL_LEN,engine_args.SEED) 
            except Exception as e:
                return {"message": "Container started successfully but failed to update the database.",
                       "container_id": container.id }
            return {
                "message": "Container started successfully and vLLM server is healthy",
                "vLLM_endpoint": endpoint,
                "user_info": user_info,
                "status": "healthy"

            }
        else:
            raise Exception("Container did not pass health check.")

    except Exception as e:
        # Log error message to the log file
        with open(log_file_path, 'a') as log_file:  # Open in append mode
            log_file.write(f"Error: {str(e)}\n")
            log_file.write(traceback.format_exc())
            log_file.flush()

        raise HTTPException(status_code=500, detail=str(e))

@router.post("/terminate/")
def stop_docker(engine_args: VllmRequest,user: User = Depends(get_current_active_user)):
    client = docker.from_env()
    container_id = engine_args.container_id
    container = client.containers.get(container_id)
    container.stop()
    container.remove()
    db_resp = delete_user_vLLM(user.username,container_id,engine_args.engine_name)
    return {"message": "Container stopped successfully."}

@router.post("/active_models/")
def available_engine(user: User = Depends(get_current_active_user)):
    response = user_engine_info(user.username)
    return response

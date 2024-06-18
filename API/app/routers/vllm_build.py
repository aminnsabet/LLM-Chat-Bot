import os
import threading
import socket
import logging
from fastapi import FastAPI, HTTPException,APIRouter
from pydantic import BaseModel, field_validator
import docker
import uvicorn
import traceback
import requests
from typing import Optional
import time
from app.logging_config import setup_logger
from app.models import VllmBuildRequest
logger = setup_logger()


router = APIRouter()

class VllmBuildRequest(VllmBuildRequest):
    
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


@router.post("/")
def run_docker(engine_args: VllmBuildRequest):
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
            return {
                "message": "Container started successfully and vLLM server is healthy",
                "vLLM_endpoint": f"http://localhost:{free_port}/v1/completions",
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
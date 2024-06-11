import os
import threading
import socket
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import docker
import uvicorn
import traceback
import requests

app = FastAPI()

class EngineArgs(BaseModel):
    HUGGING_FACE_HUB_TOKEN: str
    model: str

def capture_logs(container, log_file_path):
    with open(log_file_path, 'w') as log_file:
        for line in container.logs(stream=True):
            decoded_line = line.strip().decode('utf-8')
            log_file.write(decoded_line + '\n')

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

@app.post("/run-docker")
def run_docker(engine_args: EngineArgs):
    user_info = validate_huggingface_token(engine_args.HUGGING_FACE_HUB_TOKEN)
    if user_info is None:
        raise HTTPException(status_code=400, detail="Invalid Hugging Face token.")
    
    if not validate_huggingface_model(engine_args.model):
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
        ports = {f'{free_port}/tcp': free_port}

        container = client.containers.run(
            'vllm/vllm-openai:latest',
            command=f"--model {engine_args.model} --max-model-len 512",
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

        logging.info(f"Container started successfully. Port: {free_port}")
        logging.info(f"User info: {user_info}")
        logging.info(f"container ID: {container.id}")
        
        return {
            "message": "Container started successfully",
            "vLLM_endpoint": f"http://localhost:{free_port}/v1/",
            "user_info": user_info
        }
    except Exception as e:
        # Log error message to the log file
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Error: {str(e)}\n")
            log_file.write(traceback.format_exc())

        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

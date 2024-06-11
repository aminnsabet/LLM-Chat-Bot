import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
token = "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"
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m",token=token

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=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" \
    -p 8000:8500 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model "bigscience/bloom-560m"
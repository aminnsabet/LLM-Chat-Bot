# import requests
# from transformers import AutoTokenizer, AutoModelForCausalLM
# token = "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"
# model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m",token=token

# docker run --runtime nvidia --gpus all     -v ~/.cache/huggingface:/root/.cache/huggingface     --env "HUGGING_FACE_HUB_TOKEN=<secret>"     -p 8500:8000    
#  --ipc=host     vllm/vllm-openai:latest     --model Hastagaras/Halu-8B-Llama3-v0.3
 #docker run --runtime nvidia --gpus all     -v ~/.cache/huggingface:/root/.cache/huggingface     --env "HUGGING_FACE_HUB_TOKEN=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"     -p 8000:8000     --ipc=host     vllm/vllm-openai:latest     --model meta-llama/Llama-2-7b-chat-hf
import os
from sqlalchemy import create_engine
from langchain_community.llms import VLLMOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define the LLM
model = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8500/v1",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    model_kwargs={"stop": ["."]},
)



# "http://localhost:8500/v1"

# curl http://localhost:8500/v1/completions \
# -H "Content-Type: application/json" \
# -d '{
# "model": "meta-llama/Llama-2-7b-chat-hf",
# "prompt": "in 300 words explain the science of nuclear fusion",
# "max_tokens": 7,
# "temperature": 0
# }'

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=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" \
    -p 8500:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-2-7b-chat-hf

TheBloke/Llama-2-7B-Chat-AWQ
TechxGenus/Meta-Llama-3-8B-Instruct-AWQ
Moses25/Llama-3-8B-chat-32K-AWQ
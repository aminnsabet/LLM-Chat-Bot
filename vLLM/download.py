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

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who is answering user questions in a chat manner.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
import logging
logging.getLogger().setLevel(logging.ERROR)

runnable: Runnable = prompt | model

# Ensure the database file exists
from langchain_community.chat_message_histories import SQLChatMessageHistory


def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print(with_message_history.invoke(
    {"input": "my Name is Amin?"},
    config={"configurable": {"session_id": "abc126"}},
))
print(with_message_history.invoke(
    {"input": "What was my name?"},
    config={"configurable": {"session_id": "abc126"}},
))
messages = SQLChatMessageHistory("abc126","sqlite:///memory.db").get_messages()
for message in messages:
    print(message)
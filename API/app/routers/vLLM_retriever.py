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
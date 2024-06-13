from langchain.chains.conversation.memory import ConversationBufferMemory

import json
import re
from typing import Optional
from pydantic import BaseModel
import textwrap
from langchain.chains import RetrievalQA
import logging
from langchain_community.vectorstores import Weaviate
import weaviate
import wandb
from app.routers.LLM.backend_database import Database
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain import LLMChain
from typing import List
import json
import yaml
import time
import asyncio
from langchain_community.llms import HuggingFaceTextGenInference
from langchain.chains import LLMChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import pathlib
import os
import transformers
from torch import cuda, bfloat16
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain_community.vectorstores import Weaviate
from app.logging_config import setup_logger
from langchain import hub
from langchain_community.llms import VLLMOpenAI
import os
from sqlalchemy import create_engine
from langchain_community.llms import VLLMOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
# ------------------- Configuration --------------------
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)



class Input(BaseModel):
    username: Optional[str]
    prompt: Optional[str]
    memory: Optional[bool]
    conversation_number: Optional[int]
    AI_assistance: Optional[bool]
    collection_name: Optional[str]
    llm_model: Optional[str]


# ------------------------------ LLM Deployment -------------------------------
class LLMDeployment:
    def __init__(self, model_tokenizer,
                 temperature =  0.01,
                 max_new_tokens=  512,
                 repetition_penalty= 1.1,
                 batch_size= 2):
        

        self.logger = setup_logger()

        current_path = pathlib.Path(__file__).parent.parent.parent.parent
        config_path = current_path/ 'cluster_conf.yaml'
        self.RAG_enabled = True
        # Environment variables setup
        with open(config_path, 'r') as self.file:
            self.config = yaml.safe_load(self.file)
            self.config = Config(**self.config)
        # Initialize Weaviate client for RAG


        # try:
        #     self.weaviate_client = weaviate.Client(
        #         url=self.config.weaviate_client_url,   
        #     )
        # except:
        #     self.logger.error("Error in connecting to Weaviate")
        #     self.RAG_enabled = False

        # #setting up weight and bias logging
        # self.wandb_logging_enabled = self.config.WANDB_ENABLE
        # if self.wandb_logging_enabled:
        #     try:
        #         wandb.login(key = self.config.WANDB_KEY)
        #         wandb.init(project="Service Metrics", notes="custom step")
        #         wandb.define_metric("The number of input tokens")
        #         wandb.define_metric("The number of generated tokens")
        #         wandb.define_metric("Inference Time")
        #         wandb.define_metric("token/second")
        #         self.logger.info("Wandb Logging Enabled")
        #     except:
        #         self.wandb_logging_enabled = False
        #         self.logger.info("Wandb Logging Not Enabled")
        #         pass
        

        # Initialize deployment class
        self.model_tokenizer = model_tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.batch_size = batch_size
        self.loop = asyncio.get_running_loop()
        self.access_token = self.config.Hugging_ACCESS_TOKEN


        # initialize Prompt 
        
        #prompt_hub = hub.pull("llmchatbot/default",api_key = self.config.Langchain_access_key, api_url="https://api.hub.langchain.com")
        

        
        

        # Initialize Memory

        self.database = Database()

    
    
    

    def cut_off_text(self, text, prompt):
        cutoff_phrase = prompt
        index = text.find(cutoff_phrase)
        if index != -1:
            return text[:index]
        else:
            return text

    def remove_substring(self, string, substring):
        return string.replace(substring, "")


    def cleaning_memory(self):
        print(self.memory.chat_memory.messages)
        self.memory.clear()
        print("Chat History Deleted")

    def parse_text(self, text):
        pattern = r"\s*Assistant:\s*"
        pattern2 = r"\s*AI:\s*"
        cleaned_text = re.sub(pattern, "", text)
        cleaned_text = re.sub(pattern2, "", cleaned_text)
        wrapped_text = textwrap.fill(cleaned_text, width=100)
        return wrapped_text
    def get_session_history(self, session_id,db_path="sqlite:///memory.db"):
                    return SQLChatMessageHistory(session_id,db_path )

    def InferenceCall(self, request: Input):
      
            self.logger.info("Received request by backend: %s", request.dict())
            input_prompt = request.prompt
            AI_assistance = request.AI_assistance
            username = request.username
            memory = request.memory
            conversation_number = request.conversation_number
            collection_name = request.collection_name

            llm = VLLMOpenAI(
                openai_api_key="EMPTY",
                openai_api_base="http://localhost:8000/v1",
                model_name="meta-llama/Llama-2-7b-chat-hf",
                model_kwargs={"stop": ["."]},
            )

            if memory:
                
                # # Retrieve conversation ID from Database
                # if conversation_number <= 0 or conversation_number is None:

                #     conversation_id = self.database.retrieve_conversation(
                #         {
                #             "username": username,
                #         }
                #     )
                # else:
                #     conversation_id = self.database.retrieve_conversation(
                #         {
                #             "username": username,
                #             "conversation_number": conversation_number,
                #         }
                #     )
                import logging
                logging.getLogger().setLevel(logging.ERROR)
                
                self.prompt =  ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an chatbot assistant who is answering user's questions. Do add ",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
                runnable: Runnable = self.prompt | llm
                with_message_history = RunnableWithMessageHistory(
                    runnable,
                    self.get_session_history,
                    input_messages_key="input",
                    history_messages_key="history",
                )
                reponse = with_message_history.invoke(
                    {"input": "My name is Amin."},
                    config={"configurable": {"session_id": "123"}},
                )
    
                return reponse
            
                
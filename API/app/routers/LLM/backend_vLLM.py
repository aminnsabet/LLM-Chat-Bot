import json
import re
from typing import Optional
from pydantic import BaseModel
import textwrap
import logging
import yaml
import pathlib
import asyncio
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain_community.chat_message_histories import SQLChatMessageHistory
from app.logging_config import setup_logger
from app.routers.LLM.backend_database import Database
from langchain_community.llms import VLLMOpenAI
import tiktoken
from fastapi import FastAPI

app = FastAPI()

# Configuration class to load YAML configurations
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

# Input model defining the expected structure of the API request
class Input(BaseModel):
    username: Optional[str]
    prompt: Optional[str]
    memory: Optional[bool]
    conversation_number: Optional[int]
    AI_assistance: Optional[bool]
    collection_name: Optional[str]
    llm_model: Optional[str]

# LLM Deployment class encapsulates the model interaction logic
class LLMDeployment:
    def __init__(self, model_tokenizer, temperature=0.01, max_new_tokens=512, repetition_penalty=1.1, batch_size=2):
        self.logger = setup_logger()
        
        # Load configurations
        current_path = pathlib.Path(__file__).parent.parent.parent.parent
        config_path = current_path / 'cluster_conf.yaml'
        self.RAG_enabled = True

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            self.config = Config(**self.config)

        # Initialize tokenizer and model parameters
        self.tokenizer = tiktoken.get_encoding('cl100k_base')
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.batch_size = batch_size
        self.loop = asyncio.get_running_loop()
        self.access_token = self.config.Hugging_ACCESS_TOKEN

        # Define system prompt template
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", "You're an AI assistant who is answering user questions"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        self.database = Database()

    # Method to truncate text based on the given prompt
    def cut_off_text(self, text, prompt):
        index = text.find(prompt)
        return text[:index] if index != -1 else text

    # Method to remove a specific substring from a string
    def remove_substring(self, string, substring):
        return string.replace(substring, "")

    # Method to clear the chat memory
    def cleaning_memory(self):
        print(self.memory.chat_memory.messages)
        self.memory.clear()
        print("Chat History Deleted")

    # Method to parse and clean the response text
    def parse_text(self, text):
        cleaned_text = re.sub(r"\s*Assistant:\s*", "", text)
        cleaned_text = re.sub(r"\s*AI:\s*", "", cleaned_text)
        return textwrap.fill(cleaned_text, width=100)

    # Method to retrieve session history from the database
    def get_session_history(self, session_id, db_path="sqlite:///memory.db"):
        return SQLChatMessageHistory(session_id, db_path)

    # Method to count the number of tokens in a given text
    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    # Asynchronous method to update token usage in the database
    async def update_tokens_and_database(self, username, input_prompt, response):
        try:
            input_tokens = self.count_tokens(input_prompt)
            gen_tokens = self.count_tokens(response)
            updated_tokens = {"username": username, "prompt_token_number": input_tokens, "gen_token_number": gen_tokens}
            await self.loop.run_in_executor(None, self.database.update_token_usage, updated_tokens)
        except Exception as e:
            self.logger.error("Error in updating token usage: %s", e)

    # Asynchronous method to invoke the LLM and handle responses
    async def invoke(self, username, input_prompt, database, llm, conversation_id):
        try:
            # Invoke the LLM based on the presence of a conversation ID
            if conversation_id:
                config = {"configurable": {"session_id": conversation_id}}
                response = llm.invoke({"input": input_prompt}, config={"configurable": {"session_id": conversation_id}})
            else:
                response = llm.invoke(input_prompt)
            
            # Schedule the token update task asynchronously
            asyncio.create_task(self.update_tokens_and_database(username, input_prompt, response))
            return response
        except Exception as e:
            self.logger.error("Error in Inference Call: %s", e)
            return str(e)

    # Asynchronous method to handle the inference call
    async def InferenceCall(self, request: Input):
        try:
            self.database.add_conversation({"username": request.username})
            self.logger.info("Received request by backend: %s", request.dict())
            input_prompt = request.prompt
            AI_assistance = request.AI_assistance
            username = request.username
            memory = request.memory
            conversation_number = request.conversation_number
            collection_name = request.collection_name
            
            # Initialize the LLM with given parameters
            llm = VLLMOpenAI(
                openai_api_key="EMPTY",
                openai_api_base="http://localhost:8500/v1",
                model_name="meta-llama/Llama-2-7b-chat-hf",
                model_kwargs={"stop": ["."]},
                streaming=True,
            )
            conversation_id = None

            # Handle memory retrieval if enabled
            if memory:
                if conversation_number <= 0 or conversation_number is None:
                    conversation_id = self.database.retrieve_conversation({"username": username})["conversation_id"]
                else:
                    conversation_id = self.database.retrieve_conversation({"username": username, "conversation_number": conversation_number})["conversation_id"]

                logging.getLogger().setLevel(logging.ERROR)
                
                runnable: Runnable = self.system_prompt | llm

                llm_with_message_history = RunnableWithMessageHistory(
                    runnable,
                    self.get_session_history,
                    input_messages_key="input",
                    history_messages_key="history",
                )
                response = await self.invoke(username, input_prompt, self.database, llm_with_message_history, conversation_id)
                return response
            else:
                response = await self.invoke(username, input_prompt, self.database, llm, conversation_id)
            return response

        except Exception as e:
            self.logger.error("Error in Inference Call: %s", e)
            return str(e)

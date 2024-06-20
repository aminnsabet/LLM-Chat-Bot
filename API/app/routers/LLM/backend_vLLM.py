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
from ... models import InferenceRequest


app = FastAPI()


# LLM Deployment class encapsulates the model interaction logic
class vLLM_Inference:
    def __init__(self):
        # Initialize logger
        self.logger = setup_logger()     
        # Initialize tokenizer and model parameters
        self.tokenizer = tiktoken.get_encoding('cl100k_base')
        self.loop = asyncio.get_running_loop()
        # Define system prompt template
        self.system_prompt = ChatPromptTemplate.from_messages([
            ('''You are a helpful, respectful, and honest chatbot. Always answer as helpfully as possible while ensuring safety. Stick to relevant responses only. Use context from previous conversations to enhance your answers. If a question doesn't make sense or is factually incoherent, explain why instead of giving incorrect information. If you don't know the answer, do not share false information.

                Here is the conversation so far:
                {history}

                User's new message:
                {input}
                '''),
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
    async def invoke_no_memory(self, username, input_prompt, database, llm):
        try:
            response = llm.invoke({"input": input_prompt, "history": []})
            # Schedule the token update task asynchronously
            asyncio.create_task(self.update_tokens_and_database(username, input_prompt, response))
            return response
        except Exception as e:
            self.logger.error("Error in Inference Call: %s", e)
            return str(e)

    # Asynchronous method to handle the inference call
    async def InferenceCall(self, request: InferenceRequest):

        try:
            self.logger.info("Received request by backend: %s", request.dict())
            if not request.model or not request.inference_endpoint:
                raise Exception("Model name or inference endpoint not provided")
                return "Error: Model name not provided"

            # Initialize the LLM with given parameters
            llm = VLLMOpenAI(
                openai_api_key="EMPTY",
                openai_api_base=request.inference_endpoint,
                model_name=request.model,
                temperature=0.7,  # Adjust this value to control the randomness of responses
                max_tokens=512,  # Limit the length of the response
                top_p=0.9,
            )
            
            conversation_id = None
            runnable: Runnable = self.system_prompt | llm
            # Handle memory retrieval if enabled
            if request.memory:
                if request.conversation_number <= 0 or request.conversation_number is None:
                    conversation_id = self.database.retrieve_conversation({"username": request.username})["conversation_id"]
                else:
                    conversation_id = self.database.retrieve_conversation({"username": request.username, "conversation_number": request.conversation_number})["conversation_id"]
                logging.getLogger().setLevel(logging.ERROR)
                llm_with_message_history = RunnableWithMessageHistory(
                    runnable,
                    self.get_session_history,
                    input_messages_key="input",
                    history_messages_key="history",)
                response = await self.invoke(request.username, request.prompt, self.database, llm_with_message_history, conversation_id)
                return response
            else:
                response = await self.invoke_no_memory(request.username, request.prompt, self.database, runnable)
            return response

        except Exception as e:
            self.logger.error("Error in Inference Call: %s", e)
            return str(e)

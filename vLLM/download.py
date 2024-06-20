from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.llms import VLLMOpenAI

# Define the LLM
model = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8500/v1",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    temperature=0.7,  # Adjust this value to control the randomness of responses
    max_tokens=5,  # Limit the length of the response
    top_p=0.9,  # Use nucleus sampling to limit the tokens considered
)

# Create message history instance


system_prompt = ChatPromptTemplate.from_messages([
            ('''You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

Here is the conversation so far:
{history}

User's message:
{input}
Response:'''),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

# Combine the prompt template and the model
runnable: Runnable = system_prompt | model

# Initialize conversation history
conversation_history = []

# Function to add message to history


# Create an initial input
user_input = "What is nuclear fusion?"
input_data = {"input": user_input, "history": conversation_history}

# Invoke the runnable
response = runnable.invoke(input_data)

# Print the initial response
print(response)


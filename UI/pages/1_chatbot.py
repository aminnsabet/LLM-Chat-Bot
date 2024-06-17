import streamlit as st
import requests

from langchain.document_loaders import YoutubeLoader
import os
import streamlit as st
import requests
import json
import time
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import random
from langchain_community.chat_message_histories import SQLChatMessageHistory

BASE_URL = "http://localhost:8086"
Weaviate_endpoint = "/vector_DB_request/"

def process_text(text):
    # Remove quotes from the beginning and end of the text, if present
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Replace \n with an actual new line
    text = text.replace("\\n", "\n")

    return text

def display_user_classes(username, access_token):
    params = {
        "username": username,
        "mode": "display_classes",
        "vectorDB_type": "Weaviate",
        "mode": "display_classes",
        "class_name": "string"
        }
    file_path = None

    #headers = {"Authorization": f"Bearer {access_token}"}
    resp = send_vector_db_request(access_token, params, Weaviate_endpoint)
    #resp = requests.post(f"{BASE_URL}/vector_DB_request/",json=params, headers=headers)
        # Handle the response
    print("the response", resp, resp.content)

    response_content = resp.content.decode("utf-8")
    print("respcontent", response_content)
    user_classes = json.loads(response_content)
    if resp.status_code == 200:
        print(resp.status_code, resp.content)
        return user_classes
    else:
        print(resp.status_code, resp.content)
        return 
    
def send_vector_db_request(access_token, json_data, endpoint, uploaded_file=None):
    headers = {"Authorization": f"Bearer {access_token}"}


    response = requests.post(f"{BASE_URL}{endpoint}", data=json_data,headers=headers, files=uploaded_file)

    return response    

def authentication(username, password):
    data = {"username": username, "password": password}
    resp = requests.post(
        f"{BASE_URL}/token", data=data
    )
    if "access_token" not in resp.json():
        return None
    return resp.json()["access_token"]

def add_user(username, password,gen_token_limit,prompt_token_limit,access_token):

    headers = {"Authorization": f"Bearer {access_token}"}
    query_data = {
        "username": username,
        "password": password,
        "prompt_token_number": int(0),
        "gen_token_number": int(0),
        "gen_token_limit": int(gen_token_limit),
        "prompt_token_limit": int(prompt_token_limit)
        }
    resp = requests.post(f"{BASE_URL}/db_request/add_user/", json=query_data, headers=headers)
    return True

def get_all_users_info(access_token):   
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(f"{BASE_URL}/db_request/get_all_users/", headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        return None

def retrieve_latest_conversation(username, access_token):
    query_data = {
  "username": username
}
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.post(f"{BASE_URL}/db_request/get_user_conversations/",json=query_data, headers=headers)
    if resp.status_code == 200:
        conversations = resp.json()['conversations']
        names = [d["name"] for d in conversations]
        if names:
            return {"names": names, "conversations": conversations}
        return None
    else:
        return None
    
def add_conversation(username, conversation,access_token):
    query_data = {
  "username": username,
  "content": json.dumps(conversation),
  "conversation_name": "Current Conversation",
}
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.post(f"{BASE_URL}/db_request/add_conversation/",json=query_data, headers=headers)
    if resp.status_code == 200:
        return True
    else:
        return False

def chat(model, inference_endpoint,prompt, memory, username, conversation_number,access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    quesry_data = {
        "model": model,
        "inference_endpoint":  inference_endpoint   ,
        "prompt": prompt,
        "memory": memory,
        "conversation_number": conversation_number,
        "username": username
        }
    resp = requests.post(f"{BASE_URL}/llm_request",json=quesry_data, headers=headers)
        
    return resp.json()

def retrieving_messages(username,access_token, converation_number=None):

    headers = {"Authorization": f"Bearer {access_token}"}
    if converation_number:  
        query_data = {
        "username": username,
        "conversation_number": converation_number
        }
    else:
        query_data = {
            "username": username
            }
    resp = requests.post(f"{BASE_URL}/db_request/retrieve_conversation",json=query_data, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        return None

    #messages = SQLChatMessageHistory("abc126","/home/amin_sabet/dev/LLM-Chat-Bot/API/sqlite:///memory.db").get_messages()


def add_conversation(username, conversation_name,access_token):
    if not conversation_name:
        query_data = {
                        "username": username,
                        "conversation_name": conversation_name
                        }
    else:
        query_data = {
                        "username": username,
                        "conversation_name": conversation_name
                        }
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.post(f"{BASE_URL}/db_request/add_conversation/",json=query_data, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:   
        return None

if "username" not in st.session_state or st.sidebar.button("Logout"):
    # Login form
    if "username" not in st.session_state:
        username = st.text_input("Enter your username:")
        password = st.text_input('Enter your password', type='password') 
    else:
        username = st.session_state.username

    if  st.button("Login"):  # Add password field
                token =  authentication(username, password)
                if token:
                    st.session_state.token = token
                    st.session_state.username = username
                    st.session_state.show_logged_in_message = True
                else:
                    st.error("Invalid User")

else:
    if "show_logged_in_message" not in st.session_state:
        st.session_state.show_logged_in_message = False

    if st.session_state.show_logged_in_message:
        logedin_username = st.session_state.username
        #classes = display_user_classes(st.session_state.username, st.session_state.token)
        

        if logedin_username == "admin":
            # Section to add new users
            st.header("Admin Panel - Add Users")
            new_user = st.text_input("Enter a new user:")
            new_user_password = st.text_input("Enter password for user:", type="password")
            gen_token_limit = st.slider('Adjust the generated token limit', min_value=0, max_value=100000, value=1000)
            prompt_token_limit = st.slider('Adjust the Input prompt token limit', min_value=0, max_value=100000, value=10000)

            if st.button("Add User") and new_user and new_user_password:
                new_user = add_user(new_user, new_user_password, gen_token_limit, prompt_token_limit, st.session_state.token)
                if new_user:
                    st.success("New user added successfully!")
                else:
                    st.error("User already exists")

        else:

            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])


            if prompt := st.chat_input(key="prompt"):    
                    #response = chat(model="meta-llama/Llama-2-7b-chat-hf", inference_endpoint="http://localhost:8500/v1",prompt=prompt, memory=False, username=logedin_username, conversation_number=-1,access_token=st.session_state.token)
                    response = retrieving_messages(username=logedin_username,converation_number='1' ,access_token=st.session_state.token)
                    st.chat_message("human").write(response)
                    #st.chat_message("human").write(response["data"])
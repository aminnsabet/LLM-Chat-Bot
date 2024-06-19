import streamlit as st
import requests
import json
import time
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

BASE_URL = "http://localhost:8086"
Weaviate_endpoint = "/vector_DB_request/"

def process_text(text):
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = text.replace("\\n", "\n")
    return text

def send_vector_db_request(access_token, json_data, endpoint, uploaded_file=None):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.post(f"{BASE_URL}{endpoint}", data=json_data, headers=headers, files=uploaded_file)
    return response    

def authentication(username, password):
    data = {"username": username, "password": password}
    resp = requests.post(f"{BASE_URL}/token", data=data)
    if "access_token" not in resp.json():
        return None
    return resp.json()["access_token"]

def add_user(username, password, gen_token_limit, prompt_token_limit, access_token):
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
    query_data = {"username": username}
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.post(f"{BASE_URL}/db_request/get_user_conversations/", json=query_data, headers=headers)
    if resp.status_code == 200:
        conversations = resp.json()['conversations']
        names = [d["name"] for d in conversations]
        if names:
            return {"names": names, "conversations": conversations}
        return None
    else:
        return None


def chat(model, inference_endpoint, prompt, memory, username, conversation_number, access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    query_data = {
        "model": model,
        "inference_endpoint": inference_endpoint,
        "prompt": prompt,
        "memory": memory,
        "conversation_number": conversation_number,
        "username": username
    }
    resp = requests.post(f"{BASE_URL}/llm_request", json=query_data, headers=headers)
    return resp.json()

def retrieving_messages(username, access_token, conversation_number=None):
    headers = {"Authorization": f"Bearer {access_token}"}
    query_data = {
        "username": username,
        "conversation_number": conversation_number
    }
    resp = requests.post(f"{BASE_URL}/db_request/retrieve_conversation", json=query_data, headers=headers)
    
    if resp.status_code == 200:
        if resp.json()["conversation_id"]:
            messages = SQLChatMessageHistory(resp.json()["conversation_id"], "sqlite:////home/amin_sabet/dev/LLM-Chat-Bot/API/memory.db").get_messages()
            return messages
        else:
            return None
    else:
        return None

def add_LLM(model_name, access_token, HF_ACCESS_TOKEN, MAX_MODEL_LEN, SEED):
    headers = {"Authorization": f"Bearer {access_token}"}
    query_data = {
        "HUGGING_FACE_HUB_TOKEN": str(HF_ACCESS_TOKEN),
        "MODEL": str(model_name),
        "TOKENIZER": "auto",
        "MAX_MODEL_LEN": int(MAX_MODEL_LEN),
        "TENSOR_PARALLEL_SIZE": 1,
        "SEED": int(SEED),
        "QUANTIZATION": "None"
    }
    resp = requests.post(f"{BASE_URL}/vllm_init", json=query_data, headers=headers)

    if resp.status_code == 200:
        if resp.json()["status"] == "healthy":
            return resp.json()
        else:
            return False
    else:
        return False

def retrieve_user_conversations_info(username, access_token):
    query_data = {"username": username}
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.post(f"{BASE_URL}/db_request/retrieve_all_conversations/", json=query_data, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        return None

def add_conversation(username,access_token, conversation_name=None):
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
    if "username" not in st.session_state:
        username = st.text_input("Enter your username:")
        password = st.text_input('Enter your password', type='password')
    else:
        username = st.session_state.username

    if st.button("Login"):
        token = authentication(username, password)
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

        if logedin_username == "admin":
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
            st.sidebar.markdown("---")
            st.sidebar.markdown("<br>", unsafe_allow_html=True)
            st.sidebar.subheader("User Access Token:")
            show_token = st.sidebar.button("Show Token")
            if show_token:
                st.sidebar.code(st.session_state.token)

            st.sidebar.markdown("---")
            st.sidebar.markdown("<br>", unsafe_allow_html=True)
            expander = st.sidebar.expander("Add New Model")
            model_name = expander.text_input("Enter the model name from Huggingface:")
            HF_ACCESS_TOKEN = expander.text_input("Enter your Huggingface Access Token:")
            MAX_MODEL_LEN = expander.slider('Adjust the maximum model length', min_value=32, max_value=1024, value=512)
            SEED = expander.slider('Adjust the seed', min_value=1, max_value=100, value=42)
            add_model = expander.button("Add")
            if add_model:
                with expander:
                    with st.spinner("Adding model..."):
                        response = add_LLM(model_name, logedin_username, HF_ACCESS_TOKEN, MAX_MODEL_LEN, SEED)
                        if response:
                            expander.success("Model added successfully!")
                        else:
                            expander.error("Error adding model")

            st.sidebar.markdown("---")
            st.sidebar.markdown("<br>", unsafe_allow_html=True)
            st.sidebar.subheader("Chat History:")
            if st.sidebar.button("New Chat", key="new_chat"):
                add_conversation(logedin_username, st.session_state.token)
                user_chat_list = retrieve_user_conversations_info(logedin_username, st.session_state.token)

            user_chat_list = retrieve_user_conversations_info(logedin_username, st.session_state.token)

            if not user_chat_list:
                st.sidebar.write("No previous chats")
                selected_chat = None
            else:
                selected_chat = st.sidebar.radio(
                    "Select a conversation:",
                    user_chat_list["conversation_numbers"],
                    index=len(user_chat_list["conversation_numbers"]) - 1,
                )

            if "messages" not in st.session_state:
                st.session_state.messages = []

            if selected_chat:
                st.session_state.messages = []
                msgs = retrieving_messages(logedin_username, st.session_state.token, conversation_number=selected_chat)
                for msg in msgs:
                    st.chat_message(msg.type).write(msg.content)
                    st.session_state.messages.append({"role": msg.type, "content": msg.content})

            if prompt := st.chat_input():
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = chat(model="meta-llama/Llama-2-7b-chat-hf", inference_endpoint="http://localhost:8500/v1", prompt=prompt, memory=True, username=logedin_username, conversation_number=selected_chat, access_token=st.session_state.token)
                        if response:   
                            response = response["data"]
                            full_response = process_text(response)
                            st.write(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                            add_conversation(logedin_username, st.session_state.messages, st.session_state.token)

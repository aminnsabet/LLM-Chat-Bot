
# Large Langage Model As a Service


<p align="center">
  <img src="Diagram.svg" >
</p>

# LLM-as-a-Service Setup Guide

This  guide provides the steps to set up and run the LLM-as-a-Service on your machine.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA A100 or equivalent.
- **Software**: CUDA version 11.8 or above.

## Python Environment Setup

Install required Python packages
```
pip install -r requirements.txt
```
## Configuration

In the cluster_conf.yaml, you must specify access tokens for both your Huggingface and Weights & Biases accounts.

If you wish not to monitor deployment logging, set WANDB_ENABLE to False.
When WANDB_ENABLE is set to False, there's no need to provide a Weights & Biases token.
## Run Weaviate container 

```
sudo docker compose up -d
```

## Running the Inference Service Locally 
Initialize a RAY cluster
```
ray start --head
```
Build the configuration for your backend service (only once).
```
serve build backend:app -o config.yaml
```
Deploy the application
```
serve deploy config.yaml
```

## Running the Inference Service on a Kubernutes Cluster

First you need to set up your KubeRay Cluster. Follow the steps [here](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/raycluster-quick-start.html#kuberay-raycluster-quickstart)

Once you set up the Kubernetes cluster, set the vlaue of `cluster_URL` in cluster_config.yaml.
Submit the job tto your cluster by runing 
```
python job_submission.py
```
Modeify `Ray_service_URL` in cluster_config.yaml to your Kubernetes cluster URL address. 

### Run the authentication microservice
Change the directory to API
```
cd API
```
Run authentication micro-service. 
```
uvicorn app:app --reload --port 8081
```

# Use LLM service. 

## Use Streamlit UI

Run the Streamlit UI 

```
streamlit run UI/main.py
```
For the first time login as admin to add new users.
admin creientials are stored in ```cluster_config.yaml```
More details of streamlit UI can be found [here](UI/README.md)

## Use EscherCloudAI API
To use API service you need to get user access tokens, using streamlit UI, to be able to send request to LLMs API.
You can fine more details of API [here](eschercloud/README.md)


I want to write a documentation for a software application I have built. I want to write it industry level. 
The in the main Directory is called API in which I have app folder, main.py, cluster_config.yaml, chat_bot_db.db and memory.db.
main.py: is the file to run the application.
cluster_config.yaml stores configurations such as port number and admin passweords and so on.
chat_bot_db.db is the sql db that stores iformations about users including username, enabled/disabled, role, tokenlimit and so on.
memory.db is used for storing user chats with LLM. it takes a key and return a chat for that key.


Inside app directory I have follwoing files and directories. APIs.py, database.py, loggign_config.py, __init__.py. routers dependencies and services are directories.
APIs.py defines the formant of APIs for inference, building vllm and interacting with database. 
database.py defines functions for interacting with chat_bot_db.db.
__init__.py routes the API requests comming to apllication into VLLM build, LLm inference or interactiving with Databse. 
loggign_config.py sets up the logger for the application.


routers directory I have following files.
vllm_build.py (this files recives request for building vllm engine and start a cotainer and vllm engine and returns the enpoints for inferencing).
OpenAI_API.py recieves the inference request that are OpenAI API compatible.
inference.py recieves inference request that are not openAI API compatible but enable users to have more features such as memory chat.
db_functions.py process the database requests and returns the corresponding informations.
login_router.py generates the authentication tokens and returns it.

in routers directory I have utils directory in which I have follwoing files: backend_database.py and backend_vLLM.py.
backend_database.py defines chat_bot_db.db and all functions around it to read/write data into it. 
backend_vLLM.py defines all the functions around inference of LLm and pre/post processing of LLM inference requests. 
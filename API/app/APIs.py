from pydantic import BaseModel
from typing import Optional
from typing import Any, List

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str = None

class LoginUser(BaseModel):
    username: str
    disabled: bool = None

class UserInDB(LoginUser):  
    hashed_password: str

class Data(BaseModel): 
    prompt: str


class VllmRequest(BaseModel):
    HUGGING_FACE_HUB_TOKEN: Optional[str]="None"
    MODEL: Optional[str] = "None"
    TOKENIZER: Optional[str] = 'auto'
    MAX_MODEL_LEN: Optional[int] = 10000
    TENSOR_PARALLEL_SIZE: Optional[int] = 1
    SEED: Optional[int] = 42
    QUANTIZATION: Optional[str]="None"
    container_id: Optional[str] = "None"
    engine_name: Optional[str] = "None"


class InferenceRequest(BaseModel):
    model: str= None
    inference_endpoint:str = None
    prompt: Optional[str]="Hi Nscale chatbot"
    memory: Optional[bool]=False
    conversation_number: Optional[int]=-1
    username: Optional[str]="None" 

class UserRequest(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    prompt_token_number: Optional[int] = None
    gen_token_number: Optional[int] = None
    gen_token_limit: Optional[int] = None
    prompt_token_limit: Optional[int] = None


class DataBaseRequest(BaseModel):
    username: Optional[str] = None
    conversation_number: Optional[int] = None
    conversation_name: Optional[str]=None
    user_id: Optional[int]=None
    gen_token_number: Optional[int] = None
    gen_token_limit: Optional[int] = None
    prompt_token_limit: Optional[int] = None

class VectorDBRequest(BaseModel):
    username: Optional[str] 
    class_name: Optional[str] 
    mode: Optional[str]
    vectorDB_type: Optional[str] = "Weaviate"
    file_path: Optional[str] = None
    file_title: Optional[str] = None

class ArxivInput(BaseModel):
    username: Optional[str]
    class_name: Optional[str] = None
    query: Optional[str] = None
    paper_limit: Optional[int] = None
    recursive_mode: Optional[int] = None
    mode: Optional[str]
    title: Optional[str] = None
    url: Optional[str] = None
    file_path: Optional[str] = None
    dir_name: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mock-gpt-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

    
   
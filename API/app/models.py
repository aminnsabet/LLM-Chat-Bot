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

# 
class VllmBuildRequest(BaseModel):
    HUGGING_FACE_HUB_TOKEN: str
    MODEL: str
    TOKENIZER: Optional[str] = 'auto'
    MAX_MODEL_LEN: Optional[int] = 512
    TENSOR_PARALLEL_SIZE: Optional[int] = 1
    SEED: Optional[int] = 42
    QUANTIZATION: Optional[str]="None"


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
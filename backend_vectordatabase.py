
import os
import yaml
from typing import Optional
from sqlalchemy.orm import Session
import weaviate
import logging
import ray
from typing import Any, List
import pypdf
import ray
from ray import serve
import os
import weaviate
from langchain.vectorstores import Weaviate
from langchain.text_splitter import CharacterTextSplitter
import yaml
from langchain_community.document_loaders import TextLoader
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from typing import Optional
from pydantic import BaseModel
from backend_database import Database
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from weaviate.classes.config import Configure, Property, DataType
from weaviate.auth import AuthApiKey
from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter
from langchain_community.llms import VLLMOpenAI
from weaviate.exceptions import UnexpectedStatusCodeException
#from langchain_weaviate.vectorstores import WeaviateVectorStore
#from API.app.models import VectorDBRequest

from sentence_transformers import SentenceTransformer
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.prompts import PromptTemplate
#from langchain.chains import RetrievalQA

from langchain_community.retrievers import (
    WeaviateHybridSearchRetriever,
)
from langchain_core.documents import Document

from langchain.vectorstores import Weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import asyncio
from pypdf.errors import PdfStreamError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import Weaviate
from langchain import hub

#Error handling import:
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


with open("cluster_conf.yaml", 'r') as file:
    config = yaml.safe_load(file)
    config = Config(**config)

    
MAX_FILE_SIZE = config.max_file_size * 1024 * 1024  

class VDBaseInput(BaseModel):
    username: Optional[str] 
    class_name: Optional[str] 
    mode: Optional[str]
    vectorDB_type: Optional[str] = "Weaviate"
    file_path: Optional[str] = None
    file_title: Optional[str] = None
    query: Optional[str] = None
    model: Optional[str] = None
    inference_endpoint:Optional[str]=None
    memory: Optional[bool]=False
    conversation_number: Optional[int]=-1
    embedder: Optional[str]= "sentence-transformers/all-MiniLM-L6-v2"
    ray: Optional[bool] = False
    num_actors: Optional[int] = 1
    object_property: Optional[str] = None
    object_limit: Optional[int] = None



VDB_app = FastAPI()

class DocumentProcessingError(Exception):
    """Exception raised for errors in the document processing pipeline."""
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors



@VDB_app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(request: Request, exc: DocumentProcessingError):
    return JSONResponse(
        status_code=400,
        content=jsonable_encoder({"message": exc.message, "errors": exc.errors}),
    )

@VDB_app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # Log the error details here if necessary
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred"},
    )

class VLLMManager:
    def __init__(self):

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True


        self.database = Database()
        self.model = None
        self.logger.info(f"Checkpoint vllm manager before meta error 1! ")   
        # self.logger.info(f"Checking the init of VLLMManager and parameters: api_key {self.api_key}, base {self.api_base}, model name: {self.model_name}, kwargs: {self.model_kwargs}")

    def run_vllm_model(self, username, model, inference_endpoint):
        if self.model is None:
            self.logger.info(f"Checkpoint vllm run before meta error 1! ")   
            self.model = VLLMOpenAI(
                openai_api_key="EMPTY",
                openai_api_base=inference_endpoint,
                model_name=model,
                max_tokens= 1084
            )
            #text_to_log = self.model.invoke("Hi how are you?")
            #self.logger.info(f"Checking the result of run_vllm_model, output is model: {self.model} and logged text is : {text_to_log}")
            self.logger.info(f"Checkpoint vllm run before meta error 2! ")   
            return self.model
        
    def vllm_running_status(self):
        model_check = self.model
        self.logger.info(f"Checking the vllm running status: {model_check}")
        if model_check is None:
            return False
        else:
            return True


    def shutdown_model(self):
        self.model = None
        self.logger.info(f"Shutting down model: {self.model}")

@ray.remote(num_cpus=0.2, num_gpus=0.2)
class WeaviateEmbedder:
    def __init__(self, class_name=None):
        self.time_taken = 0
        self.text_list = []
        self.class_name= class_name
        
    async def run_split_pdf(self, document):
        serialized_docs = await self.weaviate_split_pdf(document)
        return serialized_docs

    async def run_embedder_on_text(self, serialized_docs, collection_name):
        doc_list = await self.adding_weaviate_document(serialized_docs, collection_name)
        return doc_list

    async def weaviate_split_pdf(self, docs):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_docs = text_splitter.split_documents(docs)
        serialized_docs = [await self.weaviate_serialize_document(doc) for doc in text_docs]

        return serialized_docs

    async def weaviate_serialize_document(self, doc):

        document_title = doc.metadata.get('source', '').split('/')[-1]
        return {
            "page_content": doc.page_content,
            "document_title": document_title,
        }

    async def parse_pdf(self, file_path_list):
        documents = []
        for pdf_path in file_path_list:
            try:
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())

            except PdfStreamError as e:

                continue
        return documents

    async def convert_file_to_text(self, document_path):
        documents = await self.parse_pdf(document_path)

        return documents
    
    async def adding_weaviate_document(self, text_lst, collection_name=None):
        weaviate_client = weaviate.connect_to_local(  
                    port= 8900,
                )
        self.logger.info(f"Parsing documents with Ray")
        collection = weaviate_client.collections.get(str(collection_name))
        with collection.batch.fixed_size(batch_size=100) as batch:
            for data_row in text_lst:
                batch.add_object(
                    properties=data_row,
                )
                self.text_list.append(data_row)

        return self.text_list
    
    async def terminate_actors(self):
        ray.actor.exit_actor()
    
@serve.deployment(
    # ray_actor_options={"num_gpus": config.VD_deployment_num_gpus}, autoscaling_config={
    #     #"min_replicas": config.VD_min_replicas,
    #     "initial_replicas": config.VD_initial_replicas,
    #     #"max_replicas": config.VD_max_replicas,
    #     #"max_concurrent_queries": config.VD_max_concurrent_queries,
    #     }
        )

@serve.ingress(VDB_app)
class VectorDataBase:
    def __init__(self):

        self.weaviate_client = weaviate.connect_to_local( 
                    port= 8900,
                )

        self.database = Database()

        self.vllm_manager = None
        self.embedder_model = None
        self.current_llm = None

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

        self.logger.info(f"At initilization stage embedding model is: {self.embedder_model} and vllm model is: {self.vllm_manager}")

    def weaviate_serialize_document(self,doc):
        '''
        Description:
            Serializes a document for storage in Weaviate. It extracts the title from the document's metadata and combines it with the page content.

        Parameters:

            doc (Document): The document to be serialized.

        Returns:

            dict: A dictionary containing the serialized content of the document, including its title and page content.
        '''
        document_title = doc.metadata.get('source', '').split('/')[-1]
        return {
            "page_content": doc.page_content,
            "document_title": document_title,
        }
    
    def weaviate_split_multiple_pdf(self,docs):   
        '''
        Description:
            Splits multiple PDF documents into chunks for easier processing and storage. This function uses a recursive character text splitter to create smaller, manageable text documents.

        Parameters:

            docs (list): A list of document objects to be split.

        Returns:

            list: A list of serialized document chunks.
        ''' 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        text_docs = text_splitter.split_documents(docs)

        serialized_docs = [
                    self.weaviate_serialize_document(doc) 
                    for doc in text_docs
                        ]
        return serialized_docs	

    def split_workload(self, file_paths, num_actors):
        return [file_paths[i::num_actors] for i in range(num_actors)]

    def divide_workload(self, num_actors, documents):
        '''
        Description:
            Divides a list of documents among a specified number of actors (processes or threads) to parallelize processing.

        Parameters:

            num_actors (int): The number of Ray actors (processes/threads) among which the workload will be divided.
            documents (list): A list of documents to be divided.

        Returns:

            list: A list of document lists, where each sublist corresponds to the documents assigned to one actor.
        '''
        docs_per_actor = len(documents) // num_actors

        doc_parts = [documents[i * docs_per_actor: (i + 1) * docs_per_actor] for i in range(num_actors)]

        if len(documents) % num_actors:
            doc_parts[-1].extend(documents[num_actors * docs_per_actor:])

        return doc_parts

    def parse_pdf(self, directory):    
        '''
        Description:
           Parses all PDF and text files in a given directory, creating a list of documents. It uses different loaders for PDF and text files and handles errors by skipping problematic files.

        Parameters:

            directory (str): The path to the directory containing PDF and text files.

        Returns:

            list: A list of document objects parsed from the files in the specified directory.
        '''
        documents = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(file_path, extract_images=False)
                    documents.extend(loader.load())

                elif file.endswith('.txt'):

                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                else:
                    self.logger.error(f"Unexpected error, Unsupported file type: {file}")
                    continue
            except Exception as e:
                self.logger.error(f"Error processing file {file}: {e}")
                continue 
        return documents

    def adding_weaviate_document(self, text_lst, collection_name=None):
        weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                    port= 8900,
                )
        self.text_list = []
        self.logger.info(f"Parsing documents without Ray")
        collection = weaviate_client.collections.get(str(collection_name))
        with collection.batch.fixed_size(batch_size=100) as batch:
            for data_row in text_lst:
                batch.add_object(
                    properties=data_row,
                )
                self.text_list.append(data_row)

        return self.text_list

    async def process_all_docs(self, dir, username, cls, ray, ray_actors):
        '''
        Description:
            Processes all documents in a specified directory, serializes them, and adds them to Weaviate. Handles both small and large document sets by splitting the workload for efficient processing.

        Parameters:

            dir (str): Directory containing the documents to be processed.
            username (str): The username of the user processing the documents.
            cls (str): The class name for the documents in Weaviate.

        Returns:

            dict: A response indicating the status of the processing ('success' or 'error') and a message detailing the outcome.
        '''

        self.logger.info("Processing initiated.")
        try:
            full_class = str(username) + "_" + str(cls)

            parsed_pdf = self.parse_pdf(dir)
            self.logger.info(f"Checking the parsed_pdfs: {parsed_pdf}")
            splitted_workload = self.weaviate_split_multiple_pdf(parsed_pdf)

            if ray:
                workload = self.divide_workload(ray_actors, splitted_workload)
                weaviate_embedders = [WeaviateEmbedder.remote() for _ in range(ray_actors)]
                very_finals = [weaviate_embedder.run_embedder_on_text.remote(i, str(full_class)) for weaviate_embedder, i in zip(weaviate_embedders, workload)]
                results = await asyncio.gather(*very_finals)
                self.logger.info(f"Finished processing docs using Ray:")
            else:
                results = self.adding_weaviate_document(splitted_workload, full_class)
                self.logger.info(f"Finished processing docs without Ray")
            return {"status": "success", "message": "Documents processed successfully", "results": results}
        except DocumentProcessingError as e:
            self.logger.error(f"Error processing documents: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred while processing documents")

    def adding_weaviate_document_no_ray(self, text_lst, collection_name):
        try:
            self.weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                    port= 8900,   
                )
        except:
            self.logger.error("Error in connecting to Weaviate")

        self.logger.info(f"Checkpoint no ray 1 and check list {text_lst[0]}")
        self.weaviate_client.batch.configure(batch_size=100)
        with self.weaviate_client.batch as batch:
            for text in text_lst:
                batch.add_data_object(
                    text,
                    class_name=str(collection_name), 
        )
                self.text_list.append(text)
        return self.text_list


    def add_vdb_class(self, username, class_name,embedder=None, HF_token=None):
        '''
        Description:
            Creates a new class in the Weaviate database with the specified name and username. It also adds the class to the internal database.

        Parameters:

            username (str): The username associated with the new class.
            class_name (str): The name of the new class to be created.

        Returns:

            dict: A response indicating the outcome ('success' or 'error') and relevant messages.
        '''
        response = {"status": "initiated", "message": ""}
        try:            
            weaviate_client = weaviate.connect_to_local( 
                port= 8900,
            )
            cls = str(username) + "_" + str(class_name)    
            weaviate_client.collections.create(
                        cls,
                        vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
                        vector_index_config=Configure.VectorIndex.flat(),
                        properties=[  # properties configuration is optional
                            Property(name="document_title", data_type=DataType.TEXT),
                            Property(name="page_content", data_type=DataType.TEXT),
                        ],
            )        
            response["status"] = "success"
            response["message"] = f"Class '{cls}' successfully created."
        except UnexpectedStatusCodeException as e:
            self.logger.error(f"Failed to create class {cls} due to unexpected status code: {str(e)}")
            response["status"] = "error"
            response["message"] = "Failed to create the class due to a server error."
        except Exception as e:
            self.logger.error(f"Unexpected error while creating class {cls}: {str(e)}")
            response["status"] = "error"
            response["message"] = "An unexpected error occurred."

        return response

    def delete_collection(self, username, class_name):
            '''
            Description:
                Deletes a specified class from the Weaviate database and the internal database.

            Parameters:

                username (str): The username associated with the class to be deleted.
                class_name (str): The name of the class to be deleted.

            Returns:

                dict: A response indicating the outcome ('success' or 'error') and relevant messages
            '''
            try: 
                weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                    port= 8900,
                    
                )
                full_class_name = str(username) + "_" + str(class_name)
                if full_class_name is not None:
                    weaviate_client.collections.delete(full_class_name)
                    self.database.delete_collection({"username": username, "collection_name": class_name})
                    return {"success": f"Class {full_class_name} has been removed"}
                else:
                    return {"error": "No class name provided"}
            except Exception as e:
                return {"error": str(e)}

    def delete_weaviate_document(self, name, class_name):
        '''
        Description:
            Deletes a document from Weaviate based on its title and class name.

        Parameters:

            name (str): The title of the document to be deleted.
            cls_name (str): The class name under which the document is stored in Weaviate.
        '''
        try:
            document_name = str(name)
            selected_class = self.weaviate_client.collections.get(str(class_name))
            selected_class.data.delete_many(
                where=Filter.by_property("document_title").like(str(document_name))
            )
        except Exception as e:
                return {"error": str(e)}

    def query_objects_in_collection(self, username, class_name, query, property, object_limit):
        '''Supposed to return objects containg the query. Need to check if the behaviour is normal.'''
        try:
            weaviate_client = weaviate.connect_to_local(  
                    port= 8900,
                    
                )
            full_class_name = str(username) + "_" + str(class_name)
            collection = weaviate_client.collections.get(str(full_class_name))
            self.logger.info(f"Checks query obj in col with collection: {collection}")
            response = collection.query.fetch_objects(
                filters=(
                    Filter.by_property(str(property)).like(f"*{str(query)}*")
                ),
                limit=object_limit,
            )
            for obj in response.objects:
                response = obj.properties.get(str(property))
            return response
        except Exception as e:
                    return {"error": str(e)}

    def query_weaviate_document_names(self, username, class_name):
        '''
        !!!!!!!!!!!!!!!! Need update to weaviate v4 !!!!!
        Description:
            Queries the Weaviate database for the titles of all documents in a specified class.

        Parameters:

            username (str): The username associated with the class.
            class_name (str): The class name for which document titles are queried.

        Returns:

            list/dict: A list of document titles found, or an error message if no documents are found or an error occurs.
        '''
        try:
            weaviate_client = weaviate.Client("http://localhost:8900")
            prefix = username
            prefix = prefix.capitalize()
            class_properties = ["document_title"]
            class_name = class_name
            #full_class_name = str(username) + "_" + str(class_name)
            full_class_name = prefix + "_" + str(class_name)
            query = weaviate_client.query.get(full_class_name, class_properties)
            query = query.do()
            document_title_set = set()
            documents = query.get('data', {}).get('Get', {}).get(str(full_class_name), [])
            for document in documents:
                document_title = document.get('document_title')
                if document_title is not None:
                    document_title_set.add(document_title)
            if document_title_set is not None:
                self.logger.info(f"query success: {len(document_title_set)} documents found")
                return list(document_title_set)
            else:
                return {"error": "No documents found"}
        
        except Exception as e:
                return {"error": str(e)}
        
    def get_all_objects(self, username, class_name):
        doc_list = []
        full_class_name = str(username) + "_" + str(class_name)

        collection = self.weaviate_client.collections.get(str(full_class_name))
        for item in collection.iterator():
            doc_list.append(item.properties)
            self.logger.info(f"Collection content: {item.properties}")
        return doc_list

    def get_classes(self, username):
        try:
            weaviate_client = weaviate.connect_to_local(  
                    port= 8900,
                )
            response = weaviate_client.collections.list_all()
            if response:
                return {"status": "success", "data": response} 
            else:
                return {"status": "success", "message": "No classes found", "data": []}  #TODO
        except Exception as e:
            self.logger.error(f"Error when display the collections: {str(e)}")
            return {"status": "error", "message": str(e)}
        
    ### SEARCH FUNCTIONS ###
    def basic_vector_search(self, username, cls):
        self.weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                    headers={
                        "X-HuggingFace-Api-Key": "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"
                        }
                )
        class_name = str(username) + "_" + str(cls)
        selected_class = self.weaviate_client.collections.get(class_name)
        response = selected_class.query.fetch_objects(
            #include_vector = True,
            limit= 2
        )
        return response
    
    def similarity_vector_search(self, username, cls, user_query):
        self.weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                    headers={
                        "X-HuggingFace-Api-Key": "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"
                        }
                )
        class_name = str(username) + "_" + str(cls)
        selected_class = self.weaviate_client.collections.get(class_name)
        self.logger.info(f"info on query passed: {user_query} and the class name: {selected_class}")
        response = selected_class.query.near_text(
            query=str(user_query),
            limit=2,
            return_metadata=MetadataQuery(distance=True)
        )
        for o in response.objects:
            self.logger.info(f"logged properties: {o.properties}")
            self.logger.info(f"logged distance: {o.metadata.distance}")
        return response

    def keyword_search(self, username, cls, user_query):
        self.weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                    headers={
                        "X-HuggingFace-Api-Key": "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"
                        }
                )
        class_name = str(username) + "_" + str(cls)
        selected_class = self.weaviate_client.collections.get(class_name)
        self.logger.info(f"info on query passed: {user_query} and the class name: {selected_class}")
        response = selected_class.query.bm25(
            query=str(user_query),
            return_metadata=MetadataQuery(score=True),
            limit=3
        )
        for o in response.objects:
            self.logger.info(f"logged properties: {o.properties}")
            self.logger.info(f"logged score: {o.metadata.score}")
        return response
    
    ### Retriever functions ###
    def initialize_vllm_manager(self, username, model, inference_endpoint):
        try:
            if model and inference_endpoint is not None:
                self.vllm_manager = VLLMManager()
                self.current_llm = self.vllm_manager.run_vllm_model(username, model, inference_endpoint)
                self.logger.info(f"check the success init status: {self.vllm_manager}, and the current llm : {self.current_llm}")
                return self.current_llm
            else: 
                raise ValueError("Model or inference endpoint is not provided")
        except Exception as e:
            self.logger.error(f"Failed to initialize VLLM Manager: {e}")
            return None
        
    def run_inference_on_vllm(self, username, model, query):
        self.logger.info(f"Checking the current llm status: {model} and the query: {query}")
        response = model.invoke(str(query))
        self.logger.info(f"logging the query and response: {response}")
        return response

    def generate_prompt_template(self,):
        template= """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use five sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:
        """
        prompt=ChatPromptTemplate.from_template(template)
        return prompt

    def get_collection_based_retriver(self, client, class_name, embedder):
        try:
            self.weaviate_client = weaviate.connect_to_local(   
                        port=8900
                    )
            model_name = "BAAI/bge-base-en-v1.5"#str(embedder)
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

            ## Log the hf embedder
            self.logger.info(f"collection is {str(class_name)}, and client: {self.weaviate_client}, and embedder: {hf} ")

            db = WeaviateVectorStore(client=self.weaviate_client, index_name=str(class_name), embedding=hf, text_key='page_content')

            self.logger.info(f"checking the vectorstore 2 : {db}")
            self.retriever = db.as_retriever()
            return self.retriever
        except Exception as e:
            self.logger.error(f"Failed to create retriever for {class_name} using {embedder}: {e}")
            return None

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieval_augmented_generation(self, username, class_name, embedder_name, model, inference_endpoint, query):
        try:
            self.weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                        port=8900
                    )
            # VVLM INIT
            self.current_llm = self.initialize_vllm_manager(username, model, inference_endpoint)
            if not self.current_llm:
                raise Exception("Failed to initialize language model.")

            self.logger.info(f"Checking the embedder : {embedder_name} and the current llm: {self.current_llm}")

            full_class_name = str(username) + "_" + str(class_name)
            selected_class = self.weaviate_client.collections.get(full_class_name)
            
            retriever = self.get_collection_based_retriver(self.weaviate_client, str(full_class_name), embedder_name)
            if not retriever:
                raise Exception("Failed to initialize retriever.")

            prompt_template = self.generate_prompt_template()

            rag_chain=({"context":retriever,"question":RunnablePassthrough()}
                | prompt_template
                | self.current_llm
                | StrOutputParser()
            )
            self.logger.info(f"logging the rag chain: {rag_chain} and retriever: {retriever}")
            response=rag_chain.invoke(str(query))

            self.logger.info(f"Checking the response of do rag: {response}")

            return response
        except Exception as e:
            self.logger.error(f"Error in retrieval augmented generation: {e}")
            return None

    def multi_query_retrieval_augmented_generation(self, username, class_name, model, inference_endpoint, embedder_name, query):
        from langchain_core.runnables import chain
        from operator import itemgetter
        self.current_llm = self.initialize_vllm_manager(username, model, inference_endpoint)
        full_class_name = str(username) + "_" + str(class_name)
        retriever = self.get_collection_based_retriver(self.weaviate_client, str(full_class_name), embedder_name)

        multi_query_template = """You are an AI language model assistant. Your task is to generate three different version of the given user question 
        to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of similarity search. 
        Provide these laternative questions separated by newlines, Original question: {question}"""
        prompt_perspectives = ChatPromptTemplate.from_template(multi_query_template)

        generate_queries = (
            prompt_perspectives
            | self.current_llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        def get_unique_union(documents: list[list]):
            from langchain.load import dumps, loads
            flatten = [dumps(doc) for sublist in documents for doc in sublist]
            unique_doc = list(set(flatten))
            return [loads(doc) for doc in unique_doc]
        
        question = query
        ret_chain = generate_queries | retriever.map() | get_unique_union
        docs = ret_chain.invoke({"question":question})
        self.logger.info(f"Checking the generated queries: {docs}")

        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        final_chain = (
            {"context": ret_chain,
             "question": itemgetter("question")}
             |prompt
             |self.current_llm
             |StrOutputParser()
        )
        response = final_chain.invoke({"question":question})
        self.logger.info(f"Checking the response after multi queries: {response}")
        return response

    

    def hyde_retrieval_augmented_gneneration(self, username, class_name, model, inference_endpoint, embedder_name, query):
        try:
            from langchain_core.runnables import chain
            self.weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                        port=8900
                    )
            self.current_llm = self.initialize_vllm_manager(username, model, inference_endpoint)
            if not self.current_llm:
                raise ValueError("Failed to initialize vector language model manager.")
            full_class_name = str(username) + "_" + str(class_name)
            retriever = self.get_collection_based_retriver(self.weaviate_client, str(full_class_name), embedder_name)
            if not retriever:
                raise ValueError("Failed to initialize collection-based retriever.")

            hyde_template = """
                Even if you do not know the full answer, generate a one-paragraph hypothetical answer to the below question.

                {question}"""
            
            hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
            hyde_retriever = hyde_prompt | self.current_llm | StrOutputParser()

            @chain
            def hyde_ret(question):
                hypothetical_document = hyde_retriever.invoke({"question":question})
                self.logger.info(f"Checking the hypo documents: {hypothetical_document}")
                if not hypothetical_document:
                    raise ValueError("Failed to generate hypothetical document.")
                return retriever.invoke(hypothetical_document)
            
            template = """
                Answer the question based only on the following context:
                {context}
                Question: {question}
                """
            
            prompt = ChatPromptTemplate.from_template(template)

            answer_chain = prompt | self.current_llm | StrOutputParser()

            @chain
            def final_chain(question):
                documents = hyde_ret.invoke({"question": question})
                self.logger.info(f"Logging the documents in hyde: {documents}")
                if not documents:
                    raise ValueError("Failed to retrieve documents based on hypothetical answers.")
                print("hyde documents: ", documents)
                for s in answer_chain.stream({"question":question,"context":documents}):
                    yield s

            response =  final_chain.invoke(str(query))
            self.logger.info(f"logging the response of hyde: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Error in hyde retrieval augmented generation: {e}")
            return None
    

    @VDB_app.post("/")
    async def VectorDataBase(self, request: VDBaseInput):
            try:
                if request.mode == "add_to_collection":
                    response  = await self.process_all_docs(request.file_path, request.username, request.class_name, request.ray, request.num_actors)
                    self.logger.info(f"response: {response}: %s", )
                elif request.mode == "basic_search":
                    response = self.basic_vector_search(request.username, request.class_name)
                    self.logger.info(f"Here is the response after basic search request: {response}")
                    return response
                elif request.mode == "similarity_search":
                    response = self.similarity_vector_search(request.username, request.class_name, request.query)
                    self.logger.info(f"Here is the response after similarity search request: {response}")
                    return response
                elif request.mode == "keyword_search":
                    response = self.keyword_search(request.username, request.class_name, request.query)
                    self.logger.info(f"Here is the response after keyword search request: {response}")
                    return response
                elif request.mode == "display_collections":
                    response = self.get_classes(request.username)
                    self.logger.info(f"classes: {response}: %s", )
                    return response
                elif request.mode == "display_documents":
                    response = self.query_weaviate_document_names(request.username, request.class_name)
                    return response
                elif request.mode == "query_objects_in_collection":
                    response = self.query_objects_in_collection(request.username, request.class_name, request.query, request.object_property, request.object_limit)
                    return response
                elif request.mode == "display_all_objects":
                    response = self.get_all_objects(request.username, request.class_name)
                    return response
                elif request.mode == "delete_collection":
                    response = self.delete_collection(request.username, request.class_name)
                    self.logger.info(f"collection delete: {response}: %s", )
                    return response
                elif request.mode == "delete_document":
                    username = request.username
                    class_name = request.class_name
                    full_class_name = str(username) + "_" + str(class_name)
                    self.logger.info(f"checking the request/ {request}: and file title {request.file_title}")
                    response = self.delete_weaviate_document(request.file_title, full_class_name)
                    return response
                elif request.mode == "create_collection":
                    self.logger.info(f"checking the request/ {request}: %s", )
                    response = self.add_vdb_class(request.username, request.class_name)
                    return response
                elif request.mode == "initialize_vllm":
                    self.logger.info(f"Checking the init vllm request args: {request}")
                    response = self.initialize_vllm_manager(request.username, request.model, request.inference_endpoint)
                    return response
                elif request.mode == "vllm_inference":
                    self.logger.info(f"Checking the vllm inference from VDB API and its args: {request.username}, and query {request.query} and current llm {self.current_llm}")
                    response = self.run_inference_on_vllm(request.username, self.current_llm, request.query)
                    self.logger.info(f"logging the inference response: {response}")
                    return response
                elif request.mode == "rag":
                    response = self.retrieval_augmented_generation(request.username, request.class_name, request.embedder, request.model, request.inference_endpoint, request.query)
                    self.logger.info(f"Checking the response of do rag: {response}")
                    return response
                elif request.mode == "rag_hyde":
                    response = self.hyde_retrieval_augmented_gneneration(request.username, request.class_name, request.model, request.inference_endpoint, request.embedder, request.query)
                    self.logger.info(f"CHecking after hyde the response: {response}")
                    return response
                elif request.mode == "rag_mq":
                    response = self.multi_query_retrieval_augmented_generation(request.username, request.class_name, request.model, request.inference_endpoint, request.embedder, request.query)
                    self.logger.info(f"Checking the response after MQ RAG: {response}")
                    return response
                self.logger.info(f"request processed successfully {request}: %s", )
                return {"username": request.username, "response": response}
            except Exception as e:
                self.logger.error("An error occurred: %s", str(e))
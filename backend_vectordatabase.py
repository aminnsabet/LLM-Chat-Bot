
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



VDB_app = FastAPI()

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

        # self.logger.info(f"Checking the init of VLLMManager and parameters: api_key {self.api_key}, base {self.api_base}, model name: {self.model_name}, kwargs: {self.model_kwargs}")

    def run_vllm_model(self, username, model, inference_endpoint):
        if self.model is None:
            self.model = VLLMOpenAI(
                openai_api_key="EMPTY",
                openai_api_base=inference_endpoint,
                model_name=model,
                model_kwargs={"stop": ["."]},
                streaming=True,
            )
            #text_to_log = self.model.invoke("Hi how are you?")
            #self.logger.info(f"Checking the result of run_vllm_model, output is model: {self.model} and logged text is : {text_to_log}")
            
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

@ray.remote(num_cpus=1)
class WeaviateEmbedder:
    def __init__(self):
        self.time_taken = 0
        self.text_list = []
        # adding logger for debugging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="app.log",  # specify the file name if you want logging to be stored in a file
            filemode="a",  # append to the log file if it exists
        )

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

        try:
            self.weaviate_client = weaviate.Client(
                url=config.weaviate_client_url,   
            )
        except:
            self.logger.error("Error in connecting to Weaviate")

    def adding_weaviate_document(self, text_lst, collection_name):
        self.weaviate_client.batch.configure(batch_size=100)
        with self.weaviate_client.batch as batch:
            for text in text_lst:
                batch.add_data_object(
                    text,
                    class_name=collection_name, 
                        #uuid=generate_uuid5(text),
        )
                self.text_list.append(text)
        # results= self.text_list
        # ray.get(results)
        return self.text_list

    def get(self):
        return self.lst_embeddings
    
    def get_time_taken(self):
        return self.time_taken
    
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

        self.weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                    headers={
                        "X-HuggingFace-Api-Key": "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"
                        }
                )

        self.num_actors = config.VD_number_actors
        self.chunk_size = config.VD_chunk_size
        self.chunk_overlap = config.VD_chunk_overlap
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
        #text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
        text_docs = text_splitter.split_documents(docs)

        serialized_docs = [
                    self.weaviate_serialize_document(doc) 
                    for doc in text_docs
                        ]
        return serialized_docs	

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
            if file.endswith('.pdf'):
                pdf_path = os.path.join(directory, file)
                try:
                    loader = PyPDFLoader(pdf_path, extract_images=False)
                    documents.extend(loader.load())
                except pypdf.errors.PdfStreamError as e:
                    print(f"Skipping file {file} due to error: {e}")
                    continue  # Skip this file and continue with the next one
            elif file.endswith('.txt'):
                text_path = os.path.join(directory, file)
                try:
                    loader = TextLoader(text_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error in file {file}: {e}")
                    continue
        self.logger.info(f"Check the parsed documents: {documents}")
        return documents

    def simple_add_doc(self, dir):
        weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                    headers={
                        "X-HuggingFace-Api-Key": "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"
                        }
                )

        documents = []
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}

        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        for file in os.listdir(dir):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(dir, file)
                try:
                    loader = PyPDFLoader(pdf_path, extract_images=False)
                    docs = loader.load()
                    text_splitter=CharacterTextSplitter(chunk_size=200,chunk_overlap=0)
                    document=text_splitter.split_documents(docs)
                    vs = WeaviateVectorStore.from_documents(document, embedding=hf, client=weaviate_client, index_name="Admin_test_class_4")
                    #documents.extend(loader.load())
                except pypdf.errors.PdfStreamError as e:
                    print(f"Skipping file {file} due to error: {e}")
                    continue  # Skip this file and continue with the next one

    def process_all_docs(self, dir, username, cls):
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

        response = {"status": "initiated", "message": ""}
        try:
            full_class = str(username) + "_" + str(cls)
            document_list = self.parse_pdf(dir)
            serialized_docs = self.weaviate_split_multiple_pdf(document_list)
            if len(serialized_docs) <= 3000000:
                self.add_weaviate_document(full_class, serialized_docs)
                response["status"] = "success"
                response["message"] = f"Processed {len(serialized_docs)} documents for class {full_class}."
            # else:
            #     doc_workload = self.divide_workload(self.num_actors, serialized_docs)
            #     self.add_weaviate_batch_documents(full_class, doc_workload)
            #     #self.logger.info(f"check weaviate add data, ")
            #     response["status"] = "success"
            #     response["message"] = f"Processed {len(serialized_docs)} documents in batches for class {full_class}."
            return response
        except Exception as e:
            response["status"] = "error"
            response["message"] = str(e)
            return response

    def adding_weaviate_document(self, text_lst, collection_name):
        self.weaviate_client.batch.configure(batch_size=100)
        with self.weaviate_client.batch as batch:
            for text in text_lst:
                batch.add_data_object(
                    text,
                    class_name=collection_name, 
                        #uuid=generate_uuid5(text),
        )
                self.text_list.append(text)
        results= self.text_list
        ray.get(results)
        return self.text_list


    # def adding_weaviate_document(self, text_lst, collection_name):
    #     self.weaviate_client.batch.configure(batch_size=100)

    #     with self.weaviate_client.batch as batch:
    #         for text in text_lst:
    #             batch.add_data_object(
    #                 text,
    #                 class_name=collection_name, 
    #                     #uuid=generate_uuid5(text),
    #             )
    #             self.text_list.append(text)
    #     return self.text_list

    def add_weaviate_document(self, cls, docs):
        '''
        Description:
            Adds a list of serialized documents to Weaviate under a specified class. Uses a remote WeaviateEmbedder actor for the operation.

        Parameters:

            cls (str): The class name under which the documents will be added.
            docs (list): A list of serialized documents to be added.
        '''
        actor = WeaviateEmbedder.remote()
        ray.get([actor.adding_weaviate_document.remote(docs, str(cls))])

    def add_weaviate_batch_documents(self, cls, doc_workload):
        '''
        Description:
            Adds documents to Weaviate in batches using multiple WeaviateEmbedder actors. This method is used for efficient processing of larger sets of documents.

        Parameters:

            cls (str): The class name under which the documents will be added.
            doc_workload (list): A list of document batches to be added, where each sublist is a separate batch.
        '''
        actors = [WeaviateEmbedder.remote() for _ in range(3)]
        self.logger.info(f"actors creation successful {actors}: %s", )
        [actor.adding_weaviate_document.remote(doc_part, str(cls)) for actor, doc_part in zip(actors, doc_workload)]
        self.logger.info(f"check 1st step of ray was successful", )
        self.logger.info(f"check if ray was successful:", )


    def add_vdb_class(self,username, class_name,embedder=None):
        '''
        Description:
            Creates a new class in the Weaviate database with the specified name and username. It also adds the class to the internal database.

        Parameters:

            username (str): The username associated with the new class.
            class_name (str): The name of the new class to be created.

        Returns:

            dict: A response indicating the outcome ('success' or 'error') and relevant messages.
        '''
        try:            
                weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                    headers={
                        "X-HuggingFace-Api-Key": "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"
                        }
                )

                prefix = username
                cls = str(prefix) + "_" + str(class_name)
                if embedder is None:
                    vectorizer = "sentence-transformers/all-MiniLM-L6-v2"
                else:
                    vectorizer = embedder
                weaviate_client.collections.create(
                        cls,
                        vectorizer_config=Configure.Vectorizer.text2vec_huggingface(
                            model=vectorizer,
                        ),
                        properties=[  # properties configuration is optional
                            Property(name="title", data_type=DataType.TEXT),
                            Property(name="body", data_type=DataType.TEXT),
                        ],
                    )
                database_response = self.database.add_collection({"username": username, "collection_name": class_name})
                if database_response:
                    self.logger.info("class name added successfully to database")     
                    self.logger.info(f"success: class {class_name} created for user {username}")
                    return {"success": f"Class {cls} created "}
                else:
                    return {"error": "No class name provided"}
        except Exception as e:
            return {"error": str(e)}

    def delete_weaviate_class(self, username, class_name):
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
                    headers={
                        "X-HuggingFace-Api-Key": "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"
                        }
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
            # self.weaviate_client.batch.delete_objects(
            #     class_name=cls_name,
            #     where={
            #         "path": ["document_title"],
            #         "operator": "Like",
            #         "valueText": document_name,
            #     }
            #)
        except Exception as e:
                return {"error": str(e)}

    def query_weaviate_document_names(self, username, class_name):
        '''
        Description:
            Queries the Weaviate database for the titles of all documents in a specified class.

        Parameters:

            username (str): The username associated with the class.
            class_name (str): The class name for which document titles are queried.

        Returns:

            list/dict: A list of document titles found, or an error message if no documents are found or an error occurs.
        '''
        try:
            weaviate_client = weaviate.Client("http://localhost:8080")
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
        weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                    headers={
                        "X-HuggingFace-Api-Key": "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"
                        }
                )
        full_class_name = str(username) + "_" + str(class_name)

        collection = self.weaviate_client.collections.get(str(full_class_name))
        for item in collection.iterator():
            self.logger.info(f"Collection content: {item.properties}")

    def get_classes(self, username):
        try:
            weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                    headers={
                        "X-HuggingFace-Api-Key": "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"
                        }
                )
            #weaviate_client = weaviate.Client("http://localhost:8080")
            username = username
            response = weaviate_client.collections.list_all()
            if response is not None:    
                return response
            else:
                return {"error": "No classes found"}
            # schema = weaviate_client.schema.get()
            # classes = schema.get('classes', []) 
            # prefix = str(username) + "_"
            # prefix = prefix.capitalize()
            # filtered_classes = [cls["class"].replace(prefix, "", 1) for cls in classes if cls["class"].startswith(prefix)] #[cls["class"] for cls in classes if cls["class"].startswith(prefix)]
            # if filtered_classes is not None:
            #     return filtered_classes
            # else:
            #     return {"error": "No classes found"}
        except Exception as e:
                return {"error": str(e)}
        
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
        if model and inference_endpoint is not None:   
            self.vllm_manager = VLLMManager()
            self.current_llm = self.vllm_manager.run_vllm_model(username, model, inference_endpoint)
            self.logger.info(f"check the success init status: {self.vllm_manager}, and the current llm : {self.current_llm}")
            return self.current_llm
        else: 
            self.logger.info(f"check the failed init status: {self.vllm_manager}")
            return None
        
    def run_inference_on_vllm(self, username, model, query):
        self.logger.info(f"Checking the current llm status: {model} and the query: {query}")
        response = model.invoke(str(query))
        self.logger.info(f"logging the query and response: {response}")
        return response

    def initilize_embedder(self, username, embedder_name=None):
        if embedder_name == None:
            self.embedder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.logger.info(f"Checking the embedding model mini: {self.embedder_model}")
            embeddings_test = self.embedder_model.encode("What is the capital of France?")
           # self.logger.info(f"logging the vector generated by embedder for checking: {embeddings_test}")
            return self.embedder_model
        else:
            self.embedder_model = SentenceTransformer(str(embedder_name))
            self.logger.info(f"Checking the embedding model cutsom: {self.embedder_model}")
            embeddings_test = self.embedder_model.encode("What is the capital of France?")
           # self.logger.info(f"logging the vector generated by embedder for checking: {embeddings_test}")
            return self.embedder_model

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
        # prompt_template = """Text: {context}

        # Question: {question}
        
        # Answer the question based on the text provided. If the text doesn't contain the answer, reply that the answer is not available.
        # """
        # PROMPT = PromptTemplate(
        #     tempalte=prompt_template, input_variables=["context", "question"]
        # )
        # chain_type_kwargs = {"prompt": PROMPT}
        # return chain_type_kwargs

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_collection_based_retriver(self, client, class_name, embedder):
        self.weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                    headers={
                        "X-HuggingFace-Api-Key": "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"
                        }
                )
        model_name = str(embedder)
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}

        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        ## Log the hf embedder

        self.logger.info(f"collection is {str(class_name)}, and client: {self.weaviate_client}, and embedder: {hf} ")

        db2 = WeaviateVectorStore(client=self.weaviate_client, index_name=str(class_name), embedding=hf, text_key='page_content')

        self.logger.info(f"checking the vectorstore 2 : {db2}")
        self.retriever = db2.as_retriever()

        self.logger.info(f"Check the retriever: {self.retriever}")
        return self.retriever

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieval_augmented_generation(self, username, class_name, embedder_name, model, inference_endpoint, query):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from langchain.vectorstores import Weaviate
        from langchain import hub
        self.weaviate_client = weaviate.connect_to_local(   # `weaviate_key`: your Weaviate API key
                    headers={
                        "X-HuggingFace-Api-Key": "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"
                        }
                )

        # VVLM INIT
        self.current_llm = self.initialize_vllm_manager(username, model, inference_endpoint)

        self.logger.info(f"Checking the embedder : {embedder_name} and the current llm: {self.current_llm}")

        full_class_name = str(username) + "_" + str(class_name)
        selected_class = self.weaviate_client.collections.get(full_class_name)
        
        retriever = self.get_collection_based_retriver(self.weaviate_client, str(full_class_name), embedder_name)

        #prompt_template = self.generate_prompt_template()

        prompt = hub.pull("rlm/rag-prompt")
       # docs = retriever.invoke(str(query))
       # self.logger.info(f"checking the retrievd documents: {docs}")
       
        rag_chain=(
            {"context":retriever | self.format_docs, "question":RunnablePassthrough()}
            |prompt
            |self.current_llm
            |StrOutputParser()
        )
        self.logger.info(f"logging the rag chain: {rag_chain} and retriever: {retriever}")
        response=rag_chain.invoke(str(query))

        self.logger.info(f"Checking the response of do rag: {response}")

        return response


    @VDB_app.post("/")
    async def VectorDataBase(self, request: VDBaseInput):
            try:
                if request.mode == "add_to_collection":
                    #self.logger.info(f"request received {request}: %s", )
                    response  = self.process_all_docs(request.file_path, request.username, request.class_name)
                    self.logger.info(f"Quick check of the embedder: {self.embedder_model}")
                    self.logger.info(f"response: {response}: %s", )
                elif request.mode == "simple_add_to_collection":
                    response = self.simple_add_doc(request.file_path)
                    self.logger.info(f"logging the simple add: {response}")
                    return response
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
                elif request.mode == "display_classes":
                    response = self.get_classes(request.username)
                    self.logger.info(f"classes: {response}: %s", )
                    return response
                elif request.mode == "display_documents":
                    response = self.query_weaviate_document_names(request.username, request.class_name)
                    return response
                elif request.mode == "display_all_objects":
                    response = self.get_all_objects(request.username, request.class_name)
                    return response
                elif request.mode == "delete_class":
                    response = self.delete_weaviate_class(request.username, request.class_name)
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
                    #self.logger.info(f"Quick check of the embedder: {self.embedder_model}")
                    self.logger.info(f"Checking the init vllm request args: {request}")
                    response = self.initialize_vllm_manager(request.username, request.model, request.inference_endpoint)
                    return response
                elif request.mode == "vllm_inference":
                    self.logger.info(f"Checking the vllm inference from VDB API and its args: {request.username}, and query {request.query} and current llm {self.current_llm}")
                    response = self.run_inference_on_vllm(request.username, self.current_llm, request.query)
                    self.logger.info(f"logging the inference response: {response}")
                    return response
                elif request.mode == "initialize_embedder":
                    self.embedder_model = self.initilize_embedder(request.username, request.embedder)
                    self.logger.info(f"Checking the response and the embedder creation: {self.embedder_model}, {request.embedder}")
                    return self.embedder_model
                elif request.mode == "do_rag":
                    response = self.retrieval_augmented_generation(request.username, request.class_name, request.embedder, request.model, request.inference_endpoint, request.query)
                    self.logger.info(f"Checking the response of do rag: {response}")
                    return response
                    #retrieval_augmented_generation(self, username, class_name, embedder_name, model, inference_endpoint, query):
                self.logger.info(f"request processed successfully {request}: %s", )
                return {"username": request.username, "response": response}
            except Exception as e:
                self.logger.error("An error occurred: %s", str(e))
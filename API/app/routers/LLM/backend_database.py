import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
from dotenv import load_dotenv
from sqlalchemy import Boolean
from passlib.context import CryptContext
import yaml
from langchain_core.chat_history import BaseChatMessageHistory
import pathlib
from app.logging_config import setup_logger
import secrets
import string
# from app.logging_config import setup_logger
from sqlalchemy import UniqueConstraint
current_path = pathlib.Path(__file__).parent.parent.parent.parent
config_path = current_path/ 'cluster_conf.yaml'
# Environment and DB setup
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)



DATABASE_URL = config.get("DATABASE_URL")

DB_SERVICE_URL = config.get("DB_SERVICE_URL")  # Make sure this is used somewhere in your application
DB_DIR = config.get("DB_DIR")
if DB_DIR == "CURRENT_DIR":
    DB_DIR = os.getcwd()


db_name = config.get("DB_name","chat_bot_db")
db_path = os.path.join(DB_DIR, f"{db_name}.db")
DATABASE_URL = f"sqlite:///{db_path}"

# Check if the database file exists
if not os.path.exists(db_path):
    raise FileNotFoundError("Database file not found in ",db_path)
else:
    print(f"Database file {db_path} found.")


# SQLAlchemy base class
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)
    role = Column(String, default="User")
    available_models = Column(String, default="")
    gen_token_limit = Column(Integer, default=1000)
    prompt_token_limit = Column(Integer, default=10000)
    prompt_token_number = Column(Integer, default=0)
    gen_token_number = Column(Integer, default=0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    collection_names = Column(String, default="")
    


# Conversation model
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    conversation_number = Column(Integer)  # Add conversation number column
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    conversation_name = Column(String)
    conversation_id = Column(String, unique=True, index=True)
    __table_args__ = (UniqueConstraint('conversation_id', name='_conversation_id_uc'),)


# Create database engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


# Create tables if they don't exist
# Database session generator
class Database:
    def __init__(self):
        self.db = SessionLocal()
        self.logger = setup_logger()
    def generate_random_name(self, length=12):
        characters = string.ascii_letters + string.digits
        return ''.join(secrets.choice(characters) for _ in range(length))
    def add_conversation(self, input: dict):
        try:
            user = self.db.query(User).filter(User.username == input["username"]).first()

            if not user:
                self.logger.error(f"User {input['username']} not found")
                self.db.close()
                return {"error": f"User {input['username']} not found"}

            conversation_number = self.db.query(Conversation).filter(Conversation.user_id == user.id).count() + 1
            conversation_id = f"{user.username}_{user.id}_Conv_{conversation_number}"

            conversation_name = input.get("conversation_name", self.generate_random_name())

            conversation = Conversation(
                user_id=user.id,
                conversation_number=conversation_number,
                conversation_name=conversation_name,
                conversation_id=conversation_id  # Set the unique conversation_id
            )
            self.db.add(conversation)
            self.db.commit()
            self.db.close()
            return {"message": "Conversation added", "conversation_id": conversation_id}
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            return {"error": "An unexpected error occurred. Please try again later."}

    def update_token_usage(self, input):
        try:
            user = self.db.query(User).filter(User.username == input["username"]).first()

            if not user:
                self.logger.error(f"User {input['username']} not found")
                self.db.close()
                return {f"User {input['username']} not found"}

        

            if input.get("prompt_token_number"):
                user.prompt_token_number += input["prompt_token_number"]

            if input.get("gen_token_number"):
                user.gen_token_number += input["gen_token_number"]

                if user.gen_token_number > user.gen_token_limit  or user.prompt_token_number > user.prompt_token_limit:
                    user.disabled = True
                    user.gen_token_limit = 0
                    user.prompt_token_limit = 0

            self.db.commit()
            self.db.close()
            return {"message": "Conversation updated"}
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            return {"error": "An unexpected error occurred. Please try again later."}


    def retrieve_conversation(self, input):
        try:
            self.logger.info("Retrieving conversation...")

            user = self.db.query(User).filter(User.username == input["username"]).first()
            if not user:
                self.logger.error(f"User {input['username']} not found")
                self.db.close()
                return {"error": f"User {input['username']} not found"}

            self.logger.info(f"User found: {user}")

            # Check if conversation_number is provided in the input
            if "conversation_number" in input and input["conversation_number"]:
                conversation = (
                    self.db.query(Conversation)
                    .filter(
                        Conversation.conversation_number == input["conversation_number"],
                        Conversation.user_id == user.id,
                    )
                    .first()
                )
            else:
                # Get the latest conversation for the user
                conversation = (
                    self.db.query(Conversation)
                    .filter(Conversation.user_id == user.id)
                    .order_by(Conversation.timestamp.desc())
                    .first()
                )

            if not conversation:
                self.logger.error("Conversation not found")
                self.db.close()
                return {"error": "Conversation not found"}

            self.logger.info(f"Conversation found: {conversation}")


            self.logger.info("Conversation successfully retrieved")
            self.db.close()

            return {
                "user_id": conversation.user_id,
                "conversation_number": conversation.conversation_number,
                "timestamp": conversation.timestamp,
                "conversation_name": conversation.conversation_name,
                "conversation_id": conversation.conversation_id,
            }
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            return {"error": "An unexpected error occurred. Please try again later."}

        

    def add_collection(self, input):
        try:
            user = self.db.query(User).filter(User.username == input["username"]).first()

            if not user:
                self.logger.error("User not found")
                return {"error": "User not found"}

            if input["username"] and input["username"][0].isalpha():
                input["username"] = input["username"][0].upper() + input["username"][1:]

            username = input["username"]
            new_collection_name = f"{username}_{input['collection_name']}"

            if new_collection_name in user.collection_names.split(','):
                return {"collection_name": new_collection_name}

            user.collection_names += f",{new_collection_name}" if user.collection_names else new_collection_name
            self.db.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            return {"error": "An unexpected error occurred. Please try again later."}
    
    def check_collection_exists(self, input):
        try:
            user = self.db.query(User).filter(User.username == input["username"]).first()

            if not user:
                self.logger.error("User not found")
                return {"error": "User not found"}

            if input["username"] and input["username"][0].isalpha():
                input["username"] = input["username"][0].upper() + input["username"][1:]

            username = input["username"]
            new_collection_name = f"{username}_{input['collection_name']}"

            if new_collection_name in user.collection_names.split(','):
                return True

            return False
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            return {"error": "An unexpected error occurred. Please try again later."}

    def get_collections(self, input):
        try:
            user = self.db.query(User).filter(User.username == input["username"]).first()

            if not user:
                self.logger.error("User not found")
                return {"error": "User not found"}

            return {"collections": user.collection_names.split(',')}
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            return {"error": "An unexpected error occurred. Please try again later."}
    
    def delete_collection(self, input):
        try:
            user = self.db.query(User).filter(User.username == input["username"]).first()

            if not user:
                self.logger.error("User not found")
                return {"error": "User not found"}

            if input["username"] and input["username"][0].isalpha():
                input["username"] = input["username"][0].upper() + input["username"][1:]
            username = input["username"]
            new_collection_name = f"{username}_{input['collection_name']}"

            if new_collection_name == f"{username}_General_collection":
                return {"error": "Cannot delete the default collection"}

            collection_names = user.collection_names.split(',')

            if new_collection_name not in collection_names:
                return {"error": "Collection not found"}

            collection_names.remove(new_collection_name)
            user.collection_names = ','.join(collection_names)
            self.db.commit()
            return {"message": "Collection deleted"}
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            return {"error": "An unexpected error occurred. Please try again later."}
    
    
    def get_all_data(self):
        try:
            self.db = SessionLocal()
            users = self.db.query(User).all()
            conversations = self.db.query(Conversation).all()

            data = []
            for user in users:
                user_data = {
                    "id": user.id,
                    "username": user.username,
                    "prompt_token_number": user.prompt_token_number,
                    "gen_token_number": user.gen_token_number,
                    "timestamp": user.timestamp,
                    "disabled": user.disabled,
                    "gen_token_limit": user.gen_token_limit,
                    "prompt_token_limit": user.prompt_token_limit,
                    "role": user.role,
                    "collection_names": user.collection_names.split(","),
                    "conversations": []
                }

                # Add user's conversations
                for conversation in conversations:
                    if conversation.user_id == user.id:
                        user_data["conversations"].append({
                            "conversation_id": conversation.id,
                            "conversation_number": conversation.conversation_number,
                            "timestamp": conversation.timestamp,
                            "conversation_name": conversation.conversation_name
                        })

                data.append(user_data)

            return data
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            return {"error": "An unexpected error occurred. Please try again later."}
        finally:
            self.db.close()
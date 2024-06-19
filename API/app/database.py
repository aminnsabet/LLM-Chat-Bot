
import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker,declarative_base
from sqlalchemy.sql import func
from sqlalchemy import Boolean
from passlib.context import CryptContext
import pathlib
from sqlalchemy.orm import relationship
import os
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, UniqueConstraint, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
from passlib.context import CryptContext
import yaml
from langchain_core.chat_history import BaseChatMessageHistory
import pathlib
from app.logging_config import setup_logger
import secrets
import string
from dotenv import load_dotenv
#from app.logging_config import setup_logger
import yaml
from sqlalchemy import UniqueConstraint

current_path = pathlib.Path(__file__)

config_path = current_path.parent.parent / 'cluster_conf.yaml'

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

DATABASE_URL = config.get("DATABASE_URL", "sqlite:///./test.db")  # Provide a fallback if the env variable is missing
DB_DIR = config.get("DB_DIR","CURRENT_DIR")
if DB_DIR == "CURRENT_DIR":
    DB_DIR = os.getcwd()
    
db_name = config.get("DB_name","chat_bot_db")
db_path = os.path.join(DB_DIR, f"{db_name}.db")
DATABASE_URL = f"sqlite:///{db_path}"

# Check if the database file exists
if not os.path.exists(db_path):
    Base = declarative_base()
    engine = create_engine(DATABASE_URL)

DATABASE_URL = f"sqlite:///{db_path}"
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
    gen_token_limit = Column(Integer, default=1000)
    prompt_token_limit = Column(Integer, default=10000)
    prompt_token_number = Column(Integer, default=0)
    gen_token_number = Column(Integer, default=0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    collection_names = Column(String, default="")
    engines = relationship("VLLM_Engine", back_populates="user")

class VLLM_Engine(Base):
    __tablename__ = "vllm_engines"
    id = Column(Integer, primary_key=True, index=True)
    engine_name = Column(String, unique=True, index=True)
    container_id = Column(String, unique=True, index=True)
    model_name = Column(String, default="VLLM")
    quantized = Column(String, default="None")
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    max_model_length = Column(Integer, default=512)
    seed = Column(Integer, default=42)
    inference_end_point = Column(String, unique=True,index=True)  # New column
    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship("User", back_populates="engines")
    __table_args__ = (UniqueConstraint('engine_name', name='_engine_name_uc'),)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    conversation_number = Column(Integer)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    conversation_name = Column(String)
    conversation_id = Column(String, unique=True, index=True)
    __table_args__ = (UniqueConstraint('conversation_id', name='_conversation_id_uc'),)

# Create database engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

with SessionLocal() as db:
    admin_username =  config.get("admin_username","admin")
    admin_password = config.get("admin_password","admin")

    existing_admin = db.query(User).filter(User.username == admin_username).first()
    if not existing_admin:
        hashed_password = pwd_context.hash(admin_password)
        new_admin = User(username=admin_username, hashed_password=hashed_password, role="Admin")
        db.add(new_admin)
        db.commit()
        print(f"Admin {admin_username} created.")
    else:
        print(f"Admin {admin_username} already exists.")
# Create tables if they don't exist


# Database session generator
def get_db():
    """
    Returns a database session.

    This function creates a new instance of the database session class `SessionLocal`
    and yields it. The session is closed once the function is done executing.

    Returns:
        SessionLocal: An instance of the `SessionLocal` database session class.
    """
    db = SessionLocal()  # SessionLocal should be your database session class
    try:
        yield db
    finally:
        db.close()

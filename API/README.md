

# Run demo
In API directory run the following command

python main.py

# API Endpoints Documentation

## Authentication

### `/token`
**Description:** Generates an authentication token for the user.

**Request Format:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response Format:**
```json
{
  "username": "string",
  "access_token": "string",
  "token_type": "bearer"
}
```

## VLLM Engine Management

### `/vllm_request/start`
**Description:** Starts a VLLM engine.

**Request Format:**
```json
{
  "HUGGING_FACE_HUB_TOKEN": "string",   //Huggingface Access Token Required
  "Model": "string",                    // HuggingFace model name, Required
  "MAX_MODEL_LEN": "integer",           // Maximum input token into LLM, Optional
  "TENSOR_PARALLEL_SIZE": "integer",    // Optional
  "SEED": "integer"                     // Optional
}
```

**Response Format:**
```json
{
  "message": "string",
  "vLLM_endpoint": "string",
  "user_info": {
    // User-specific information
  },
  "status": "string" //healthy, unhealthy
}
```

### `/vllm_request/terminate`
**Description:** Terminates the Docker container running the VLLM engine.

**Request Format:**
```json
{
  "container_id": "string",    // Required
  "engine_name": "string"      // Optional
}
```

**Response Format:**
```json
{
  "status": "string"
}
```

### `/active_models/`
**Description:** Returns a list of active VLLMs for the user.

**Response Format:**
```json
[
  {
    "engine_name": "string",
    "container_id": "string",
    "model_name": "string",
    "quantized": "boolean",
    "max_model_length": "integer",
    "seed": "integer",
    "inference_end_point": "string",
    "timestamp": "string",
    "user_id": "string"
  }
]
```

## LLM Inference

### `/llm_request`
**Description:** Makes an inference request to a specified model.

**Request Format:**
```json
{
  "model": "string",               // Required
  "inference_endpoint": "string",  // Required
  "prompt": "string",              // Required
  "memory": "boolean",             // Optional, default: false
  "conversation_number": "integer" // Optional, required if memory=true
}
```

**Response Format:**
```json
{
  "username": "string",
  "data": "string" // Inference response
}
```

## Database Management

### `/db_request/add_user/`
**Description:** Adds a new user to the database.

**Request Format:**
```json
{
  "username": "string",                // Required
  "password": "string",                // Required
  "prompt_token_number": "integer",    // Optional
  "gen_token_number": "integer",       // Optional
  "gen_token_limit": "integer",        // Optional
  "prompt_token_limit": "integer"      // Optional
}
```

**Response Format:**
```json
{
  "status": "string",
  "message": "string" // Confirmation message
}
```



# API Documentation

## Directory Structure Overview

The main directory named `API` contains essential files and directories for running and managing the application. Below is a detailed description of the structure and functionality of each file and directory.

### Main Directory

- **main.py**
  - This is the main entry point for running the application. Executing this file starts the application.

- **cluster_config.yaml**
  - Stores configuration settings for the application, including port numbers, admin passwords, and other critical configurations.

- **chat_bot_db.db**
  - SQLite database that stores user information, including:
    - Username
    - Enabled/Disabled status
    - Role
    - Token limit
    - Other user-specific data

- **memory.db**
  - SQLite database used for storing user chat histories with the Language Learning Model (LLM). It uses a key to retrieve the associated chat history.

### app Directory

The `app` directory contains the core logic for API management, database interaction, and logging configuration.

- **APIs.py**
  - Defines the API formats for inference, VLLM building, and database interactions.

- **database.py**
  - Contains functions for interacting with the `chat_bot_db.db` database.

- **__init__.py**
  - Routes incoming API requests to the appropriate services such as VLLM build, LLM inference, or database interactions.

- **logging_config.py**
  - Sets up the logger for the application, configuring how logs are managed and stored.

### Routers Directory

The `routers` directory contains specific modules for handling different types of API requests.

- **vllm_build.py**
  - Handles requests for building the VLLM engine. It starts a container, initiates the VLLM engine, and returns the endpoints for inference.

- **OpenAI_API.py**
  - Receives inference requests compatible with the OpenAI API.

- **inference.py**
  - Receives inference requests that are not compatible with the OpenAI API but offer additional features such as memory chat.

- **db_functions.py**
  - Processes database requests and returns the corresponding information from `chat_bot_db.db`.

- **login_router.py**
  - Generates authentication tokens for users and returns them.

#### Utils Directory in Routers

The `utils` directory within `routers` contains backend files for database and VLLM operations.

- **backend_database.py**
  - Defines the structure and functions for interacting with `chat_bot_db.db`, including reading and writing data.

- **backend_vLLM.py**
  - Contains functions for LLM inference and pre/post-processing of LLM inference requests.

## Detailed File Descriptions

### main.py

This file initializes and runs the application. It imports necessary modules, sets up configurations from `cluster_config.yaml`, and starts the web server.

### cluster_config.yaml

This YAML file stores critical configuration settings for the application. These settings include:
- Port number for the server
- Admin passwords
- Other configurable parameters

### chat_bot_db.db

An SQLite database that holds user information. The database schema includes tables for:
- Usernames
- Account status (enabled/disabled)
- User roles
- Token limits
- Additional user-specific data

### memory.db

An SQLite database designed to store user chat histories with the LLM. Each chat history is associated with a unique key, allowing for efficient retrieval.

### app/APIs.py

Defines the format and structure of various APIs provided by the application. This includes:
- Inference APIs
- VLLM build APIs
- Database interaction APIs

### app/database.py

Contains functions to interact with the `chat_bot_db.db`. These functions handle:
- Reading user data
- Writing user data
- Updating user data
- Deleting user data

### app/__init__.py

Routes incoming API requests to appropriate modules. Depending on the request type, it directs the call to either VLLM build, LLM inference, or database interaction services.

### app/logging_config.py

Configures the logging framework for the application. This includes setting log levels, log formats, and log file locations.

### routers/vllm_build.py

Handles API requests for building the VLLM engine. It starts the necessary containers, initiates the VLLM engine, and provides endpoints for inference.

### routers/OpenAI_API.py

Receives and processes inference requests that are compatible with the OpenAI API.

### routers/inference.py

Handles inference requests that are not OpenAI API compatible but offer additional features, such as memory chat capabilities.

### routers/db_functions.py

Processes database-related requests and returns relevant information from `chat_bot_db.db`.

### routers/login_router.py

Generates and returns authentication tokens for users, allowing secure access to the application's services.

### routers/utils/backend_database.py

Defines the structure and provides functions for interacting with `chat_bot_db.db`. This includes:
- Reading data from the database
- Writing data to the database
- Updating existing records
- Deleting records

### routers/utils/backend_vLLM.py

Contains functions for handling LLM inference requests. It includes pre-processing and post-processing of these requests to ensure accurate and efficient responses.

---

This documentation provides an overview of the application's structure and the role of each component. For further details on specific functions and their implementations, refer to the respective source files.

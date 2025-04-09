# Kubectl NLP Service

A FastAPI-based service that accepts natural language queries and uses an LLM (like OpenAI's GPT) to generate corresponding `kubectl` commands. It can optionally execute these commands.

## Features

*   FastAPI web framework
*   LangChain integration for LLM interaction
*   Input sanitization and basic command safety checks
*   Optional command execution via `asyncio.create_subprocess_exec`
*   Caching for LLM responses (`cachetools`)
*   Rate limiting (`slowapi`)
*   API Key authentication
*   Configurable via environment variables
*   Health check endpoint (`/health`)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url> 
    cd <repository-directory> 
    ```
    *(Replace `<repository-url>` and `<repository-directory>` with the actual URL and directory name)*

2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    *   Copy `.env-sample` to a new file named `.env`.
    *   Edit `.env` and provide values for:
        *   `OPENAI_API_KEY`: Your OpenAI API key (Required).
        *   `API_AUTH_KEY`: A secret key for API authentication (defaults to `test-key` in sample, **change for production**).
        *   `OPENAI_MODEL`: (Optional) Specify the OpenAI model (defaults to `gpt-3.5-turbo`).
        *   `OPENAI_BASE_URL`: (Optional) Use if you need a custom OpenAI-compatible API endpoint.

## Running the Service

Start the server using Uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
*(Note: Host and port can be configured via `HOST` and `PORT` environment variables)*

## API Usage

### Endpoint: `POST /kubectl-command`

*   **Authentication:** Requires an `X-API-Key` header matching the `API_AUTH_KEY` environment variable (if set).
*   **Request Body:**
    ```json
    {
      "query": "natural language query for kubectl",
      "execute": false
    }
    ```
    *   `query` (string, required): The natural language request.
    *   `execute` (boolean, optional, default: `false`): Set to `true` to attempt execution of the generated command.
*   **Example Request (using curl):**
    ```bash
    curl -X POST "http://localhost:8000/kubectl-command" \
         -H "Content-Type: application/json" \
         -H "X-API-Key: your-api-key" \
         -d '{"query": "list all deployments in default namespace", "execute": false}'
    ```
*   **Response Body:**
    ```json
    {
      "kubectl_command": "kubectl get deployments -n default",
      "execution_result": null, // or command output if execute=true and successful
      "execution_error": null, // or error message if execute=true and failed
      "from_cache": false // true if the command was retrieved from cache
    }
    ```

### Endpoint: `GET /health`

*   Returns `{"status": "healthy"}`.

## Configuration

The service can be configured using the following environment variables (set in a `.env` file or directly in the environment):

*   `OPENAI_API_KEY`: (Required) Your OpenAI API key.
*   `API_AUTH_KEY`: Secret key for API access. If not set, authentication is disabled.
*   `OPENAI_MODEL`: OpenAI model to use (default: `gpt-3.5-turbo`).
*   `OPENAI_BASE_URL`: Custom base URL for OpenAI-compatible APIs.
*   `CACHE_MAXSIZE`: Max items in cache (default: 100).
*   `CACHE_TTL`: Cache item time-to-live in seconds (default: 300).
*   `LLM_TIMEOUT`: Timeout for LLM requests in seconds (default: 60).
*   `EXECUTION_TIMEOUT`: Timeout for command execution in seconds (default: 30).
*   `RATE_LIMIT`: Request rate limit (default: `10/minute`).
*   `LOG_LEVEL`: Logging level (default: `INFO`).
*   `PORT`: Port to run the server on (default: 8000).
*   `HOST`: Host address to bind the server to (default: `0.0.0.0`).

# Kubectl NLP Service

A FastAPI service that converts natural language queries into kubectl commands and optionally executes them.

## Features

- Natural language to kubectl command conversion using OpenAI
- Command execution with safety validation
- Caching for improved performance
- Rate limiting
- Prometheus metrics
- Separate endpoints for command generation and execution

## API Endpoints

### POST /kubectl-command
Generates a kubectl command from natural language.

**Request Body:**
```json
{
  "query": "string"
}
```

**Response:**
```json
{
  "kubectl_command": { "string" },
  "from_cache": "boolean",
  "metadata": {
    "start_time": "string",
    "end_time": "string", 
    "duration_ms": "float",
    "success": "boolean"
  }
}
```

### POST /execute
Executes a provided kubectl command.

**Request Body:**
```json
{
  "execute": "kubectl command"
}
```

**Response:**
```json
{
  "kubectl_command": "string",
  "execution_result": {},
  "execution_error": {},
  "metadata": {
    "start_time": "string",
    "end_time": "string",
    "duration_ms": "float",
    "success": "boolean"
  }
}
```

## Setup

1. Clone the repository
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy .env-sample to .env and configure:
   ```bash
   cp .env-sample .env
   ```
5. Start the server:
   ```bash
   python app.py
   ```

## Configuration

See `.env-sample` for all available environment variables. Key variables:

- `API_AUTH_KEY`: Required API key for authentication
- `OPENAI_API_KEY`: Required OpenAI API key
- `RATE_LIMIT`: API rate limit (default: 10/minute)
- `CACHE_TTL`: Cache duration in seconds (default: 300)

## Usage

1. Generate a command:
   ```bash
   curl -X POST http://localhost:8000/kubectl-command \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your_api_key" \
     -d '{"query":"list all pods"}'
   ```

2. Execute a command:
   ```bash
   curl -X POST http://localhost:8000/execute \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your_api_key" \ 
     -d '{"execute":"kubectl get pods"}'
   ```

## Architecture

![Architecture Diagram](./assets/diagram.svg)

The system follows a clean architecture with these main components:

1. **API Layer**: FastAPI endpoints handling HTTP requests
2. **Service Layer**: Business logic for command generation and execution
3. **LLM Integration**: OpenAI API for natural language processing
4. **Caching Layer**: TTLCache for performance optimization
5. **Execution Layer**: Async subprocess for safe command execution

## Recent Changes

- Separated command generation and execution into distinct endpoints
- Updated LLM chain implementation to use RunnableSequence
- Added comprehensive environment variable documentation
- Improved error handling and validation

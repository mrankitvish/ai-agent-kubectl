import os
import asyncio
import logging
import time
import datetime
import shlex
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request, Response, Header, status
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from cachetools import TTLCache, cached
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_fastapi_instrumentator import Instrumentator # Added for Prometheus

# --- Configuration ---
load_dotenv()

# Environment Variables with Defaults
API_AUTH_KEY = os.getenv("API_AUTH_KEY") # Required, no default
CACHE_MAXSIZE = int(os.getenv("CACHE_MAXSIZE", "100"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "300")) # seconds
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60")) # seconds
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "30")) # seconds
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not API_AUTH_KEY:
    logger.warning("API_AUTH_KEY environment variable not set. API authentication is disabled.")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables. Application will likely fail.")
    # Consider raising an exception here if the key is absolutely critical at startup
    # raise ValueError("OPENAI_API_KEY not found in environment variables")

# --- LLM Setup ---
prompt_template_str = """
You are a Kubernetes CLI specialist.
When given a user request, output exactly one valid, single-line `kubectl` command that fulfils it.
Do not include comments, explanations, or shell operators (`;`, `&&`, `||`, (```) etc.).
Only output the command itself, nothing else.
User Request: {query}
Kubectl Command:
"""

# Basic input sanitization (can be expanded)
def sanitize_query(query: str) -> str:
    """
    Normalize multi-line query to a single line and remove excessive whitespace.
    """
    # Replace newlines and tabs with spaces
    normalized = query.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Collapse multiple spaces into one
    normalized = ' '.join(normalized.split())
    return normalized.strip()


# Basic command validation (can be expanded)
def is_safe_kubectl_command(command: str) -> bool:
    """Checks if the command starts with 'kubectl' and avoids obvious shell risks."""
    command = command.strip()
    if not command.startswith("kubectl "):
        logger.warning(f"Generated command does not start with 'kubectl ': {command}")
        return False
    # Basic check for common shell metacharacters - might need refinement
    if any(char in command for char in [';', '&&', '||', '`', '$', '(', ')', '<', '>']):
        logger.warning(f"Generated command contains potentially unsafe characters: {command}")
        return False
    # Check if shlex can parse it without errors (helps catch unclosed quotes)
    try:
        shlex.split(command)
    except ValueError as e:
        logger.warning(f"Generated command failed shlex parsing: {command} - Error: {e}")
        return False
    return True

class KubectlOutputParser(StrOutputParser):
    """
    Parses the LLM output (expected to be a string command) and validates it
    for basic safety before returning.
    """
    def parse(self, text: str) -> str:
        # Use the parent StrOutputParser to get the raw string output
        command = super().parse(text).strip()
        # Remove any code block markers (```) if present
        if command.startswith('```') and command.endswith('```'):
            command = command[3:-3].strip()
        # Apply safety checks
        if not is_safe_kubectl_command(command):
            raise ValueError(f"Generated command failed safety checks: {command}")
        return command

try:
    template = PromptTemplate(input_variables=["query"], template=prompt_template_str)
    llm_kwargs = {
        "temperature": 0,
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
        "request_timeout": LLM_TIMEOUT, # Add timeout to the LLM call itself
    }
    if OPENAI_BASE_URL:
        llm_kwargs["base_url"] = OPENAI_BASE_URL

    llm = ChatOpenAI(**llm_kwargs)
    chain = template | llm | KubectlOutputParser()
except Exception as e:
    logger.exception("Failed to initialize LangChain components.")
    # Decide if the app should exit or continue in a degraded state
    chain = None # Ensure chain is None if setup fails

# --- Caching Setup ---
cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)

# --- Rate Limiting Setup ---
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

# --- FastAPI App Initialization ---
app = FastAPI(title="Kubectl NLP Service", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# --- Prometheus Instrumentation ---
# Expose /metrics endpoint
Instrumentator().instrument(app).expose(app)

# --- API Key Authentication ---
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if not API_AUTH_KEY: # If no key is set, disable auth
        logger.debug("API key auth disabled.")
        return
    if not x_api_key:
        logger.warning("Missing X-API-Key header.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key header")
    if x_api_key != API_AUTH_KEY:
        logger.warning("Invalid API Key received.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    logger.debug("API key verified.")

# --- Request Body Schema ---
class Query(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language query for kubectl.")

class ExecuteRequest(BaseModel):
    execute: str = Field(..., description="kubectl command to execute.")

# --- Response Body Schema ---
class ExecutionMetadata(BaseModel):
    start_time: str
    end_time: str
    duration_ms: float
    success: bool
    error_type: Optional[str] = None
    error_code: Optional[str] = None

class CommandResponse(BaseModel):
    kubectl_command: str
    execution_result: Optional[Dict[str, Any]] = None
    execution_error: Optional[Dict[str, Any]] = None
    from_cache: bool = False
    metadata: ExecutionMetadata

# --- Helper Functions ---
async def run_llm_chain_async(query: str) -> str:
    """Runs the LLM chain asynchronously with timeout."""
    if not chain:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM Chain not initialized")
    try:
        # Use the new RunnableSequence interface with async invocation
        command = await asyncio.wait_for(
            chain.ainvoke({"query": query}),
            timeout=LLM_TIMEOUT
        )
        logger.info(f"LLM generated command for query '{query}': {command}")
        return command
    except asyncio.TimeoutError:
        logger.error(f"LLM chain timed out after {LLM_TIMEOUT}s for query: {query}")
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="LLM request timed out")
    except ValueError as ve: # Catch validation errors from parser
         logger.error(f"LLM generated unsafe command: {ve}")
         raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"LLM generated unsafe command: {ve}")
    except Exception as e:
        logger.exception(f"Error running LLM chain for query '{query}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing query with LLM: {e}")

# Removed @cached decorator here. Caching is handled in the endpoint.
async def get_command_from_llm(sanitized_query: str) -> str:
    """Wrapper for the LLM chain call (no longer directly cached)."""
    logger.debug(f"Calling LLM for query: {sanitized_query}")
    return await run_llm_chain_async(sanitized_query)

async def execute_command_async(command: str) -> Dict[str, Any]:
    """Executes the command asynchronously and securely."""
    start_time = datetime.datetime.utcnow().isoformat()
    start_ts = time.time()
    logger.info(f"Attempting to execute command: {command}")
    try:
        # IMPORTANT: Use shlex.split to prevent shell injection vulnerabilities
        args = shlex.split(command)
        if args[0] != 'kubectl': # Double check
             raise ValueError("invalid_command", "Command does not start with kubectl")

        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=EXECUTION_TIMEOUT)

        end_ts = time.time()
        metadata = {
            "start_time": start_time,
            "end_time": datetime.datetime.utcnow().isoformat(),
            "duration_ms": (end_ts - start_ts) * 1000,
            "success": process.returncode == 0
        }

        result = {"metadata": metadata}
        if process.returncode == 0:
            result_stdout = stdout.decode().strip()
            logger.info(f"Command executed successfully. Output:\n{result_stdout}")
            # Parse kubectl output into structured JSON if possible
            try:
                if "\n" in result_stdout:  # Likely tabular output
                    lines = result_stdout.splitlines()
                    headers = [h.lower() for h in lines[0].split()]
                    items = []
                    for line in lines[1:]:
                        values = line.split()
                        items.append(dict(zip(headers, values)))
                    result["execution_result"] = {"type": "table", "data": items}
                else:
                    result["execution_result"] = {"type": "raw", "data": result_stdout}
            except Exception as parse_err:
                logger.warning(f"Failed to parse kubectl output: {parse_err}")
                result["execution_result"] = {"type": "raw", "data": result_stdout}
        else:
            result_stderr = stderr.decode().strip()
            logger.error(f"Command execution failed with code {process.returncode}. Error:\n{result_stderr}")
            result["execution_error"] = {
                "type": "kubectl_error",
                "code": str(process.returncode),
                "message": result_stderr
            }
            result["metadata"].update({
                "error_type": "kubectl_error",
                "error_code": str(process.returncode)
            })
        return result

    except asyncio.TimeoutError:
        logger.error(f"Command execution timed out after {EXECUTION_TIMEOUT}s: {command}")
        # Attempt to kill the process if it timed out
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=2) # Wait briefly for termination
        except Exception as kill_err:
            logger.error(f"Error terminating timed-out process: {kill_err}")
        return {"execution_error": f"Command execution timed out after {EXECUTION_TIMEOUT}s"}
    except FileNotFoundError:
        logger.error(f"kubectl command not found. Is it installed and in PATH?")
        return {"execution_error": "kubectl command not found"}
    except ValueError as ve: # Catch shlex or validation errors
         logger.error(f"Invalid command for execution: {command} - {ve}")
         return {"execution_error": f"Invalid command format: {ve}"}
    except Exception as e:
        logger.exception(f"Error executing command '{command}': {e}")
        return {"execution_error": f"An unexpected error occurred during execution: {e}"}

# --- API Endpoints ---
@app.post("/kubectl-command",
          response_model=CommandResponse,
          dependencies=[Depends(verify_api_key)],
          summary="Generate and optionally execute a kubectl command from natural language",
          responses={
              200: {"description": "Command generated (and optionally executed)"},
              400: {"description": "Invalid input query"},
              401: {"description": "Unauthorized (Missing or invalid API Key)"},
              422: {"description": "Unsafe command generated by LLM"},
              429: {"description": "Rate limit exceeded"},
              500: {"description": "Internal server error"},
              503: {"description": "Service unavailable (LLM or execution issue)"},
              504: {"description": "Gateway timeout (LLM or execution)"}
          })
@limiter.limit(RATE_LIMIT) # Apply rate limiting here as well
async def get_kubectl_command(q: Query, request: Request, response: Response):
    """
    Takes a natural language query, generates a kubectl command using an LLM,
    validates it, and optionally executes it.
    """
    logger.info(f"Received query: '{q.query}'")
    sanitized_query = sanitize_query(q.query)

    from_cache = False
    command = None
    from_cache = False
    try:
        # Check cache first
        cached_command = cache.get(sanitized_query) # Use get() for non-blocking check
        if cached_command is not None:
            logger.info(f"Cache hit for query: {sanitized_query}")
            command = cached_command
            from_cache = True
        else:
            logger.info(f"Cache miss for query: {sanitized_query}")
            # Call the LLM (no longer cached by decorator)
            command = await get_command_from_llm(sanitized_query)
            # Manually store the result in the cache
            cache[sanitized_query] = command
            logger.debug(f"Stored result in cache for query: {sanitized_query}")

    except HTTPException as http_exc:
        # Re-raise known HTTP exceptions from helpers
        raise http_exc
    except Exception as e:
        # Catch unexpected errors during cache/LLM logic
        logger.exception(f"Unexpected error processing query '{sanitized_query}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error processing request")

    execution_data = {"metadata": {
        "start_time": datetime.datetime.utcnow().isoformat(),
        "end_time": datetime.datetime.utcnow().isoformat(),
        "duration_ms": 0.0,
        "success": True
    }}

    return CommandResponse(
        kubectl_command=command,
        execution_result=execution_data.get("execution_result"),
        execution_error=execution_data.get("execution_error"),
        from_cache=from_cache,
        metadata=execution_data["metadata"]
    )

@app.get("/health",
         summary="Health check endpoint",
         status_code=status.HTTP_200_OK,
         responses={200: {"description": "Service is healthy"}})
async def health_check():
    # Basic health check, can be expanded (e.g., check LLM connectivity)
    return {"status": "healthy"}

@app.post("/execute",
          response_model=CommandResponse,
          dependencies=[Depends(verify_api_key)],
          summary="Execute a kubectl command",
          responses={
              200: {"description": "Command executed successfully"},
              400: {"description": "Invalid command"},
              401: {"description": "Unauthorized (Missing or invalid API Key)"},
              429: {"description": "Rate limit exceeded"},
              500: {"description": "Internal server error"},
              504: {"description": "Gateway timeout (execution)"}
          })
@limiter.limit(RATE_LIMIT)
async def execute_kubectl_command(req: ExecuteRequest, request: Request, response: Response):
    """
    Executes a provided kubectl command.
    """
    logger.info(f"Received execute request for command: '{req.execute}'")
    
    if not is_safe_kubectl_command(req.execute):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Command failed safety checks"
        )
    
    execution_data = await execute_command_async(req.execute)
    
    return CommandResponse(
        kubectl_command=req.execute,
        execution_result=execution_data.get("execution_result"),
        execution_error=execution_data.get("execution_error"),
        from_cache=False,
        metadata=execution_data["metadata"]
    )

# --- Uvicorn Entrypoint ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    log_level_uvicorn = os.getenv("LOG_LEVEL", "info").lower() # Uvicorn uses lowercase

    logger.info(f"Starting Uvicorn server on {host}:{port}")
    # Use standard uvicorn, which includes uvloop if installed via [standard]
    uvicorn.run("app:app", host=host, port=port, reload=False, log_level=log_level_uvicorn)

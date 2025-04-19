from fastapi import APIRouter, Request, Response, status, HTTPException
from fastapi import Depends
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter
from ..schemas import Query, ExecuteRequest, CommandResponse
from ..llm_service import get_command_from_llm, is_safe_kubectl_command
from ..utils import execute_command_async, sanitize_query
from ..auth import verify_api_key
from ..config import limiter, cache, logger, EXECUTION_TIMEOUT
import datetime

router = APIRouter()

@router.post("/kubectl-command",
             response_model=CommandResponse,
             dependencies=[Depends(verify_api_key)],
             summary="Generate kubectl commands from natural language",
             responses={
                 200: {"description": "Commands generated successfully"},
                 400: {"description": "Invalid input query or non-Kubernetes request"},
                 401: {"description": "Unauthorized (Missing or invalid API Key)"},
                 422: {"description": "Unsafe command generated or validation failed"},
                 429: {"description": "Rate limit exceeded"},
                 500: {"description": "Internal server error"},
                 503: {"description": "Service unavailable (LLM issue)"},
                 504: {"description": "Gateway timeout (LLM)"}
             })
@limiter.limit("10/minute")
async def get_kubectl_command(
    q: Query,
    request: Request,
    response: Response
):
    logger.info(f"Received command generation query: '{q.query}'")
    sanitized_query = sanitize_query(q.query)

    from_cache = False
    command_string = None # Store the raw string from LLM/cache
    try:
        cached_commands = cache.get(sanitized_query)
        if cached_commands is not None:
            logger.info(f"Cache hit for query: {sanitized_query}")
            command_string = cached_commands
            from_cache = True
        else:
            logger.info(f"Cache miss for query: {sanitized_query}")
            command_string = await get_command_from_llm(sanitized_query)
            cache[sanitized_query] = command_string # Cache the raw string
            logger.debug(f"Stored result in cache for query: {sanitized_query}")

    except HTTPException as http_exc:
        # Log the specific HTTP exception details
        logger.error(f"HTTPException during command generation for '{sanitized_query}': Status={http_exc.status_code}, Detail={http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error processing query '{sanitized_query}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing request"
        )

    # Split the command string into a list
    command_list = [cmd.strip() for cmd in command_string.splitlines() if cmd.strip()]
    logger.debug(f"Split commands into list: {command_list}")

    # Prepare metadata (execution part is removed from this endpoint)
    metadata = {
        "start_time": datetime.datetime.utcnow().isoformat(), # Placeholder times
        "end_time": datetime.datetime.utcnow().isoformat(),
        "duration_ms": 0.0,
        "success": True, # Indicates generation success
        "error_type": None,
        "error_code": None
    }

    return CommandResponse(
        kubectl_command=command_list, # Return the list
        execution_result=None, # No execution in this endpoint
        execution_error=None,
        from_cache=from_cache,
        metadata=metadata
    )

@router.post("/execute",
             # Keep response model as CommandResponse, but kubectl_command will be a list with one item
             response_model=CommandResponse,
             dependencies=[Depends(verify_api_key)],
             summary="Execute a single kubectl command",
             responses={
                 200: {"description": "Command executed successfully"},
                 400: {"description": "Invalid or unsafe command"},
                 401: {"description": "Unauthorized (Missing or invalid API Key)"},
                 429: {"description": "Rate limit exceeded"},
                 500: {"description": "Internal server error"},
                 504: {"description": "Gateway timeout (execution)"}
             })
@limiter.limit("10/minute") # Consider a different limit for execution?
async def execute_kubectl_command(
    req: ExecuteRequest,
    request: Request,
    response: Response
):
    command_to_execute = req.execute.strip()
    logger.info(f"Received execute request for command: '{command_to_execute}'")

    # Use the updated safety check from llm_service
    if not is_safe_kubectl_command(command_to_execute):
        logger.error(f"Execution rejected: Command failed safety checks: {command_to_execute}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Command failed safety checks. Cannot execute."
        )

    logger.info(f"Executing command: {command_to_execute}")
    execution_data = await execute_command_async(command_to_execute, EXECUTION_TIMEOUT)

    # Ensure metadata reflects execution success/failure
    final_metadata = execution_data.get("metadata", {}) # Get metadata from execution result
    if "execution_error" in execution_data:
         final_metadata["success"] = False # Mark as failed if execution error occurred
         # error_type and error_code should be populated by execute_command_async

    return CommandResponse(
        kubectl_command=[command_to_execute], # Return as a list with one item
        execution_result=execution_data.get("execution_result"),
        execution_error=execution_data.get("execution_error"),
        from_cache=False, # Execution is never cached
        metadata=final_metadata
    )
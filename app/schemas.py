from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class Query(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language query for kubectl.")

class ExecuteRequest(BaseModel):
    execute: str = Field(..., description="kubectl command to execute.") # Keep as single string for /execute

class ExecutionMetadata(BaseModel):
    start_time: str
    end_time: str
    duration_ms: float
    success: bool
    error_type: Optional[str] = None
    error_code: Optional[str] = None

class CommandResponse(BaseModel):
    # Change kubectl_command to always be a list of strings
    kubectl_command: List[str] = Field(..., description="List of generated kubectl commands.")
    execution_result: Optional[Dict[str, Any]] = None
    execution_error: Optional[Dict[str, Any]] = None
    from_cache: bool = False
    metadata: ExecutionMetadata
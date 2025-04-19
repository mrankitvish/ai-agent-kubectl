from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class Query(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language query for kubectl.")

class ExecuteRequest(BaseModel):
    execute: str = Field(..., description="kubectl command to execute.")

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
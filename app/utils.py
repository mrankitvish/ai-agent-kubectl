import asyncio
import time
import datetime
import shlex
from typing import Dict, Any
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def sanitize_query(query: str) -> str:
    """Normalize multi-line query to a single line and remove excessive whitespace."""
    normalized = query.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    normalized = ' '.join(normalized.split())
    return normalized.strip()

async def execute_command_async(command: str, execution_timeout: int) -> Dict[str, Any]:
    """Executes the command asynchronously and securely."""
    start_time = datetime.datetime.utcnow().isoformat()
    start_ts = time.time()
    logger.info(f"Attempting to execute command: {command}")
    
    try:
        args = shlex.split(command)
        if args[0] != 'kubectl':
            raise ValueError("invalid_command", "Command does not start with kubectl")

        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), 
            timeout=execution_timeout
        )

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
            try:
                if "\n" in result_stdout:
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
        logger.error(f"Command execution timed out after {execution_timeout}s: {command}")
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=2)
        except Exception as kill_err:
            logger.error(f"Error terminating timed-out process: {kill_err}")
        return {"execution_error": f"Command execution timed out after {execution_timeout}s"}
    except FileNotFoundError:
        logger.error("kubectl command not found. Is it installed and in PATH?")
        return {"execution_error": "kubectl command not found"}
    except ValueError as ve:
        logger.error(f"Invalid command for execution: {command} - {ve}")
        return {"execution_error": f"Invalid command format: {ve}"}
    except Exception as e:
        logger.exception(f"Error executing command '{command}': {e}")
        return {"execution_error": f"An unexpected error occurred during execution: {e}"}
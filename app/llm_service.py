import logging
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import shlex
import asyncio
from fastapi import HTTPException, status
from .config import logger, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL, LLM_TIMEOUT

# Prompt Template
prompt_template_str = """
You are a Kubernetes CLI specialist.
When given a user request, output exactly one valid, single-line `kubectl` command that fulfils it.
Do not include comments, explanations, or shell operators (`;`, `&&`, `||`, (```) etc.).
Only output the command itself, nothing else.
User Request: {query}
Kubectl Command:
"""

class KubectlOutputParser(StrOutputParser):
    """Parses the LLM output and validates it for basic safety before returning."""
    def parse(self, text: str) -> str:
        command = super().parse(text).strip()
        if command.startswith('```') and command.endswith('```'):
            command = command[3:-3].strip()
        if not is_safe_kubectl_command(command):
            raise ValueError(f"Generated command failed safety checks: {command}")
        return command

# Initialize LLM components
try:
    template = PromptTemplate(input_variables=["query"], template=prompt_template_str)
    llm_kwargs = {
        "temperature": 0,
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
        "request_timeout": LLM_TIMEOUT,
    }
    if OPENAI_BASE_URL:
        llm_kwargs["base_url"] = OPENAI_BASE_URL

    llm = ChatOpenAI(**llm_kwargs)
    chain = template | llm | KubectlOutputParser()
except Exception as e:
    logger.exception("Failed to initialize LangChain components.")
    chain = None

async def run_llm_chain_async(query: str) -> str:
    """Runs the LLM chain asynchronously with timeout."""
    if not chain:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM Chain not initialized"
        )
    try:
        command = await asyncio.wait_for(
            chain.ainvoke({"query": query}),
            timeout=LLM_TIMEOUT
        )
        logger.info(f"LLM generated command for query '{query}': {command}")
        return command
    except asyncio.TimeoutError:
        logger.error(f"LLM chain timed out after {LLM_TIMEOUT}s for query: {query}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LLM request timed out"
        )
    except ValueError as ve:
        logger.error(f"LLM generated unsafe command: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"LLM generated unsafe command: {ve}"
        )
    except Exception as e:
        logger.exception(f"Error running LLM chain for query '{query}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query with LLM: {e}"
        )

async def get_command_from_llm(sanitized_query: str) -> str:
    """Wrapper for the LLM chain call."""
    logger.debug(f"Calling LLM for query: {sanitized_query}")
    return await run_llm_chain_async(sanitized_query)

def is_safe_kubectl_command(command: str) -> bool:
    """Checks if the command starts with 'kubectl' and avoids obvious shell risks."""
    command = command.strip()
    if not command.startswith("kubectl "):
        logger.warning(f"Generated command does not start with 'kubectl ': {command}")
        return False
    if any(char in command for char in [';', '&&', '||', '`', '$', '(', ')', '<', '>']):
        logger.warning(f"Generated command contains potentially unsafe characters: {command}")
        return False
    try:
        shlex.split(command)
    except ValueError as e:
        logger.warning(f"Generated command failed shlex parsing: {command} - Error: {e}")
        return False
    return True
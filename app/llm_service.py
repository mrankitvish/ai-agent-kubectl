import logging
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import shlex
import asyncio
import json # Use json.loads instead of eval for safety
from fastapi import HTTPException, status
from .config import logger, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL, LLM_TIMEOUT

# Reasoning Agent Prompt
reasoning_prompt_str = """
Analyze this user request and determine if it's related to Kubernetes operations.
If unrelated, respond with "UNRELATED". If related, output a valid JSON object (no markdown formatting) with keys:
- "is_kubernetes": boolean
- "refined_query": string (technically precise version of request)
- "requirements": list of strings (implicit requirements like default namespace, read-only)
- "is_destructive": boolean (if operation modifies cluster state)
- "needs_confirmation": boolean (if operation is high-risk)

User Request: {query}
Analysis:
"""

# Command Generator Prompt
command_prompt_str = """
You are a Kubernetes CLI specialist. Given a validated request, generate:
1. Exactly one valid kubectl command per line for simple requests
2. Multiple commands (one per line) for complex tasks
3. Include all required flags, context, and namespaces
4. Default to read-only operations unless specified
5. For destructive operations, include --dry-run=client unless overridden
6. Do NOT include any comments, explanations, or markdown formatting like ```
7. Only output the raw command(s), each on a new line.
8. IMPORTANT: Do NOT use placeholders like <pod-name>, <namespace>, etc. If information is missing, generate the most likely command or ask for clarification via the reasoning agent's output (though this prompt focuses only on command generation).

Validated Request: {refined_query}
Requirements: {requirements}
Kubectl Commands:
"""

class KubectlOutputParser(StrOutputParser):
    """Enhanced Safety Agent with three-tier validation"""
    def parse(self, text: str) -> str:
        logger.debug(f"Raw LLM output for command generation: {text}")

        # Clean and split commands
        text = text.strip()
        # Remove potential markdown code blocks
        if text.startswith('```json') and text.endswith('```'):
             text = text[7:-3].strip()
        elif text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()

        commands = [cmd.strip() for cmd in text.splitlines() if cmd.strip()]
        logger.debug(f"Generated commands before validation: {commands}")

        if not commands:
            logger.error("LLM did not generate any commands.")
            raise ValueError("No valid commands generated")

        validated_commands = []
        for command in commands:
            # Syntax Validation
            if not command.startswith("kubectl "):
                logger.error(f"Invalid command syntax: {command}")
                raise ValueError(f"Invalid command syntax: {command}")

            # Intent Validation - Allow angle brackets for now, rely on shlex later if needed
            # We might need a more sophisticated check later, but placeholders are the current issue
            if any(char in command for char in [';', '&&', '||', '`', '$', '(', ')']): # Removed < >
                logger.error(f"Command contains unsafe characters: {command}")
                raise ValueError(f"Command contains unsafe characters: {command}")

            # Risk Validation
            destructive_verbs = ['delete', 'edit', 'apply', 'patch', 'replace', 'scale']
            is_destructive = any(verb in command.split() for verb in destructive_verbs)

            # Check if it's a destructive command missing dry-run
            if is_destructive and '--dry-run=client' not in command and '--dry-run=server' not in command:
                 # Check if it's explicitly forced (though we might want user confirmation later)
                 if '--force' not in command:
                     command += " --dry-run=client"
                     logger.warning(f"Added --dry-run=client to destructive command: {command}")

            validated_commands.append(command)


        final_output = '\n'.join(validated_commands)
        logger.debug(f"Validated commands output: {final_output}")
        return final_output

# Initialize LLM components
try:
    reasoning_template = PromptTemplate(input_variables=["query"], template=reasoning_prompt_str)
    command_template = PromptTemplate(input_variables=["refined_query", "requirements"], template=command_prompt_str)

    llm_kwargs = {
        "temperature": 0,
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
        "request_timeout": LLM_TIMEOUT,
    }
    if OPENAI_BASE_URL:
        llm_kwargs["base_url"] = OPENAI_BASE_URL

    llm = ChatOpenAI(**llm_kwargs)
    reasoning_chain = reasoning_template | llm | StrOutputParser()
    command_chain = command_template | llm | KubectlOutputParser()
except Exception as e:
    logger.exception("Failed to initialize LangChain components.")
    reasoning_chain = None
    command_chain = None

async def analyze_query(query: str) -> Dict[str, Any]:
    """Reasoning Agent: Validates Kubernetes relevance and refines query"""
    if not reasoning_chain:
        logger.error("Reasoning chain not initialized during analyze_query call.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reasoning chain not initialized"
        )

    logger.debug(f"Analyzing query: {query}")
    try:
        analysis_raw = await asyncio.wait_for(
             reasoning_chain.ainvoke({"query": query}),
             timeout=LLM_TIMEOUT
        )
        logger.debug(f"Raw analysis from LLM: {analysis_raw}")

        # Strip markdown fences before parsing JSON
        analysis_text = analysis_raw.strip()
        if analysis_text.startswith('```json') and analysis_text.endswith('```'):
             analysis_text = analysis_text[7:-3].strip()
        elif analysis_text.startswith('```') and analysis_text.endswith('```'):
            analysis_text = analysis_text[3:-3].strip()

        if analysis_text.upper() == "UNRELATED":
            logger.info(f"Query determined as UNRELATED: {query}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query is not related to Kubernetes operations"
            )

        # Use json.loads for safer parsing
        analysis_json = json.loads(analysis_text)
        logger.debug(f"Parsed analysis JSON: {analysis_json}")
        return analysis_json

    except json.JSONDecodeError as json_err:
         logger.error(f"Failed to parse reasoning analysis JSON: {json_err}. Raw text after stripping: {analysis_text}")
         raise HTTPException(
             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
             detail=f"Failed to parse reasoning analysis from LLM: {json_err}"
         )
    except asyncio.TimeoutError:
         logger.error(f"Reasoning chain timed out for query: {query}")
         raise HTTPException(
             status_code=status.HTTP_504_GATEWAY_TIMEOUT,
             detail="Reasoning request timed out"
         )
    except Exception as e:
        logger.exception(f"Error during query analysis for '{query}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing query: {e}"
        )


async def generate_commands(refined_query: str, requirements: List[str]) -> str:
    """Command Generator: Creates production-ready kubectl commands"""
    if not command_chain:
        logger.error("Command chain not initialized during generate_commands call.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Command chain not initialized"
        )

    logger.debug(f"Generating commands for refined query: {refined_query} with requirements: {requirements}")
    try:
        commands = await asyncio.wait_for(
            command_chain.ainvoke({
                "refined_query": refined_query,
                "requirements": requirements
            }),
            timeout=LLM_TIMEOUT
        )
        logger.debug(f"Commands generated and validated: {commands}")
        return commands
    except asyncio.TimeoutError:
         logger.error(f"Command generation chain timed out for query: {refined_query}")
         raise HTTPException(
             status_code=status.HTTP_504_GATEWAY_TIMEOUT,
             detail="Command generation request timed out"
         )
    except ValueError as ve: # Catch validation errors from KubectlOutputParser
         logger.error(f"Command validation failed: {ve}")
         raise HTTPException(
             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
             detail=f"Generated command failed validation: {ve}"
         )
    except Exception as e:
        logger.exception(f"Error during command generation for '{refined_query}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating commands: {e}"
        )

async def get_command_from_llm(sanitized_query: str) -> str:
    """Orchestrates the three-agent workflow"""
    logger.info(f"Processing query with multi-agent workflow: {sanitized_query}")

    # Step 1: Reasoning Agent
    logger.debug("Invoking Reasoning Agent...")
    analysis = await analyze_query(sanitized_query)
    if not analysis.get("is_kubernetes"):
        # This case should be handled within analyze_query now, but double-check
        logger.warning("Analysis marked non-kubernetes but didn't raise exception.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query is not related to Kubernetes"
        )
    logger.debug("Reasoning Agent completed.")

    # Step 2: Command Generator
    logger.debug("Invoking Command Generator...")
    commands = await generate_commands(
        analysis["refined_query"],
        analysis.get("requirements", []) # Use .get for safety
    )
    logger.debug("Command Generator completed.")

    # Step 3: Safety Agent (handled by KubectlOutputParser within command_chain)
    logger.info(f"Successfully generated commands for '{sanitized_query}': {commands}")
    return commands

def is_safe_kubectl_command(command: str) -> bool:
    """Legacy safety check (maintained for backward compatibility with /execute endpoint)"""
    # This check might need refinement depending on how strict we want the /execute endpoint to be
    try:
        # Use a temporary parser instance for this check
        temp_parser = KubectlOutputParser()
        temp_parser.parse(command)
        logger.debug(f"Legacy safety check passed for command: {command}")
        return True
    except ValueError as e:
        logger.warning(f"Legacy safety check failed for command '{command}': {e}")
        return False

__all__ = ['get_command_from_llm', 'is_safe_kubectl_command']
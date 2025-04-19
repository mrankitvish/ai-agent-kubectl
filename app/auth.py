from fastapi import HTTPException, Header, Depends, status
from typing import Optional
import logging
from .config import API_AUTH_KEY, logger

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Dependency for API key verification."""
    if not API_AUTH_KEY:
        logger.debug("API key auth disabled.")
        return
    if not x_api_key:
        logger.warning("Missing X-API-Key header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header"
        )
    if x_api_key != API_AUTH_KEY:
        logger.warning("Invalid API Key received.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    logger.debug("API key verified.")
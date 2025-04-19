import os
import logging
from dotenv import load_dotenv
from cachetools import TTLCache
from slowapi import Limiter
from slowapi.util import get_remote_address

# Load environment variables
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

# Logging Setup
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not API_AUTH_KEY:
    logger.warning("API_AUTH_KEY environment variable not set. API authentication is disabled.")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables. Application will likely fail.")

# Cache Setup
cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)

# Rate Limiting Setup
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
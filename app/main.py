from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from .config import limiter, logger
from .auth import verify_api_key
from .routers import commands, health

app = FastAPI(title="Kubectl NLP Service", version="1.0.0")

# Setup middleware and exception handlers
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Include routers
app.include_router(commands.router)
app.include_router(health.router)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn
    from .config import PORT, HOST, LOG_LEVEL
    
    logger.info(f"Starting Uvicorn server on {HOST}:{PORT}")
    uvicorn.run("app.main:app", 
                host=HOST, 
                port=PORT, 
                reload=False, 
                log_level=LOG_LEVEL.lower())
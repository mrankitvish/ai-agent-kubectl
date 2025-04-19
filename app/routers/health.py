from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/health",
         summary="Health check endpoint",
         status_code=status.HTTP_200_OK,
         responses={200: {"description": "Service is healthy"}})
async def health_check():
    return {"status": "healthy"}
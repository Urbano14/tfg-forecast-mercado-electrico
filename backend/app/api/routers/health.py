from fastapi import APIRouter

router = APIRouter(tags=["health"])

#comprobar que la API está funcionando.

@router.get("/health")
def health_check():
    return {"status": "ok"}
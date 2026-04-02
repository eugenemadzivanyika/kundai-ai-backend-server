from fastapi import APIRouter

from services.health_service import get_health_status

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check():
    return get_health_status()

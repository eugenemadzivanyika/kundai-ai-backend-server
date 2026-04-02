from fastapi import APIRouter

from services.content_service import generate_content

router = APIRouter(prefix="/content", tags=["content"])


@router.post("/generate")
def generate(payload: dict):
    return generate_content(payload)

from fastapi import APIRouter

from services.ocr_service import perform_ocr

router = APIRouter(prefix="/ocr", tags=["ocr"])


@router.post("/extract")
async def extract_text(payload: dict):
    return await perform_ocr(payload)

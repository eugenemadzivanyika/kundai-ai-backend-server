import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.gemini_ocr_service import perform_gemini_ocr
from services.ocr_service import perform_ocr

router = APIRouter(prefix="/ocr", tags=["ocr"])

ALLOWED_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp",
    "image/gif", "image/tiff", "application/pdf",
}


@router.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    mime = file.content_type or "application/octet-stream"
    if mime not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {mime}. Accepted: images and PDF.",
        )

    suffix = os.path.splitext(file.filename or "upload")[1] or ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        if os.getenv("GEMINI_API_KEY"):
            result = await perform_gemini_ocr(tmp_path, mime)
        else:
            result = await perform_ocr(tmp_path, mime)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return result

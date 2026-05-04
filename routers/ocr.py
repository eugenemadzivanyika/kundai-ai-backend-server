import json
import os
import tempfile
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.gemini_ocr_service import perform_gemini_ocr, perform_gemini_ocr_batch
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


@router.post("/extract-batch")
async def extract_text_batch(
    files: List[UploadFile] = File(...),
    questions: Optional[str] = Form(None),
):
    """Accept multiple images/PDFs and transcribe them all in one Gemini call.
    Optionally accepts a JSON-serialised questions array for question-aware mapping.
    """
    for f in files:
        mime = f.content_type or "application/octet-stream"
        if mime not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {mime}. Accepted: images and PDF.",
            )

    uploads = [
        (await f.read(), f.content_type or "image/jpeg", f.filename or "upload")
        for f in files
    ]

    parsed_questions: Optional[list] = None
    if questions:
        try:
            parsed_questions = json.loads(questions)
        except json.JSONDecodeError:
            pass  # bad JSON — proceed without question-aware mapping

    if os.getenv("GEMINI_API_KEY"):
        return await perform_gemini_ocr_batch(uploads, parsed_questions)

    # Fallback: run each file individually through the non-Gemini OCR service
    results = []
    for content, mime, filename in uploads:
        suffix = os.path.splitext(filename)[1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            r = await perform_ocr(tmp_path, mime)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        results.append(r)

    return {"files": results}

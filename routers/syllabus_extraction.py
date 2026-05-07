from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.syllabus_extraction_service import extract_attributes_from_syllabus
from utils.logger import log_error

router = APIRouter(
    prefix="/syllabus-extraction",
    tags=["Syllabus Attribute Extraction"],
)


class ExtractionRequest(BaseModel):
    file_content_b64: str   # base64-encoded syllabus file
    mime_type: str          # e.g. "application/pdf"


@router.post("/extract")
async def extract_attributes(payload: ExtractionRequest):
    """
    Extract curriculum attributes from a syllabus document using Gemini vision.
    Returns a list of attribute objects compatible with the CourseAttribute schema.
    """
    try:
        attributes = await extract_attributes_from_syllabus(
            file_content_b64=payload.file_content_b64,
            mime_type=payload.mime_type,
        )
        return {"attributes": attributes, "count": len(attributes)}
    except HTTPException:
        raise
    except Exception as exc:
        log_error(f"POST /syllabus-extraction/extract - {exc}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(exc)}")

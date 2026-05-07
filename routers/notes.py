from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.notes_service import generate_notes

router = APIRouter(prefix="/notes", tags=["Notes"])


class NotesRequest(BaseModel):
    subject_id: str
    topic: str
    attribute_name: str
    level: str
    student_profile: Optional[dict] = None


@router.post("/generate")
async def generate_notes_endpoint(body: NotesRequest):
    try:
        return await generate_notes(
            subject_id=body.subject_id,
            topic=body.topic,
            attribute_name=body.attribute_name,
            level=body.level,
            student_profile=body.student_profile,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

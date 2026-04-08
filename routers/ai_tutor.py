from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from services.ai_tutor_service import AiTutorService

router = APIRouter(prefix="/ai-tutor", tags=["AI Tutor"])
ai_tutor_service = AiTutorService()

class CoachingContext(BaseModel):
    goal: Optional[str] = None
    step: Optional[str] = None
    canvas: Optional[str] = None
    mode: str = "socratic"

class CoachingRequest(BaseModel):
    message: str
    history: List[Dict] = []
    context: CoachingContext

@router.post("/coaching")
async def get_coaching(payload: CoachingRequest):
    try:
        guidance = await ai_tutor_service.generate_coaching_response(
            message=payload.message,
            history=payload.history,
            context=payload.context.dict()
        )
        return {"guidance": guidance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from services.ai_tutor_service import AiTutorService

router = APIRouter(prefix="/ai-tutor", tags=["AI Tutor"])
ai_tutor_service = AiTutorService()


class ExitCheckpoint(BaseModel):
    expectedLogic: Optional[str] = None
    isPassed: Optional[bool] = False


class StepData(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    type: Optional[str] = None
    exitCheckpoint: Optional[ExitCheckpoint] = None


class CognitiveProfile(BaseModel):
    strengths: Optional[List[Dict]] = []
    deficiencies: Optional[List[Dict]] = []
    suggestedTutorApproach: Optional[str] = None


class CoachingContext(BaseModel):
    goal: Optional[str] = None
    step_data: Optional[StepData] = None
    profile: Optional[CognitiveProfile] = None
    canvas: Optional[str] = None
    mode: str = "socratic"


class CoachingRequest(BaseModel):
    message: str
    history: List[Dict] = []
    context: CoachingContext
    subject_id: Optional[str] = None


class IntroductionRequest(BaseModel):
    profile: CognitiveProfile
    plan: Dict  # { missions: [{ task, objective, steps: [{ title, type, ... }] }] }
    subject_id: Optional[str] = None


@router.post("/introduce")
async def get_plan_introduction(payload: IntroductionRequest):
    try:
        result = await ai_tutor_service.generate_plan_introduction(
            profile=payload.profile.dict(),
            plan=payload.plan,
            subject_id=payload.subject_id,
        )
        return result  # { "introduction": "..." }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/coaching")
async def get_coaching(payload: CoachingRequest):
    try:
        result = await ai_tutor_service.generate_coaching_response(
            message=payload.message,
            history=payload.history,
            context=payload.context.dict(),
            subject_id=payload.subject_id,
        )
        return result  # { "guidance": "...", "checkpointPassed": bool }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
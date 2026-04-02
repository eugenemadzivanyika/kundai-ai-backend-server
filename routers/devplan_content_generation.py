from fastapi import APIRouter
from services.devPlan_content_generation_service import generate_mastery_quiz, generate_personalized_theory, generate_practice_set

router = APIRouter(prefix="/devPlan-content-gen", tags=["Content Generation"])

@router.post("/theory")
async def generate_theory(payload: dict):
    return await generate_personalized_theory(payload)

@router.post("/practice")
async def get_practice(payload: dict):
    return await generate_practice_set(payload)

@router.post("/quiz")
async def get_quiz(payload: dict):
    return await generate_mastery_quiz(payload)
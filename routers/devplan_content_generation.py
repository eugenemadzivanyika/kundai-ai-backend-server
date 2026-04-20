from fastapi import APIRouter
from services.devPlan_content_generation_service import (
    generate_mastery_quiz,
    generate_personalized_theory,
    generate_practice_set,
    generate_unit_challenge,
)

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

@router.post("/unit-challenge")
async def get_unit_challenge(payload: dict):
    """
    Generates MCQ practice questions for a student-facing unit challenge.
    Payload: { attributes, unit_title, subject_name, count, difficulty }
    Returns: { title, questions: [{ id, prompt, options, correctOptionIndex, explanation }] }
    """
    return await generate_unit_challenge(payload)
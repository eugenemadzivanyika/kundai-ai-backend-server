from fastapi import APIRouter
from services.devPlan_content_generation_service import (
    generate_mastery_quiz,
    generate_personalized_theory,
    generate_practice_set,
    generate_unit_challenge,
    generate_subject_challenge,   # Phase 3
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


@router.post("/subject-challenge")
async def get_subject_challenge(payload: dict):
    """
    Generates a personalised subject challenge (Phase 3).

    Payload:
      subject_id    - MongoDB _id of the course
      subject_name  - display name of the subject
      student_attrs - list of { name, attribute_id, description, mastery }
                      where mastery is a float 0–1 from StudentAttribute.currentMastery
      count         - total questions to generate (default 12)
      difficulty    - 'easy' | 'medium' | 'hard' (default 'medium')

    Returns: { title, questions, weak_topic_count }
    Questions are weighted toward the student's lowest-mastery attributes.
    """
    return await generate_subject_challenge(payload)
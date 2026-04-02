from fastapi import APIRouter, HTTPException
from services.assessment_generation_service import generate_syllabus_questions

router = APIRouter(
    prefix="/assessment-gen", 
    tags=["Syllabus Assessment Generator"]
)

@router.post("/generate")
async def create_assessment(payload: dict):
    """
    Endpoint for teachers to generate a full ZIMSEC-aligned assessment.
    Expects: syllabus_context (list of attributes), count, and difficulty.
    """
    try:
        # payload['syllabus_context'] should contain:
        # name, attribute_id, level, description, prerequisites
        return await generate_syllabus_questions(payload)
    except Exception as e:
        print(f"[ASSESSMENT ROUTE ERROR]: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate assessment: {str(e)}"
        )